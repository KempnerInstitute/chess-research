"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --cfg.batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import dataclasses
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed

# -----------------------------------------------------------------------------
from eztils import wlog
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from chess_research import Config, Globals
from chess_research.data.zstd_process import StreamingBlockPGNDataset
from chess_research.eval.evaluation import evaluate
from chess_research.eval.utils import get_ckpt_path
from chess_research.model import GPT, GPTConfig

# -----------------------------------------------------------------------------


def train(cfg: Config):
    ##########################################################
    # various inits, derived attributes, I/O setup
    ##########################################################

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.backend, timeout=timedelta(seconds=3000))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = (
        cfg.gradient_accumulation_steps
        * ddp_world_size
        * cfg.batch_size
        * cfg.block_size
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(cfg.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in cfg.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    def get_dataloader(train: bool):
        return iter(
            torch.utils.data.DataLoader(
                StreamingBlockPGNDataset(
                    cfg.dataset,
                    seed=cfg.seed,
                    high_elo=cfg.high_elo,
                    low_elo=cfg.low_elo,
                    length_gen=cfg.length_gen,
                    elo_condition=cfg.elo_condition,
                    win_conditioning=cfg.win_condition,
                    train=train,
                ),
                num_workers=1,
                batch_size=cfg.batch_size,
            )
        )

    ds_block = get_dataloader(train=True)
    val_ds_block = get_dataloader(train=False)

    def get_batch(split):
        dloader = ds_block if split == "train" else val_ds_block
        data = next(dloader)
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            data = data.pin_memory().to(cfg.device, non_blocking=True)
        else:
            data = data.to(cfg.device)
        return data[:, :-1], data[:, 1:]

    # init these up here, can override if cfg.init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_vocab_size = Globals.meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size}")

    # model init
    model_args = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        bias=cfg.bias,
        vocab_size=None,
        dropout=cfg.dropout,
    )  # start with model_args from command line

    ##########################################################
    # model setup
    ##########################################################

    if cfg.resume_from == "":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        print(f"Resuming training from {cfg.resume_from}")
        assert os.path.exists(
            cfg.resume_from
        ), "Resuming from directory that does not exist!"

        ckpt = f"ckpt_{cfg.resume_iter_num or get_ckpt_path(cfg.resume_from)}.pt"
        ckpt_path = os.path.join(cfg.resume_from, ckpt)
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    # crop down the model block size if desired, using model surgery
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args[
            "block_size"
        ] = cfg.block_size  # so that the checkpoint will have the right value
    model.to(cfg.device)

    ##########################################################
    # GradScaler, optimizer, DDP setup
    ##########################################################

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    if cfg.resume_from != "":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for cfg.warmup_iters steps
        if it < cfg.warmup_iters:
            return cfg.learning_rate * it / cfg.warmup_iters
        # 2) if it > cfg.lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # logging
    if cfg.wandb_log and master_process:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=dataclasses.asdict(cfg),
        )

    ##################################
    # training loop
    ##################################

    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0

    while True:
        # termination conditions
        if iter_num > cfg.max_iters:
            break
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.save_interval == 0 and master_process:
            if cfg.debug:
                losses = {"train": -1.0, "val": -1.0}
            else:
                losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            wlog(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                },
                commit=True,
            )

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "val_loss": losses["val"],
                "train_loss": losses["train"],
                "config": dataclasses.asdict(cfg),
            }
            weight_file = os.path.join(Globals.LOG_DIR, f"ckpt_{iter_num}.pt")
            print(f"saving checkpoint to {weight_file}")
            os.makedirs(Globals.LOG_DIR, exist_ok=True)
            torch.save(checkpoint, weight_file)

            if (
                iter_num != 0
                and (iter_num % (cfg.save_interval * cfg.eval_every_n_saves) == 0)
                and master_process
            ):
                # TODO make this evaluate across multiple GPUs
                assert cfg.eval_job_id == 0, "eval_job_id must be 0 during training"
                assert (
                    cfg.eval_job_total == 1
                ), "eval_job_total must be 1 during training"
                evaluate(
                    cfg=cfg,
                    model=raw_model,
                    n_games=10,
                    skill_levels=[1, 4],
                    iter_num=iter_num,
                )

        if ddp:
            torch.distributed.barrier()

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(cfg.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == cfg.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / cfg.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # if iter_num < 10:
            #     print("Batch")
            #     print(X)
            #     print("y")
            #     print(Y)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    cfg.batch_size * cfg.gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
            wlog(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "a100_bf16_mfu": running_mfu * 100,  # convert to percentage
                },
                commit=True,
            )
        iter_num += 1
        local_iter_num += 1

    # final evaluation
    if master_process:
        evaluate(
            cfg=cfg,
            model=raw_model,
            n_games=cfg.eval_n_games,
            skill_levels=[1, 2, 3, 4, 5, 6],
            iter_num=iter_num,
        )
    torch.distributed.barrier()

    if ddp:
        destroy_process_group()
