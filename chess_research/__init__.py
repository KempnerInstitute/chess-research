"""project_tag"""

""" 
Other global variables
"""
from typing import List

import dataclasses

from dotenv import load_dotenv

load_dotenv()

import os
from argparse import Namespace
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from eztils import datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from rich import print

from chess_research.globals import Globals


def get_version() -> str:
    try:
        return importlib_metadata.version("chess_research")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version


def setup_experiment():
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")

    """SETUP CONFIG"""
    parser = HfArgumentParser(Config)
    parser.add_argument("-c", "--config", type=str)

    conf: Config
    extras: Namespace
    conf, extras = parser.parse_args_into_dataclasses()

    if extras.config is not None:  # parse config file
        (original_conf,) = parser.parse_json_file(extras.config)
        for field_ in dataclasses.fields(original_conf):
            # print(field.name, getattr(original_conf, field.name))
            val = getattr(original_conf, field_.name)
            if isinstance(val, list):
                setattr(original_conf, field_.name, field(default_factory=lambda: val))
        # reinit the parser so that the command line args overwrite the file-specified args
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    # create run dir
    Globals.RUN_DIR = setup_path(Globals.DATA_ROOT / "runs")

    try:
        exp_name = conf.wandb_run_name
    except AttributeError:
        exp_name = ""

    Globals.LOG_DIR = setup_path(Globals.RUN_DIR / exp_name / f"{datestr()}_{exp_name}")
    print(f"LOG DIR: {Globals.LOG_DIR}")

    # symlink repo dir / runs to Globals.run_dir
    if (
        not (Globals.REPO_DIR / "runs").exists()
        and (Globals.REPO_DIR / "runs") != Globals.RUN_DIR
    ):
        print(f'Creating symlink from {Globals.REPO_DIR / "runs"} to {Globals.RUN_DIR}')
        (Globals.REPO_DIR / "runs").symlink_to(Globals.RUN_DIR)

    os.chdir(Globals.LOG_DIR)

    parser.to_json([conf], Globals.LOG_DIR / "config.json")
    return conf


@dataclass
class Config:
    save_interval: int = 20_000
    eval_every_n_saves: int = 1
    log_interval: int = 1
    eval_iters: int = 100  # loss number of batches to calculate

    eval_only: bool = False
    eval_n_games: int = 100
    eval_default_elo: int = 2100
    eval_job_id: int = 0
    eval_job_total: int = 1
    eval_stockfish_levels: List[int] = field(default_factory=lambda: [1, 3, 5])
    eval_temperatures: List[int] = field(
        default_factory=lambda: [
            0.001,
            0.5,
            1,
            1.5,
        ]
    )

    always_save_checkpoint: bool = True
    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "transcendence"
    wandb_run_name: str = "chess-original"
    # data
    resume_from: str = ""
    resume_iter_num: int = None
    dataset: str = "/change/to/your/dataset"
    gradient_accumulation_steps: int = 1
    batch_size: int = 32  # 60 = 70% mem, 70 = 82% mem
    block_size: int = 1023
    # model

    # 50M params gpt-med
    n_layer: int = 16
    n_head: int = 8
    n_embd: int = 512

    # # 350M params gpt-med
    # n_layer: int = 24
    # n_head: int = 16
    # n_embd: int = 1024

    # 774M params gpt-lg
    # n_layer: int = 36
    # n_head: int = 20
    # n_embd: int = 1280

    # 1.5B params gpt-xl (largest reasonable size)
    # n_layer: int = 48
    # n_head: int = 36
    # n_embd: int = 1620

    # # 2.7B params gpt-xxl
    # n_layer: int = 56
    # n_head: int = 50
    # n_embd: int = 2000

    # # 4.6B params gpt-xxl (max size, only works with batch size =1)
    # n_layer: int = 96
    # n_head: int = 50
    # n_embd: int = 2000

    dropout: float = 0.0
    bias: bool = False
    # adamw optimizer
    learning_rate: float = 0.0003
    max_iters: int = 4000000.0
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 50
    lr_decay_iters: int = 200000.0
    min_lr: float = 3e-5
    # DDP settings
    backend: str = "nccl"  # Other options could be 'gloo', etc.
    # system
    device: str = "cuda"  # Other options: 'cpu', 'cuda:0', 'cuda:1', 'mps' etc.
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # Other options: 'float32', 'float16'
    compile: bool = False
    # our changes
    low_elo: int = -1
    high_elo: int = 9999
    win_condition: bool = False
    elo_condition: bool = False

    length_gen: int = None

    temperature: float = 0.01
    # lichess_bot_token: str = os.environ["lichess_bot_token"]
    seed: int = 1337
    debug: bool = False
    temperature_sampling: bool = True


def main():
    conf = setup_experiment()
    if conf.win_condition:
        Globals.meta["vocab_size"] = 35
        Globals.meta["itos"].update(
            {
                32: "W",  # white win
                33: "L",  # white loss
                34: "D",  # draw
            }
        )

        Globals.meta["stoi"].update(
            {
                "W": 32,  # white win
                "L": 33,  # white loss
                "D": 34,  # draw
            }
        )

    print(f"[bold green]Welcome to chess_research v{version}[/]")
    print(conf)

    from eztils.torch import (
        seed_everything,  # install torch first to uncomment this line (by getting `poetry add eztils[torch]`` as a dependency)
    )

    seed_everything(conf.seed)

    if conf.eval_only:
        from chess_research.eval.evaluation import evaluate

        evaluate(conf)
    else:
        from chess_research.train import train

        train(conf)


if __name__ == "__main__":
    main()
