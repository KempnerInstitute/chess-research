# chess_research

## Installation

```

need to set gradient accumulation steps to number of gpus

make install
git clone https://huggingface.co/datasets/ezipe/adam-chess-data
mv adam-chess-data/original lichess_hf_dataset
chess_research -c $PWD/config/original.json
```

## Usage

CUDA_VISIBLE_DEVICES=$2 chess_research --config $PWD/config/50M_1000.json --wandb_run_name 50M-High-Elo-$1-No-Elo-Conditioning --high_elo $1

Use json to modify the config:

```
chess_research
```

Reuse an old config file by specifying the path:

```
chess_research --config $PWD/runs/<YYYY-MM-DD>---<HH-MM-SS>/config.json
```

DDP run (model params are duplicated, batch size = original_batch_size \* num_gpus)

```
cd chess_research # the outer chess_research
torchrun --standalone --nproc_per_node=4  chess_research/__init__.py -c $PWD/config/302_1000.json
```

Resume from prior run

```
chess_research --resume_from $PWD/runs/770-High-Elo-2000 --config $PWD/config/707_1000.json
```

Use tmux to run in the background.

# 1. switch dataloader to elo conditioned

# 2. increase batch size and model size

# 3. ddp run
