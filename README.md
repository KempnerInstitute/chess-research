![Latent Board State](/figures/latent_board_state_reward_tsne.png)


# Transcendence Chess Research


<div align="center">


[![Build status](https://github.com/ezhang7423/language-control-diffusion/workflows/build/badge.svg?branch=master&event=push)](https://github.com/ezhang7423/chess_research/pulls)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/ezhang7423/language-control-diffusion/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Utility: isort](https://img.shields.io/badge/imports-isort-orange.svg)](https://pycqa.github.io/isort/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/ezhang7423/chess_research/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/ezhang7423/chess_research/releases)
[![License](https://img.shields.io/github/license/ezhang7423/language-control-diffusion)](https://github.com/ezhang7423/chess_research/blob/main/LICENSE)


## Generative Models Can Outperform The Experts That Train Them

</div>

![Low Temperature](/figures/advantage-analysis.png)

An visual of the main principle that we explore for transcendence: low temperature sampling. For more information, please see our [website](https://transcendence.eddie.win).

![nanoGPT](/figures/rating_temp.png)

Ratings of our autoregressive decoder-only transformer, ChessFormer, over several different temperatures. Each model is trained only on games with players up to a certain rating (1000, 1300, 1500), respectively.

## 🚀 Features

This repository contains months of research aimed toward demonstrating the phenomenon we have termed "transcendence". The training and the evaluation code against Stockfish are contained in the [`chess_research`](./chess_research/) directory.

- The [`scripts`](./scripts) directory provides a template for running the training or evaluations desired.

- For figures that we generated, explore the [`analysis`](./analysis/). For code used to generate transcendence line plots above, look at the [`fig-gen`](https://github.com/ezhang7423/fig-gen-transcendence/tree/vincent-transcendence-work/figgen) repository that we wrote.

- The [`./analysis/advantage`](./analysis/advantage) branch dives into how the stockfish engine is used to calculate the reward of each move made in a game. This is actually how the analysis is generated for a game on Lichess.org. Look into this [post](https://www.landonlehman.com/post/2021-01-25-how-to-reproduce-a-lichess-advantage-chart-in-python/) for a better intuition of what was being evaluated here.

- There are a few different config.json files in the [`config`](./config) directory that are ready for immediate use. There is a 50M, 302M, and 707M parameter model with appropriate batch sizes for a H100 80GB gpu.

- [`./eval/player.py`](https://github.com/ezhang7423/chess-research/blob/main/chess_research/eval/player.py) contains the Stockfish and NanoGPT player classes that process the game strings that are passed in and outputs a move based on what the model predicts.

- There is also the integration of the [`glicko2`](https://github.com/fsmosca/glicko2calculator) repository, the method of calculating the elo ratings of players over a series of games. It is simple to use and adjust to your preference.

## 🧪 Experiment Logs

To compare agains, the experiments for the Max_Elo 1000 are logged [here](https://wandb.ai/project-eval/50M-Training/reports/Transcendence-Chess-Research---Vmlldzo4MzAxODA2?accessToken=9r9uih3djihscx3w67h47dfeh9rynd69toc001mr0a9qzqa2cxvie9izlu8yomp1). The evaluations done in these runs are not statistically significant, and are only done to get an idea of how the model is performing. We evaluate for a 100 games against Stockfish levels 1, 3, and 5 each for the final results in the paper.

## Installation

We require [`Conda`](https://docs.conda.io/en/latest/miniconda.html) for this repository. To install, simply run

```bash
$ make install
$ source .vevn/bin/activate
$ # make sure you've installed git-lfs
$ git lfs clone https://huggingface.co/datasets/ezipe/lichess_elo_binned_debug```

This will set up a new conda environment with [`Poetry`](https://python-poetry.org/) as the dependencies manager.

To download models and dataset, we recommend installing [`git lfs`](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) to interface with our [`huggingface`](https://huggingface.co/docs/hub/en/repositories-getting-started) repos that house the data and models. You can find those repos here:

- [50M Parameter Models for Ratings 1000 - 1500](https://huggingface.co/datasets/ezipe/lichess-models/tree/main)
- [Massive Dataset of Chess Games from Lichess.org split by Different Max Ratings](https://huggingface.co/datasets/ezipe/lichess_elo_binned_debug/tree/main)

## Usage

To train your own models from scratch, set up a configuration JSON file that matches your specifications. Here is an example of our [50M parameter model config file](https://github.com/ezhang7423/chess_research/blob/main/config/50M_1000.json).

Next, run the following

```
chess_research --config $PWD/config/your_config.json
```

To use arguments, you can run like the following (this is an example of running a training with a max rating of 1500 and evaluating with different temperatures):

```
chess_research --config $PWD/config/your_config.json --temperature_sampling true --high_elo 1500
```

To resume from a prior run

```
chess_research --resume_from $PWD/runs/50M-High-Elo-1000-No-Elo-Conditioning/2024-04-30---03-05-07_50M-High-Elo-1000-No-Elo-Conditioning --resume_iter_num 100000 --eval_only true
```

Here is an explanation of some useful arguments to look at that we used to run our experiments and their purpose:

- temperature_sampling: if true, will run evaluations with temperatures [0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5], used to denoise the models
- eval_only: if true, will not run the training, only an evaluation of the model passed in against Stockfish (levels 1, 3, 5)
- high_elo: sets the maximum elo ratings of the games that the model will see during train time
- wandb_log: if true, will send the data of the training to a Wandb project that you set up and authenticate, to view your results.
- resume_from: sets the path of which directory to resume the training from. The directory should include the weight files and the config.json necessary to continue training

Distributed Data Parallel (DDP):
DDP is incorporated into the repo to speed up training by running on multiple GPUs in parallel (model params are duplicated, batch size = original_batch_size \* num_gpus).
To run with DDP,

```
torchrun --standalone --nproc_per_node=4  chess_research/__init__.py -c $PWD/config/350_1100_elo_gen.json
```

Make sure to set the number of gradient accumulation steps equal to the number of GPUs. In the example above, we would want 4.

Tips:

- Use tmux to run in the background.
- There is a script for running multiple trainings at different high elos that takes in different configuration files in the scripts directory found [here](https://github.com/ezhang7423/chess_research/blob/cleanup/scripts/train_big.py).
- You can run the above script with ipython or python in the terminal; they will run in the background.

## 🏗️ Development

### Directory Structure

```
CHESS_RESEARCH
├── .devcontainer
├── .empty
├── .github
├── .vscode
├── adam-chess-data
├── chess_research
│   ├── __pycache__
│   ├── data
│   │   ├── __pycache__
│   │   └── zstd_process.py # takes the training dataset and processes it
│   ├── eval
│   │   ├── __pycache__
│   │   ├── data_structures.py # Holds different objects used in utils
│   │   ├── evaluation.py # Loads the model for eval and configures params (temp, elo, skill_lvl, etc.)
│   │   ├── glicko2.py # Modeule for calculating rating given num wins, losses, draws
│   │   ├── player.py # Contians the NanoGPT and Stockfish player classes
│   │   ├── utils.py # Holds all the play game code for evaluating 2 player models.
│   ├── __init__.py # Builds config object from args and runs training or large eval
│   ├── .env.example
│   ├── globals.py # Contains global varaibles
│   ├── model.py # Definition of GPT langauge model
│   ├── train.py # Training code and saves checkpoints to runs
├── config
│   ├── 50M_1000.json # 50M model, defaults to high_elo 1000
│   ├── 302M_1000.json # 302M model, defaults to high_elo 1000
│   ├── 707M_1000.json # 707M model, defaults to high_elo 1000
├── figures # Holds the figures generated for this project
├── scripts
|   ├── train_big.py # Template code for running trainings
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile # Make install script
├── poetry.lock
├── pyproject.toml
├── README.md
├── requirements.txt
```

## 🎯 What's next

The temperature sampling for denoising is only the beginning. The main branch also contains code for an elo conditioning experiment. Moreover, as mentioned above, there were many other experimental settings this project can explore. There are also length_generalization and win_condition configuration options of this repository that we only explored with some preliminary work. Here is a brief summary and explanation of the intention for each setting. Each option is integrated into the main branch.

- `length_gen`: We want to see if the model can perform out of distribution. In this setting, we train the model with games in a dataset of a maximum length instead of a maximum elo rating. Then, during test time, we prompt with already played out games and have it start with a certain number of starting moves that are already played in that game. The ideal outcome we wanted to see is the model performing well even when the games go beyond the maximum length that it was trained on.

- `win_condition`: The idea behind win conditioning is that the result of each game from the dataset that the model sees during train time already tells the model who is going to win with a W or an L prompt that comes before the game string PGN and marks either white will win and black will loss or white will loss and black will win. Then, during test time, we prompt the model by prepending a W to the start of the game state PGN passed in. We want to see if this kind of conditioning can improve the model's performance in terms of rating when put through the evaluation against Stockfish. The ideal outcome here is that the model will perform better given the conditioning of seeing a winning marker 'W' at the start of all the game string PGNs.

- `elo_condition`: It is a configurable setting in the config.json. This setting is very similar to the win_conditioning example: the games that the model sees during training have a prompt that tells the model the elo of the white player and the elo of the black player. Note, the games in the training dataset still have a maximum rating. Then, during test time, the game string PGN are also marked with an elo that is greater than the maximum rating that the model was trained with. We want to see if the model can transcend from the prompting of higher elos.

## 👏 Credits

### Massive Thanks to Adam Karvonen

Adam Karvonen studied the application of language models to playing chess, building on previous ML research, including gpt-3.5-turbo-instruct's ability to play chess at an 1800 Elo level, and Kenneth Li's work on [Emergent World Representations](https://arxiv.org/abs/2210.13382). You can take a look at Adam's work [here](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html). Our work was originally forked from Adam Karvonen's NanoGPT repository.

### Thank you Lichess.org

Lichess gave us many resources by providing the data, giving us calibrated ratings, and allowing us to run games against other pre-established models through the [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) api.
The [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) repository was a huge help in getting up-to-date information on Stockfish and allowing us to build our own bot with the models that we trained. Our [`ChessFormer-Bot`](https://lichess.org/@/ChessFormer-1000) trained with a max rating of 1000 is up and running on Lichess.org and has been performing significantly better against other bots (maia1, maia5, maia9). Additionally, you can challenge our bot to a real-time game yourself at this [link](https://lichess.org/?user=ChessFormer-1000#friend), if you want to see it in action.

This project would have been much harder to complete without these resources and past works, so thank you!
