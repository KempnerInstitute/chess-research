from typing import List, Optional

import dataclasses
import functools
import json
import os
import platform
from collections import defaultdict

import chess.engine
import torch
import wandb
from eztils import wlog
from eztils.run_parallel import calculate_split, prod

from chess_research import Config
from chess_research.eval.data_structures import EvaluationParams, create_wandb_table
from chess_research.eval.glicko2 import GlickoCalc
from chess_research.eval.player import NanoGptPlayer, StockfishPlayer
from chess_research.eval.utils import get_ckpt_path, play_games
from chess_research.model import GPT, GPTConfig


def evaluate(
    cfg: Config,
    model: GPT = None,
    n_games: int = None,
    skill_levels: List = None,
    iter_num: int = -1,
):
    if model is None:
        assert os.path.exists(
            cfg.resume_from
        ), f"Checkpoint {cfg.resume_from} not found for evaluation only"

        def load_model(weight_file):
            print(f"loading model {weight_file}...")
            device = "cuda"
            checkpoint = torch.load(weight_file, map_location=device)
            gptconf = GPTConfig(**checkpoint["model_args"])

            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            model = GPT(gptconf)
            model.load_state_dict(state_dict)
            model = torch.compile(model)
            print("loaded!")
            return model

        resume_iter_num = cfg.resume_iter_num or get_ckpt_path(cfg.resume_from)
        ckpt = f"ckpt_{resume_iter_num}.pt"
        weight_file = os.path.join(cfg.resume_from, ckpt)
        json_filepath = os.path.join(cfg.resume_from, "config.json")
        with open(json_filepath) as file:
            data = json.load(file)
            original_cfg = Config(**data)

        model = load_model(weight_file)

        if wandb.run is None:
            wandb.init(
                project=cfg.wandb_project + "-Eval-Full",
                name=original_cfg.wandb_run_name + "-Eval-Full",
                config=dataclasses.asdict(cfg),
            )
            print("SKILL LEVELS:", cfg.eval_stockfish_levels)

        eval_elo = cfg.eval_default_elo or original_cfg.high_elo
    else:
        weight_file = "/nanogpt"
        eval_elo = cfg.eval_default_elo or cfg.high_elo
        resume_iter_num = None

    conditions = [eval_elo + n * 100 for n in range(-3, 1)]
    elo_conditions = [("#fx+", "#fx+")] + prod(  # nonsense elo for ablation
        conditions, conditions
    )

    all_hparams = []

    if skill_levels is None:
        skill_levels = cfg.eval_stockfish_levels

    print("OPPONENT STOCKFISH SKILL LEVELS:", skill_levels)

    if n_games is None:
        n_games = cfg.eval_n_games

    if iter_num == -1 and resume_iter_num is not None:
        iter_num = resume_iter_num

    if cfg.elo_condition:
        print("ELO CONDITIONS:", elo_conditions)
        all_hparams += prod(skill_levels, elo_conditions, [0.001])

    if cfg.temperature_sampling:
        print("TEMPERATURES:", cfg.eval_temperatures)
        default_elo = eval_elo if cfg.elo_condition else None
        print("DEFAULT ELO", default_elo)
        all_hparams += prod(
            skill_levels,
            [(default_elo, default_elo)],
            cfg.eval_temperatures,
        )

    if len(all_hparams) == 0:
        all_hparams = prod(skill_levels, ((None, None),), [0.001])

    start_index, end_index = calculate_split(
        cfg.eval_job_total,
        len(all_hparams),
        cfg.eval_job_id,
    )

    print(
        "\n\n\n",
        "TOTAL HPARAMS:",
        len(all_hparams),
        "\n",
        "JOB HPARAMS:",
        end_index - start_index,
        "\n\n\n",
    )
    print(
        f"Will play {len(all_hparams)} * {n_games} each = {len(all_hparams) * n_games} games in total across all processes"
    )

    for stockfish_skill_level, (nanogpt_elo, opponent_elo), temp in all_hparams[
        start_index:end_index
    ]:
        play_stockfish_with_nanogpt(
            EvaluationParams(
                nanogpt_weight_file=weight_file,
                temperature=temp,
                nanogpt_elo=nanogpt_elo,
                opponent_elo=opponent_elo,
                model=model,
                stockfish_skill_level=stockfish_skill_level,
                num_games=n_games,
                log_locally=False,
                iter=resume_iter_num,
            )
        )


def play_stockfish_with_nanogpt(params: EvaluationParams):
    assert params.num_games >= 2, "Number of games must be at least 2"

    ############################
    # SETUP
    ############################

    wandb_eval_table = None
    if wandb.run is not None:
        wandb_eval_table = create_wandb_table()
    params.wandb_eval_table = wandb_eval_table

    ############################
    # PLAY GAMES
    ############################

    params.num_games = (
        params.num_games // 2
    )  # divide by 2 because we play half games with each player as the white player

    params.nanogpt_idx = "two"
    info_dicts = play_games(params)

    params.nanogpt_idx = "one"
    info_dicts += play_games(params)

    ############################
    # CALCULATE ELO, LOSSES, WINS, DRAWS
    ############################
    model_name = f"Stockfish_{params.stockfish_skill_level}_NanoGPT"
    eval_analysis_results = {
        f"{model_name}_Losses": 0,
        f"{model_name}_Wins": 0,
        f"{model_name}_Draws": 0,
    }
    score_str = {
        "0": "Losses",
        "1": "Wins",
        "2": "Draws",
    }

    glicko = GlickoCalc(params.stockfish_skill_level)
    for info_dict in info_dicts:
        nanogpt_index = (
            "two" if info_dict["player_one"].startswith("Stockfish") else "one"
        )
        score = info_dict[f"player_{nanogpt_index}_score"]
        if score == "1/2":
            score = "2"
        glicko.glicko2_update(score)
        eval_analysis_results[f"{model_name}_{score_str[score]}"] += 1

    eval_analysis_results[f"{model_name}_Win_Percentage"] = eval_analysis_results[
        f"{model_name}_Wins"
    ] / (len(info_dicts))

    eval_analysis_results[f"{model_name}_Elo_Rating"] = glicko.current_elo

    eval_analysis_results[
        f"{model_name}_Elo_Rating_Deviation"
    ] = glicko.current_deviation

    wlog(
        {
            f"eval_table_{iter=}_{params.temperature=}_{params.nanogpt_elo=}_{params.opponent_elo=}": wandb_eval_table
        },
        commit=True,
    )

    wlog(
        {
            f"eval/{params.temperature=}/{params.nanogpt_elo=}_{params.opponent_elo=}": eval_analysis_results
        },
        commit=True,
    )


if __name__ == "__main__":
    player_ones = ["lichess_16layers_ckpt_no_optimizer.pt"]
    temperature = 0.001
    nanogpt_elo = 1100
    opponent_elo = 1100
    for player in player_ones:
        player_one_recording_name = player  # Because we are using list [player_ones], RECORDING_FILE is overwritten
        for i in range(11):
            num_games = 100
            # player_one = StockfishPlayer(skill_level=-1, play_time=0.1)
            player_one = NanoGptPlayer(
                model_name=player_one_recording_name, model=model
            )
            player_two = StockfishPlayer(skill_level=i, play_time=0.1)
            play_games(
                player_one,
                player_two,
                num_games,
                temperature,
                nanogpt_elo,
                opponent_elo,
                player_one_recording_name=player,
                player_two_recording_name=f"stockfish_{i}",
            )
