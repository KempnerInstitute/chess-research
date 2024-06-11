from typing import List

import argparse
import io
import json
from pathlib import Path

import chess
import chess.pgn
import pandas as pd
import torch

from chess_research.eval import sample_many_moves
from chess_research.eval.nanogpt_module import NanoGptPlayer
from chess_research.evaluation import calculate_split, load_model


def generate_distribution(
    player,
    is_black,
    game_id,
    game_moves: list,
    num_moves: int = 100,
    temp: List[float] = None,
) -> str:
    if temp is None:
        temp = [0.001, 0.75, 1]

    board = chess.Board()
    print(f"Generating distribution for game {game_id}...")
    analysis = []

    game_string = "\n\n"
    for i, move in enumerate(game_moves):
        # only generate for given color
        if i % 2 == 0:
            game_string += f"{i//2 + 1}. "

        if (
            not (i % 2) == is_black
        ):  # if we're black, we only will generate for half move clock odd
            analysis.append(None)
            game_string += board.san(move) + " "
            board.push(move)
            continue

        print(f" ->Generating for move {i // 2 + 1}")
        info = sample_many_moves(
            player, board, game_string, num_moves=num_moves, temperatures=temp
        )
        torch.cuda.empty_cache()
        analysis.append(info)
        game_string += board.san(move) + " "
        board.push(move)

    print(f"Done analyzing {game_id}")
    return analysis


def main(weight_file: str, csv_file: str, job_id: int, total_jobs: int) -> None:
    games_info = pd.read_csv(csv_file)
    num_games = len(games_info)
    all_res = []
    start, end = calculate_split(total_jobs, num_games, job_id)
    print("START", start, "END", end)
    print("\n\n\n")
    for i in range(start, end):
        is_black = (
            True if games_info.iloc[i]["player_one"].startswith("Stockfish") else False
        )
        all_res.append(
            json.dumps(
                generate_distribution(
                    NanoGptPlayer(model_name="1000", model=load_model(weight_file)),
                    is_black,
                    i,
                    chess.pgn.read_game(
                        io.StringIO(games_info.iloc[i]["transcript"]),
                    ).mainline_moves(),
                )
            )
        )

    # Write results as new pandas series
    # games_info["all_sampled_moves"] = all_res
    save_path = f"/path/to/saved/csv/{Path(csv_file).stem}_{job_id}.csv"

    torch.save(all_res, save_path)
    print(f"saved to {save_path}")
    # games_info.to_csv(csv_file[:-4] + f"_sampled_moves_{job_id}.csv")


if __name__ == "__main__":
    # make into argparser

    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--weight_file",
        type=str,
        default="/path/to/50M-High-Elo-1000-No-Elo-Conditioning/ckpt_100000.pt",
    )
    argparse.add_argument("--job_id", type=int, default=0)
    argparse.add_argument("--total_jobs", type=int, default=1)
    args = argparse.parse_args()

    csv_file_base = "/path/to/saved/csv/Stockfish_1_vs_ckpt_100000_pt_{{temp}}.csv"
    for temp in ["0_001", "0_75", "1"]:
        # main(weight_file, csv_file)
        csv_file = csv_file_base.replace("{{temp}}", temp)
        main(args.weight_file, csv_file, args.job_id, args.total_jobs)
