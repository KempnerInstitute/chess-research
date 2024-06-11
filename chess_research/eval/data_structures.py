from typing import Optional

from dataclasses import dataclass

import chess
import torch
import wandb

# from chess_research.eval.player import Player

info_dict_keys = [
    "game_id",
    "transcript",
    "result",
    "player_one",
    "player_two",
    "player_one_time",
    "player_two_time",
    "player_one_score",
    "player_two_score",
    "player_one_illegal_moves",
    "player_two_illegal_moves",
    "player_one_legal_moves",
    "player_two_legal_moves",
    "player_one_resignation",
    "player_two_resignation",
    "player_one_failed_to_find_legal_move",
    "player_two_failed_to_find_legal_move",
    "game_title",
    "number_of_moves",
    "time_taken",
    "total_moves",
    "temperature",
    "nanogpt_elo",
    "opponent_elo",
]


def create_wandb_table():
    if wandb.run is not None:
        return wandb.Table(columns=info_dict_keys)


@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False


@dataclass
class PlayerStateTracker:
    illegal_moves: int = 0
    legal_moves: int = 0
    resignation: bool = False
    failed_to_find_legal_move: bool = False


@dataclass
class EvaluationParams:
    nanogpt_weight_file: str
    temperature: float
    nanogpt_elo: int
    opponent_elo: int
    num_games: int = 10
    model: Optional[torch.nn.Module] = None
    stockfish_skill_level: int = 1
    randomize_opening_moves: int = 0  # number of random moves to start off with
    log_locally: bool = True
    nanogpt_idx: str = None
    wandb_eval_table: wandb.Table = None
    iter: int = -1
