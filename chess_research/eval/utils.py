# resume training from a checkpoint.
from typing import Optional, Tuple

import csv
import os
import random
import time
from dataclasses import asdict

import chess

from chess_research.eval.data_structures import (
    EvaluationParams,
    LegalMoveResponse,
    PlayerStateTracker,
    info_dict_keys,
)
from chess_research.eval.player import NanoGptPlayer, Player, StockfishPlayer
from chess_research.globals import Globals


def play_games(params: EvaluationParams):
    ############################
    # SETUP
    ############################

    assert params.nanogpt_idx == "one" or params.nanogpt_idx == "two"

    player_sp = StockfishPlayer(
        skill_level=params.stockfish_skill_level, play_time=0.1
    )  # Is 0.1 a good value? Tradeoff between fair and eval time...
    player_np = NanoGptPlayer(model_name=params.nanogpt_weight_file, model=params.model)

    player_one = player_sp if params.nanogpt_idx == "two" else player_np
    player_two = player_np if params.nanogpt_idx == "two" else player_sp
    player_one.recording_name = (
        f"Stockfish_{params.stockfish_skill_level}"
        if params.nanogpt_idx == "two"
        else params.nanogpt_weight_file.split("/")[-1]
    )
    player_two.recording_name = (
        params.nanogpt_weight_file.split("/")[-1]
        if params.nanogpt_idx == "two"
        else f"Stockfish_{params.stockfish_skill_level}"
    )

    ############################
    # Play games
    ############################

    info_dicts = []
    for _ in range(params.num_games):
        game_state = '[White "Magnus Carlsen"]\n[Black "Stockfish"]\n\n'
        player_one.reset_game_state()
        player_two.reset_game_state()
        if params.nanogpt_elo is not None and params.opponent_elo is not None:
            if player_one.recording_name.startswith(
                "Stockfish"
            ):  # player one is stockfish
                game_state += f"{params.opponent_elo} {params.nanogpt_elo} "
            else:
                # player one is nanogpt
                game_state += f"{params.nanogpt_elo} {params.opponent_elo} "

        board = chess.Board()

        if params.randomize_opening_moves > 0:
            game_state, board = initialize_game_with_random_moves(
                board, game_state, params.randomize_opening_moves
            )

        start_time = time.time()
        total_moves = 0

        finish_the_game = False
        while not board.is_game_over():
            if params.log_locally:
                with open("game.txt", "w") as f:
                    f.write(game_state)
            current_move_num = str(board.fullmove_number) + "."
            total_moves += 1

            # increment legal moves here so player_two isn't penalized for the game ending before its turn
            player_one.legal_moves += 1
            player_two.legal_moves += 1

            if board.fullmove_number != 1:
                game_state += " "

            game_state += current_move_num
            print(f"{current_move_num}", end="")

            for i, player in enumerate([player_one, player_two]):
                (
                    game_state,
                    illegal_moves_one,
                ) = play_turn(
                    player,
                    board,
                    game_state,
                    params,
                    player_one=i == 0,
                )
                player_one.illegal_moves += illegal_moves_one
                if illegal_moves_one != 0:
                    player.legal_moves -= 1

                if (
                    board.is_game_over()
                    or player.resignation
                    or player.failed_to_find_legal_move
                    or total_moves > Globals.MAX_MOVES
                ):
                    finish_the_game = True

            if finish_the_game:
                break

            print("\n", end="")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nGame over. Total time: {total_time} seconds")
        print(f"Result: {board.result()}")
        print(board)
        print()

        ############################
        # Record game results
        ###########################

        info_dict = record_results(
            board,
            player_one,
            player_two,
            game_state,
            total_time,
            total_moves,
            params,
        )
        info_dicts.append(info_dict)

    ############################
    # Cleanup
    ############################

    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

    return info_dicts


def play_turn(
    player: Player,
    board: chess.Board,
    game_state: str,
    params: EvaluationParams,
    player_one: bool,
) -> Tuple[str, bool, bool, int]:
    result = get_legal_move(
        player,
        board,
        game_state,
        player_one,
        params,
        max_attempts=5,
    )
    num_illegal_moves = result.attempts
    move_san = result.move_san
    move_uci = result.move_uci
    resignation = result.is_resignation
    failed_to_find_legal_move = result.is_illegal_move

    if resignation:
        print(f"{player} resigned with result: {board.result()}")
    elif failed_to_find_legal_move:
        print(f"Game over: 5 consecutive illegal moves from {player}")
    elif move_san is None or move_uci is None:
        print(f"Game over: {player} failed to find a legal move")
    else:
        board.push(move_uci)
        game_state += move_san
        print(move_san, end=" ")

    player.resignation = resignation
    player.failed_to_find_legal_move = failed_to_find_legal_move

    return game_state, num_illegal_moves


def get_legal_move(
    player: Player,
    board: chess.Board,
    game_state: str,
    player_one: bool,
    params: EvaluationParams,
    max_attempts: int = 5,
) -> LegalMoveResponse:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None

    for attempt in range(max_attempts):
        move_san = player.get_move(board, game_state, params)

        # Sometimes when GPT thinks it's the end of the game, it will just output the result
        # Like "1-0". If so, this really isn't an illegal move, so we'll add a check for that.
        if move_san is not None:
            if move_san == "1-0" or move_san == "0-1" or move_san == "1/2-1/2":
                print(f"{move_san}, player has resigned")
                return LegalMoveResponse(
                    move_san=None,
                    move_uci=None,
                    attempts=attempt,
                    is_resignation=True,
                )

        try:
            move_uci = board.parse_san(move_san)
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            continue

        if move_uci in board.legal_moves:
            if not move_san.startswith(" "):
                move_san = " " + move_san
            return LegalMoveResponse(move_san, move_uci, attempt)
        print(f"Illegal move: {move_san}")

    # If we reach here, the player has made illegal moves for all attempts.
    print(f"{player} provided illegal moves for {max_attempts} attempts.")
    return LegalMoveResponse(
        move_san=None, move_uci=None, attempts=max_attempts, is_illegal_move=True
    )


def record_results(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    game_state: str,
    total_time: float,
    total_moves: int,
    eval_params: EvaluationParams,
):
    """
    Create the info_dict and populat the wandb eval table if exists, to record the results of the game.
    """
    unique_game_id = generate_unique_game_id()

    (
        player_one_title,
        player_two_title,
        player_one_time,
        player_two_time,
    ) = get_player_titles_and_time(player_one, player_two)

    if player_one.resignation or player_one.failed_to_find_legal_move:
        result = "0-1"
        player_one_score = "0"
        player_two_score = "1"
    elif player_two.resignation or player_two.failed_to_find_legal_move:
        result = "1-0"
        player_one_score = "1"
        player_two_score = "0"
    else:
        result = board.result()
        if "-" in result:
            player_one_score = result.split("-")[0]
            player_two_score = result.split("-")[1]
        elif result == "*":  # Draw
            player_one_score = "1/2"
            player_two_score = "1/2"

    info_dict = {}
    for key, value in zip(
        info_dict_keys,
        [
            unique_game_id,
            game_state,
            result,
            player_one_title,
            player_two_title,
            player_one_time,
            player_two_time,
            player_one_score,
            player_two_score,
            player_one.illegal_moves,
            player_two.illegal_moves,
            player_one.legal_moves,
            player_two.legal_moves,
            player_one.resignation,
            player_two.resignation,
            player_one.failed_to_find_legal_move,
            player_two.failed_to_find_legal_move,
            f"{player_one_title} vs. {player_two_title}",
            board.fullmove_number,
            total_time,
            total_moves,
            eval_params.temperature,
            eval_params.nanogpt_elo,
            eval_params.opponent_elo,
        ],
    ):
        info_dict[key] = value

    if eval_params.wandb_eval_table is not None:
        eval_params.wandb_eval_table.add_data(*info_dict.values())

    csv_file_path = f"{player_one.recording_name}_vs_{player_two.recording_name}"
    csv_file_path = csv_file_path.replace(
        ".", "_"
    )  # filenames can't have periods in them. Useful for e.g. gpt-3.5 models
    csv_file_path += ".csv"

    # Determine if we need to write headers (in case the file doesn't exist yet)
    write_headers = not os.path.exists(csv_file_path)

    if eval_params.log_locally:
        # Append the results to the CSV file
        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=info_dict.keys())
            if write_headers:
                writer.writeheader()
            writer.writerow(info_dict)

        with open("game.txt", "w") as f:
            f.write(game_state)

    return info_dict


def initialize_game_with_opening(
    game_state: str, board: chess.Board
) -> Tuple[str, chess.Board]:
    with open("openings.csv") as file:
        lines = file.readlines()[1:]  # Skip header
    moves_string = random.choice(lines)
    game_state += moves_string
    # Splitting the moves string on spaces
    tokens = moves_string.split()

    for token in tokens:
        # If the token contains a period, it's a move number + move combination
        if "." in token:
            move = token.split(".")[-1]  # Take the move part after the period
        else:
            move = token

        board.push_san(move)
    return game_state, board


def initialize_game_with_random_moves(
    board: chess.Board, initial_game_state: str, randomize_opening_moves: int
) -> Tuple[str, chess.Board]:
    # We loop for multiple attempts because sometimes the random moves will result in a game over
    MAX_INIT_ATTEMPTS = 5
    for attempt in range(MAX_INIT_ATTEMPTS):
        board.reset()  # Reset the board for a new attempt
        game_state = initial_game_state  # Reset the game state for a new attempt
        moves = []
        for moveIdx in range(1, randomize_opening_moves + 1):
            for player in range(2):
                moves = list(board.legal_moves)
                if not moves:
                    break  # Break if no legal moves are available

                move = random.choice(moves)
                moveString = board.san(move)
                if moveIdx > 1 or player == 1:
                    game_state += " "
                game_state += (
                    str(moveIdx) + "." + moveString if player == 0 else moveString
                )
                board.push(move)

            if not moves:
                break  # Break if no legal moves are available

        if moves:
            # Successful generation of moves, break out of the attempt loop
            break
    else:
        # If the loop completes without a break, raise an error
        raise Exception("Failed to initialize the game after maximum attempts.")

    print(game_state)
    return game_state, board


def get_ckpt_path(directory):
    z = [
        int(path.split("_")[1].split(".")[0])
        for path in os.listdir(directory)
        if path.startswith("ckpt_") and path.endswith(".pt")
    ]  # all of the saved checkpoint iteration numbers
    return sorted(z)[-1]


def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)  # 4-digit random number
    return f"{timestamp}-{random_num}"


def get_player_titles_and_time(
    player_one: Player, player_two: Player
) -> Tuple[str, str, Optional[float], Optional[float]]:
    player_one_config = player_one.get_config()
    player_two_config = player_two.get_config()

    # For player one
    if "model" in player_one_config:
        player_one_title = player_one_config["model"]
        player_one_time = None
    else:
        player_one_title = f"Stockfish {player_one_config['skill_level']}"
        player_one_time = player_one_config["play_time"]

    # For player two
    if "model" in player_two_config:
        player_two_title = player_two_config["model"]
        player_two_time = None
    else:
        player_two_title = f"Stockfish {player_two_config['skill_level']}"
        player_two_time = player_two_config["play_time"]

    return (player_one_title, player_two_title, player_one_time, player_two_time)
