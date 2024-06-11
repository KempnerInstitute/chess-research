"""
Sample from a trained model
"""

from typing import Optional

import inspect
import os
import pickle
import platform
import random
import re
from contextlib import nullcontext
from inspect import getsourcefile
from pathlib import Path

import chess
import torch

from chess_research.eval.data_structures import EvaluationParams
from chess_research.globals import Globals
from chess_research.model import GPT, GPTConfig


# Define base Player class
class Player:
    def __init__(self) -> None:
        self.illegal_moves: int = 0
        self.legal_moves: int = 0
        self.resignation: bool = False
        self.failed_to_find_legal_move: bool = False
        self.recording_name = None
        
    def get_move(
        self,
        board: chess.Board,
        params: EvaluationParams,
    ) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def reset_game_state(self):
        self.illegal_moves: int = 0
        self.legal_moves: int = 0
        self.resignation: bool = False
        self.failed_to_find_legal_move: bool = False

class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating system.
        """
        if platform.system() == "Linux":
            return Globals.STOCKFISH_PATH
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            return "stockfish"
        elif platform.system() == "Windows":
            return (
                r"C:\Users\adamk\Documents\Stockfish\stockfish-windows-x86-64-avx2.exe"
            )
        else:
            raise OSError("Unsupported operating system")

    def __init__(self, skill_level: int, play_time: float):
        super().__init__()
        self._skill_level = skill_level
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        stockfish_path = StockfishPlayer.get_stockfish_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(
        self,
        board: chess.Board,
        game_state: str,
        params: EvaluationParams,
    ) -> Optional[str]:
        if self._skill_level == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)
        elif self._skill_level < 0:
            self._engine.configure({"Skill Level": 0})
            result = self._engine.play(
                board, chess.engine.Limit(time=1e-8, depth=1, nodes=1)
            )

        else:
            self._engine.configure({"Skill Level": self._skill_level})
            result = self._engine.play(board, chess.engine.Limit(time=self._play_time))
        if result.move is None:
            return None
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


class NanoGptPlayer(Player):
    def __init__(
        self,
        model_name: str,
        model: Optional[torch.nn.Module] = None,
        activation_name: Optional[str] = None,
        activation_coefficient: Optional[float] = None,
    ):
        super().__init__()
       
        self.model_name = model_name

        device = "cuda"
        if model is None:
            # init from a model saved in a specific directory
            # ckpt_path = os.path.join(BASE_DIR, out_dir, self.model_name)
            checkpoint = torch.load(self.model_name, map_location=device)
            gptconf = GPTConfig(**checkpoint["model_args"])

            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            if activation_name is not None:
                state_dict, gptconf = add_activation_bias_to_state_dict(
                    state_dict, device, activation_name, gptconf, activation_coefficient
                )
            model = GPT(gptconf)
            model.load_state_dict(state_dict)
            model = torch.compile(model)

        model.eval()
        model.to(device)

        # look for the meta pickle in case it is available in the dataset folder
        stoi, itos = Globals.meta["stoi"], Globals.meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

        self.encode = encode
        self.decode = decode
        self.model = model
        self.device = device

    def get_nanogpt_response(self, game_state: str, temperature: float) -> str:
        num_samples = 1  # number of samples to draw
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        max_new_tokens = 10

        # Remove ["stockfish elo xxx"]\n["stockfish elo xxx"]\n\n from game_state
        # nanogpt was trained only on pgn transcripts
        game_state = game_state.split("\n\n")[1].strip()

        # Nanogpt was trained on pgn transcripts of this format: 1.e4 e5 2.Nf3 (not 1. e4 e5 2. Nf3)
        # I did this to save on tokens
        # We remove the space after the move number to match the training data
        game_state = re.sub(r"(\d+\.) ", r"\1", game_state)

        game_state = ";" + game_state

        # print("game_state", game_state)

        start_ids = self.encode(game_state)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            for k in range(num_samples):
                y = self.model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )

                model_response = self.decode(y[0].tolist())

        # print("model_response", model_response)
        # model_response includes the input string
        model_response = model_response[len(game_state) :]
        if ";" in model_response:
            model_response = model_response.split(";")[0]

        if "+" in model_response:  # UNTESTED
            model_response = model_response.split("+")[0]

        return model_response

    def get_move_from_response(self, response: str) -> str:
        # Parse the response to get only the first move
        moves = response.split()
        if not moves:
            return ""
        first_move = moves[0]

        return first_move

    def get_move(
        self,
        board: str,
        game_state: str,
        params: EvaluationParams
    ) -> str:
        completion = self.get_nanogpt_response(game_state, params.temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        return {"model": self.model_name}



def add_activation_bias_to_state_dict(
    state_dict,
    device,
    activation_names: list[str],
    config: GPTConfig,
    activation_coefficient: float,
):
    activation_dir = ""
    config.bias = True
    print(config)

    state_dict["transformer.ln_f.bias"] = torch.zeros_like(
        state_dict["transformer.ln_f.weight"]
    )

    for i in range(config.n_layer):
        layer_key = f"transformer.h.{i}"

        state_dict[f"{layer_key}.ln_1.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_1.weight"]
        )
        state_dict[f"{layer_key}.ln_2.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_2.weight"]
        )

        mlp_bias_shape = state_dict[f"{layer_key}.mlp.c_fc.weight"].shape[0]

        assert mlp_bias_shape == config.n_embd * 4

        state_dict[f"{layer_key}.mlp.c_fc.bias"] = torch.zeros(
            mlp_bias_shape, device=device
        )
        state_dict[f"{layer_key}.mlp.c_proj.bias"] = torch.zeros(
            config.n_embd, device=device
        )

        state_dict[f"{layer_key}.attn.c_attn.bias"] = torch.zeros(
            config.n_embd * 3, device=device
        )
        state_dict[f"{layer_key}.attn.c_proj.bias"] = torch.zeros(
            config.n_embd, device=device
        )

    for activation_name in activation_names:
        activation_state_dict = torch.load(
            f"activations/{activation_dir}{activation_name}",
            map_location=device,
        )
        difference_vector = activation_state_dict["difference_vector"]
        difference_vector *= activation_coefficient
        layer = activation_state_dict["layer"]
        # print(activation_state_dict.keys())

        # Add the difference vector to the attention bias
        state_dict[f"transformer.h.{layer}.mlp.c_proj.bias"] = difference_vector

    return state_dict, config
