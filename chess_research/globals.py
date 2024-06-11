import os
from pathlib import Path

from eztils import abspath, setup_path


class Globals:
    MAX_MOVES = 89  # Due to time constraints, max number of moves per game

    REPO_DIR = setup_path(Path(abspath()) / "..")
    DATA_ROOT = setup_path(os.getenv("DATA_ROOT") or REPO_DIR)
    RUN_DIR = LOG_DIR = Path()
    STOCKFISH_PATH = "stockfish"

    def print():
        attributes = "\n".join(
            f"{key}={value!r}"
            for key, value in Globals.__dict__.items()
            if not key.startswith("_") and key != "print"
        )
        print(f"Globals({attributes})")

    meta = {
        "vocab_size": 32,
        "itos": {
            0: " ",
            1: "#",
            2: "+",
            3: "-",
            4: ".",
            5: "0",
            6: "1",
            7: "2",
            8: "3",
            9: "4",
            10: "5",
            11: "6",
            12: "7",
            13: "8",
            14: "9",
            15: ";",
            16: "=",
            17: "B",
            18: "K",
            19: "N",
            20: "O",
            21: "Q",
            22: "R",
            23: "a",
            24: "b",
            25: "c",
            26: "d",
            27: "e",
            28: "f",
            29: "g",
            30: "h",
            31: "x",
        },
        "stoi": {
            " ": 0,
            "#": 1,
            "+": 2,
            "-": 3,
            ".": 4,
            "0": 5,
            "1": 6,
            "2": 7,
            "3": 8,
            "4": 9,
            "5": 10,
            "6": 11,
            "7": 12,
            "8": 13,
            "9": 14,
            ";": 15,
            "=": 16,
            "B": 17,
            "K": 18,
            "N": 19,
            "O": 20,
            "Q": 21,
            "R": 22,
            "a": 23,
            "b": 24,
            "c": 25,
            "d": 26,
            "e": 27,
            "f": 28,
            "g": 29,
            "h": 30,
            "x": 31,
        },
    }
