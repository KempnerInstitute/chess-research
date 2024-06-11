# URL,whiteelo,blackelo,outcome,transcript
# iML6DQBA,1877,1936,1-0,1. e4 d4.....

from typing import List, Optional

import functools
import io
import math
import os
import random
import re
import time

import numpy as np
import torch
import tqdm
import zstandard
from eztils.run_parallel_os_system import calculate_split
from torch.utils.data import IterableDataset

from chess_research.globals import Globals


def bound(directory, lower_bound=-1, upper_bound=9999):
    bins = []
    for d in os.listdir(directory):
        try:
            bin_ = int(d)
            if lower_bound <= bin_ <= upper_bound:
                bins.append(d)
        except ValueError:
            continue

    j = os.path.join

    return sum(
        [
            [j(directory, bin_, f) for f in os.listdir(j(directory, bin_))]
            for bin_ in bins
        ],
        [],
    )


class StreamingPGNDataset(IterableDataset):
    def __init__(
        self,
        directory,
        seed=42,
        low_elo=-1,
        high_elo=9999,
        train=True,
        train_test_split=0.95,
    ):
        self.set_file_paths(
            bound(directory, lower_bound=low_elo, upper_bound=high_elo),
            seed,
            train,
            train_test_split,
        )

        self.high_elo = high_elo
        self.low_elo = low_elo

    def set_file_paths(self, file_paths, seed, train, train_test_split):
        self.file_paths = file_paths
        self.rng = random.Random(seed)
        self.rng.shuffle(self.file_paths)

        # this could be done at the level of granularity of actual games as well, but would be a bit more difficult. doing at level of months is probably fine
        train_len = int(len(self.file_paths) * train_test_split)
        if train_len == len(self.file_paths):
            train_len -= 1

        if train:
            self.file_paths = self.file_paths[:train_len]
        else:
            self.file_paths = self.file_paths[train_len:]

    def read_game(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # multiprocessing
            # print('multiprocessing')
            assert worker_info.num_workers <= len(
                self.file_paths
            ), f"Num workers {worker_info.num_workers} greater than number of files {len(self.file_paths)}."
            start, end = calculate_split(
                worker_info.num_workers, len(self.file_paths), worker_info.id
            )
            self.file_paths = self.file_paths[start:end]
            # print(worker_info.id, start, end)
        # else:
        # print('not multiprocessing')

        def game_generator(path):
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as pgn_file:
                stream_reader = dctx.stream_reader(pgn_file)
                text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

                for i in text_stream:
                    # yield i
                    url, white_elo, black_elo, result, game = i.split(",")

                    yield {
                        "url": url,
                        "WhiteElo": white_elo,
                        "BlackElo": black_elo,
                        "result": result,
                        "transcript": game[:-1],
                    }  # game[:-1] for removing newline character

        gen = [game_generator(file) for file in self.file_paths]

        i = 0
        while len(gen) > 0:
            try:
                game = next(gen[i % len(gen)])
                if (
                    game.get("transcript") is None
                    or game.get("result") == "None"
                    or game.get("result") == "*"
                ):  # a bit of hack, should only filter on no result for win_conditioning
                    i += 1
                    continue

            except StopIteration:
                del self.file_paths[i % len(gen)]
                del gen[i % len(gen)]
                continue

            i += 1
            yield game

            # parse txt

    def __iter__(self):
        return self.read_game()


def interleave(game, interleave_type):
    transcript = game["transcript"].split(".")
    vocab_lookup = {"1/2-1/2": "D", "1-0": "W", "0-1": "L"}
    res = []
    for i in transcript:
        res.append(i)
        if interleave_type == "result":
            res.append(vocab_lookup[game["result"]])
        if interleave_type == "elo":
            res.append(
                game["WhiteElo"]
            )  # may need to use a positional embedding here like cosine. on the other hand, could alos use less tokens
            res.append(game["BlackElo"])

    return ".".join(res[:-1])  # remove last result


def get_game_length(game):
    numbers = re.findall(r"\b\d+\b", game["transcript"])
    # Convert extracted strings to integers
    numbers = list(map(int, numbers))
    # Find the maximum number
    return max(numbers)


class StreamingBlockPGNDataset(StreamingPGNDataset):
    def __init__(
        self,
        directory,
        seed=42,
        block_size=1024,
        high_elo: int = 9999,
        low_elo: int = -1,
        win_conditioning: bool = True,
        elo_condition: bool = False,
        length_gen: int = None,
        train: bool = True,
        train_test_split: float = 0.95,
    ):
        super().__init__(
            directory=directory,
            seed=seed,
            low_elo=low_elo,
            high_elo=high_elo,
            train=train,
            train_test_split=train_test_split,
        )
        self.block_size = block_size
        self.length_gen = length_gen
        self.win_conditioning = win_conditioning
        self.elo_condition = elo_condition  # elo conditioning

        assert not (
            self.win_conditioning and self.elo_condition
        ), "Cannot have both win_conditioning and elo_condition set to True."
        if self.win_conditioning:
            self.transform = functools.partial(interleave, interleave_type="result")
        elif self.elo_condition:
            self.transform = functools.partial(interleave, interleave_type="elo")
        else:
            self.transform = lambda x: x["transcript"]
        self.tokenizer = Globals.meta

    def read_game_block(self):
        ds = self.read_game()
        game = None
        full_block = ""
        while True:
            if (
                game is not None
            ):  # use the previous game that was cut off in the last block
                full_block += ";" + self.transform(game)

                game = None

            while len(full_block) < self.block_size:
                while True:
                    game = next(ds)
                    if (
                        self.length_gen is None
                        or get_game_length(game) <= self.length_gen
                    ):
                        break

                full_block += ";" + self.transform(game)

            out = full_block[: self.block_size]
            full_block = ""
            yield np.array([self.tokenizer["stoi"][c] for c in out], dtype=np.int64)

    def __iter__(self):
        return self.read_game_block()


# takes around 17 minutes to cycle through a single shard

if __name__ == "__main__":
    ############
    # StreamingPGNDataset
    ############
    data_dir = "/path/to/lichess_elo_binned"

    # ds = StreamingPGNDataset(
    #     data_dir,
    #     high_elo=1000
    # )
    # for k in tqdm.tqdm(ds):
    #     pass

    ############
    # StreamingBlockPGNDataset
    ############

    ds_block = torch.utils.data.DataLoader(
        StreamingBlockPGNDataset(data_dir, low_elo=2000, length_gen=20),
        # StreamingBlockPGNDataset(data_dir, low_elo=2000),
        # StreamingBlockPGNDataset(data_dir),
        # num_workers=24,
        num_workers=12,
        batch_size=125,
    )
    itr_block = iter(ds_block)
    # next(itr_block)
    z = next(itr_block)
    print(z)
    print(z.shape)

    t = time.time()
    itr_blocks = 1000000000000
    for i in tqdm.tqdm(range(itr_blocks)):
        z = next(itr_block)
    end = time.time() - t

    print(
        f"There are approximately 100M games per month...so this would take approximately {int(end * 1e8 / itr_blocks / 60)} minutes to process the full month."
    )
