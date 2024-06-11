# %%
import argparse
import json
import time

import torch
import tqdm
from data.zstd_process import StreamingPGNDataset

############
# StreamingPGNDataset
############
# add max_elo argparser
parser = argparse.ArgumentParser()
parser.add_argument("--max-elo", type=int, default=1000)

args = parser.parse_args()

print("max_elo:", args.max_elo)
data_dir = "/path/to/lichess_elo_binned"

ds = StreamingPGNDataset(data_dir, max_elo=args.max_elo)


import io

# %%
from collections import Counter, defaultdict

import chess
import chess.pgn

state_visitation_distribution = defaultdict(Counter)
state_action_distribution = defaultdict(lambda: defaultdict(Counter))

MAX_GAMES = 500_000
# MAX_GAMES = 1_000
# game_dict = {
#     "url": "O31LDwfb",
#     "WhiteElo": "761",
#     "BlackElo": "667",
#     "result": "1-0",
#     "transcript": "1.d4 d6 2.e3 e5 3.Bb5+ Bd7 4.Bxd7+ Qxd7 5.Bd2 Nc6 6.Nf3 exd4 7.exd4 Nxd4 8.Nxd4 Qe7+ 9.Be3 Qf6 10.Nd2 c5 11.Ne4 cxd4 12.Nxf6+ gxf6 13.O-O Bg7 14.Bxd4 Ne7 15.Re1 O-O 16.Rxe7 b6 17.Re3 f5 18.Bxg7 Kxg7 19.Kf1 f4 20.Re4 f5 21.Re6 Rad8 22.Qh5 Rfe8 23.Rae1 Rxe6 24.Rxe6 h6 25.Qg6+ Kf8 26.Rf6+ Ke7 27.Qg7+ Ke8 28.Rf8#",
# }
for game_iter, game_dict in tqdm.tqdm(enumerate(ds), total=MAX_GAMES):
    if game_iter > MAX_GAMES:
        break

    game = chess.pgn.read_game(io.StringIO(game_dict["transcript"]))
    b = game.board()
    for i, move in enumerate(game.mainline_moves()):
        state_visitation_distribution[i][b.fen()] += 1
        state_action_distribution[i][b.fen()][move.uci()] += 1
        b.push(move)


torch.save(
    {
        "state_visitation_distribution": json.dumps(state_visitation_distribution),
        "state_action_distribution": json.dumps(state_action_distribution),
    },
    f"state_action_dist_{args.max_elo}_{time.time()}.pt",
)
