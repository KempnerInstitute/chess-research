# %%
# %%
import ast
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm


# %%
def process_sampled_moves_and_advantages(i, row):
    sampled_moves = json.loads(all_sampled_moves[i])

    adv = ast.literal_eval(row["adv_analysis"])

    # sort all advantages
    adv = [dict(sorted(i.items(), key=lambda item: item[1][0])) for i in adv]
    return sampled_moves, adv


def advantage_calculation(advantages: dict, distribution: Counter):
    total = distribution.total()
    adv = 0
    acc = 0
    for k, v in distribution.items():
        if k is None:
            continue
        adv += advantages[k][1] * v / total
        acc += advantages[k][0] * v / total
    return adv, acc


def top_k_calculation(advantages: dict, distribution: Counter, k=1):
    total = distribution.total()
    correct = 0
    top_k = list(advantages.keys())[:k]
    for k, v in distribution.items():
        if k is None:
            continue
        if k in top_k:
            rank = top_k.index(k)
            # correct += v / (rank + 1)
            correct += v
    return correct / total


def calculate_statistics(adv: dict, sampled_moves: list, pgn: str, empty: bool = False):
    ret = defaultdict(list)
    ret.update(
        {
            "advantage_deltas_0.001": [],
            "advantage_deltas_0.75": [],
            "average_advantages": [],
            "acc_deltas_0.001": [],
            "acc_deltas_0.75": [],
            "average_accs": [],
        }
    )
    if empty:
        return ret

    total_moves = 0

    for half_move_clock, ply in enumerate(sampled_moves):
        if ply is None:
            continue
        if half_move_clock >= len(adv):
            break

        if half_move_clock % 2 == 0:
            # reverse as nanogpt is white here, and the advantages are initally from black's perspetive
            # print("white!")
            advantages = {k: (1 - v[0], -v[1]) for k, v in adv[half_move_clock].items()}
        else:
            # print("black!")
            advantages = adv[half_move_clock]
        advantages = dict(
            sorted(advantages.items(), key=lambda item: item[1][0], reverse=True)
        )
        total_moves += 1

        # print('Half Move:', half_move_clock)

        avg_advantages = {}
        avg_accs = {}

        ucis = {}
        for k, v in ply.items():
            # uci = [i[1] for i in v]
            uci = v

            ucis[k] = Counter(uci)
            avg, acc = advantage_calculation(advantages, ucis[k])
            # print(
            #     k, f"advantage: {avg:.2f}"
            # )
            avg_advantages[k] = avg
            avg_accs[k] = acc

        ret["average_advantages"].append(avg_advantages)
        ret["average_accs"].append(avg_accs)

        if avg_accs["0.001"] - avg_accs["1"] > 0.3:
            print()
            print(f"{pgn=}")
            print(f"dist={ucis}")
            print(f"{half_move_clock=}")
            print(f"{advantages=}")
            print()

        for k, v in avg_advantages.items():
            if k != "1":
                ret[f"advantage_deltas_{k}"].append(
                    avg_advantages[k] - avg_advantages["1"]
                )
                ret[f"acc_deltas_{k}"].append(avg_accs[k] - avg_accs["1"])
        # get top k calculation
        for temp, uci in ucis.items():
            for k in [1, 3, 5]:
                ret[f"{temp}_{k}_acc"].append(top_k_calculation(advantages, uci, k=k))

        # print()

    return ret, total_moves


if __name__ == "__main__":
    df = pd.concat(
        [
            # pd.read_csv(
            #     "/path/to/advantage-analysis/2024-05-19---10-37-53_chess-original/ckpt_100000_pt_0_75_vs_Stockfish_1.csv"
            # ),
            # pd.read_csv(
            #     "/path/to/advantage-analysis/2024-05-19---10-37-54_chess-original/Stockfish_1_vs_ckpt_100000_pt_0_75.csv"
            # ),
            # pd.read_csv(
            #     "/path/to/advantage-analysis/2024-05-19---10-37-55_chess-original/Stockfish_1_vs_ckpt_100000_pt_0_75.csv"
            # ),
            pd.read_csv(
                # "/path/to/advantage-analysis/all_Stockfish_1_vs_ckpt_100000_pt_0_001_adv_long_analysis.csv"
                "/path/to/analysis-games/Stockfish_1_vs_ckpt_100000_pt_0_001_adv_long_analysis.csv"
                # "/path/to/advantage-analysis/all_Stockfish_1_vs_ckpt_100000_pt_1_adv_long_analysis.csv"
            ),
        ]
    )
    all_sampled_moves = torch.load("/path/to/analysis-try-3/all_data.pt")["0_001"]
    # temp=0.001: Counter"{'0"1': 97" '1"0': 2" '1/2-1"2': 1})
    # temp=0.75: Counter"{'0"1': 95" '1"0': 4" '1/2-1"2': 1})
    # temp=1.0: Counter"{'0"1': 97" '1"0': 2" '1/2-1"2': 1})

    # df.to_csv("all_Stockfish_1_vs_ckpt_100000_pt_0_75.csv")

    stats = calculate_statistics(None, None, None, empty=True)
    total_moves_all = 0
    for i, row in tqdm.tqdm(df.iterrows()):
        sampled_moves, adv = process_sampled_moves_and_advantages(i, row)
        new_stats, total_moves = calculate_statistics(
            adv, sampled_moves, row["transcript"]
        )
        total_moves_all += total_moves
        for k, v in new_stats.items():
            stats[k].extend(v)
    print("Total moves all:", total_moves_all)
    torch.save(stats, "stats_3.pt")

    # advantage_deltas_0.001 -0.1
    # advantage_deltas_0.75 -0.13

    # average_advantages
    # 0.001 -16.89
    # 0.75 -16.92
    # 1 -16.79

    # acc_deltas_0.001 0.01
    # acc_deltas_0.75 0.01

    # average_accs
    # 0.001 0.27
    # 0.75 0.26
    # 1 0.25

    # 0.001_1_acc 0.07
    # 0.001_3_acc 0.11
    # 0.001_5_acc 0.12
    # 0.75_1_acc 0.08
    # 0.75_3_acc 0.11
    # 0.75_5_acc 0.12
    # 1_1_acc 0.07
    # 1_3_acc 0.11
    # 1_5_acc 0.12

    for i in stats:
        if not isinstance(stats[i][0], dict):
            print(i, round(np.mean(np.array(stats[i])), 4))
        else:
            print()
            print(i)
            for k in stats[i][0]:
                print(k, round(np.mean(np.array([j[k] for j in stats[i]])), 4))
            print()

    # advantage_deltas_0.001 0.09
    # advantage_deltas_0.5 0.07

    # average_advantages
    # 0.001 -5.74
    # 0.5 -5.76
    # 1 -5.83

    # acc_deltas_0.001 0.02
    # acc_deltas_0.5 0.02

    # average_accs
    # 0.001 0.43
    # 0.5 0.42
    # 1 0.4

    # 0.001_1_acc 0.06
    # 0.001_3_acc 0.09
    # 0.001_5_acc 0.1
    # 0.5_1_acc 0.06
    # 0.5_3_acc 0.09
    # 0.5_5_acc 0.1
    # 1_1_acc 0.06
    # 1_3_acc 0.09
    # 1_5_acc 0.1

    # white only
    # advantage_deltas_0.001 0.48
    # advantage_deltas_0.5 0.29

    # average_advantages
    # 0.001 -6.12
    # 0.5 -6.31
    # 1 -6.59

    # acc_deltas_0.001 0.02
    # acc_deltas_0.5 0.01

    # average_accs
    # 0.001 0.42
    # 0.5 0.41
    # 1 0.4

    # 0.001_1_acc 0.06
    # 0.001_3_acc 0.12
    # 0.001_5_acc 0.17
    # 0.5_1_acc 0.06
    # 0.5_3_acc 0.13
    # 0.5_5_acc 0.17
    # 1_1_acc 0.06
    # 1_3_acc 0.13
    # 1_5_acc 0.18
