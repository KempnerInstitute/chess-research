import asyncio
import csv
import io

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd


async def analyze_game(
    engine, game_id, game_moves: list, time_limit: float = 1.0
) -> str:
    board = chess.Board()
    print(f"Analyzing {game_id}...")
    analysis = []
    for i, move in enumerate(game_moves):
        info = await engine.analyse(
            board, chess.engine.Limit(time=time_limit), multipv=256
        )

        timestep_analysis = {}
        for mv in info:
            key = mv["pv"][0].uci()
            adv_prob = mv["score"].black().wdl(model="lichess").expectation()
            score = mv["score"].black().score(mate_score=10000) / 100
            timestep_analysis[key] = (adv_prob, score)

        analysis.append(timestep_analysis)

        board.push(move)
        # print(info)
        # analysis.append((adv_prob, score))
        # because chess is zero-sum, white_score = -black_score and white_adv_prob = 1 - black_adv_prob

        # print(f"Move: {i} | Adv Prob: {adv_prob} | Score: {score}")

    print(f"Done analyzing {game_id}")
    return analysis


async def main(engine_path: str, csv_file: str, num_workers: int = 300) -> None:
    games_info = pd.read_csv(csv_file)
    engines = [
        (await chess.engine.popen_uci(engine_path))[1] for _ in range(num_workers)
    ]

    num_games = len(games_info)
    analyzed_games = 0
    all_res = []
    try:
        while analyzed_games < num_games:
            tasks = []
            for engine in engines:
                if analyzed_games >= num_games:
                    break
                tasks.append(
                    analyze_game(
                        engine,
                        analyzed_games,
                        chess.pgn.read_game(
                            io.StringIO(games_info.iloc[analyzed_games]["transcript"]),
                        ).mainline_moves(),
                    ),
                )
                analyzed_games += 1

            res = await asyncio.gather(*tasks)
            all_res.extend(res)

        # print(all_res)

        # Write results as new pandas series
        games_info["adv_analysis"] = [str(k) for k in all_res]
        games_info.to_csv(csv_file[:-4] + "_adv_long_analysis.csv")
    finally:
        for engine in engines:
            await engine.quit()


async def multi_main(engine_path, csv_files):
    await asyncio.gather(*[main(engine_path, csv_file) for csv_file in csv_files])


if __name__ == "__main__":
    engine_path = "stockfish"
    engine_path = "/path/to/stockfish"
    csv_files = [
        "/path/to/all_Stockfish_1_vs_ckpt_100000_pt_0_001.csv",
    ]

    csv_files = [
        "/path/to/all_Stockfish_1_vs_ckpt_100000_pt_0_001.csv",
        "/path/to/all_Stockfish_1_vs_ckpt_100000_pt_0_75.csv",
        "/path/to/all_Stockfish_1_vs_ckpt_100000_pt_1.csv",
    ]

    asyncio.run(multi_main(engine_path, csv_files))
