import itertools
import os
import platform

my_hostname = platform.node().split(".")[0]

hostnames = [
    "slurm_gpu1",
    "slurm_gpu2",
    "slurm_gpu3",
    "slurm_gpu4",
    "slurm_gpu4",
    "slurm_gpu5",
]
code_dir = "/path/to/chess_research"

os.chdir(code_dir)


high_elos = [1000, 1300, 1500]
model_sizes = ["707M", "302M"]

params = list(itertools.product(high_elos, model_sizes))

for node_id, hostname in enumerate(hostnames):
    # cmd = f"CUDA_VISIBLE_DEVICES={job_id}  ./scripts/50M_win_condition.sh {1000 + 100 * cmd_num}"
    high_elo, model_size = params[node_id]

    cmd = f"{code_dir}/.venv/bin/torchrun --standalone --nproc_per_node=4  {code_dir}/chess_research/__init__.py -c {code_dir}/config/{model_size}_1000.json  --dataset /path/to/lichess_elo_binned --high_elo {high_elo} --wandb_run_name {model_size}-High-Elo-{high_elo}"
    ssh_cmd = f"ssh {hostname} -t -T '{cmd}' > /path/to/big_model_{node_id}.out 2>&1  &"

    # cmd = 'pkill -9 python'
    # cmd = 'nvidia-smi'
    # ssh_cmd = f"ssh {hostname} -t -T '{cmd}' "

    print(ssh_cmd)
    os.system(ssh_cmd)
