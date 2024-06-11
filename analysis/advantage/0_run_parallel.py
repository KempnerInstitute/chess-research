import os
import platform

my_hostname = platform.node().split(".")[0]


folder = "50M-High-Elo-1000-No-Elo-Conditioning"
root_dir = "/path/to/runs/lichess-models"
TOTAL_JOBS = 4
for job_id in range(TOTAL_JOBS):
    abs_folder = os.path.join(root_dir, folder)
    os.chdir(abs_folder)
    latest_dir = sorted([f for f in os.listdir() if f.startswith("2024")])[-1]
    cmd = f"CUDA_VISIBLE_DEVICES={job_id} python /path/to/scripts/alternative_generation_0.py  --weight_file {os.path.join(abs_folder, latest_dir, 'ckpt_100000.pt')} --job_id {job_id} --total_jobs {TOTAL_JOBS}"
    log_cmd = f"{cmd} > /path/to/{job_id}_{folder}_adv_analysis_5-19-24-18:00-full_sample.out 2>&1 &"

    print(log_cmd)
    os.system(log_cmd)
