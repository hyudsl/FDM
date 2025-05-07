import os
from cleanfid import fid

def create_dataset_stats(dataset_name, dataset_path):
    print("create ", dataset_name, " stats...")
    if fid.test_stats_exists(dataset_name, mode="clean"):
        return
    
    fid.make_custom_stats(dataset_name, dataset_path, mode="clean", device='cpu')

    return

def cal_fid(dataset_name, generated_path):
    score = fid.compute_fid(generated_path, dataset_name=dataset_name,
          mode="clean", dataset_split="custom", device='cpu', num_workers=0)

    return score

def remove_dataset_stats(dataset_name):
    fid.remove_custom_stats(dataset_name, mode="clean")

    