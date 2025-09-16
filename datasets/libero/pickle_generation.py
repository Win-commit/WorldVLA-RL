import os
import pickle
from tqdm import tqdm
import numpy as np
import sys
import argparse
PROJECT_ROOT = "/liujinxin/zhy"
sys.path.append(f"{PROJECT_ROOT}/ICLR2026/datasets")
from normalize_pi0 import RunningStats, save
def sort_by_int(filename):
    return int(os.path.splitext(filename)[0])

def load_files(file_folder, scene, **kwargs):
    if kwargs.get('special_name'):
        special_folder = os.path.join(file_folder, scene, kwargs.get('special_name'))
        if not os.path.exists(special_folder):
            return []
        files = [os.path.join(special_folder, file) for file in sorted(os.listdir(special_folder), key=sort_by_int)]
    else:
        picture_dir = os.path.join(file_folder, scene)
        if not os.path.exists(picture_dir):
            return []
        files = [os.path.join(picture_dir, file) for file in sorted(os.listdir(picture_dir), key=sort_by_int)]
    return files

def main(dataset_path, output_path, normalizer_path, output_filename):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(normalizer_path, exist_ok=True)

    language_dir = os.path.join(dataset_path, "libero_all")
    vq_dir = os.path.join(dataset_path, "libero_all_codes_200")
    gripper_vq_dir = os.path.join(dataset_path, "libero_all_gripper_codes_200")

    min_frames = 8
    result_file = []

    print("Loading scenes from:", language_dir)
    for scene in tqdm(os.listdir(language_dir)):
        instr_file = os.path.join(language_dir, scene, "instruction.txt")
        if not os.path.exists(instr_file):
            continue
        with open(instr_file, "r") as f:
            text = f.read()

        # Load action sequences
        action_files = load_files(language_dir, scene, special_name = "actions")
        if len(action_files) < min_frames:
            continue
        action = [np.load(a) for a in action_files]
        
        # Load state sequences
        state_files = load_files(language_dir, scene, special_name = "states")
        state = [np.load(s) for s in state_files]

        # Load reward sequences
        reward_files = load_files(language_dir, scene, special_name = "reward")
        reward = [np.load(r) for r in reward_files]

        # Load returnToGo sequences
        returnToGo_files = load_files(language_dir, scene, special_name = "returnToGo")
        returnToGo = [np.load(r) for r in returnToGo_files]

        # Load image tokens
        img_files = load_files(vq_dir, scene)

        # Load gripper image tokens
        gripper_img_files = load_files(gripper_vq_dir, scene)

        # Filter out short clips
        if len(img_files) < min_frames or len(gripper_img_files) < min_frames:
            continue

        result_file.append({
            "text": text,
            "image": img_files,
            "action": action,
            "gripper_image": gripper_img_files,
            "state": state,
            "reward": reward,
            "returnToGo": returnToGo
        })
    print(f"Total number of valid scenes: {len(result_file)}")
    if not result_file:
        raise ValueError("No valid scenes found. Check your dataset path.")

    # === Normalize actions ===
    action_normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in result_file])
    action_normalizer.update(action_data)
    action_stats = action_normalizer.get_statistics()

    print("Mean:", action_stats.mean)
    print("Std:", action_stats.std)
    print("Q01:", action_stats.q01)
    print("Q99:", action_stats.q99)
    
    for scene in result_file:
        action = scene["action"]
        # Normalize to [-1, 1] using Q01 and Q99 as bounds
        normalized = 2 * (action - action_stats.q01) / (action_stats.q99 - action_stats.q01 + 1e-8) - 1
        scene["action"] = np.clip(normalized, -1, 1)
    # === Normalize states ===
    state_normalizer = RunningStats()
    state_data = np.concatenate([scene["state"] for scene in result_file])
    state_normalizer.update(state_data)
    state_stats = state_normalizer.get_statistics()

    print("Mean:", state_stats.mean)
    print("Std:", state_stats.std)
    print("Q01:", state_stats.q01)
    print("Q99:", state_stats.q99)

    for scene in result_file:
        state = scene["state"]
        normalized = 2 * (state - state_stats.q01) / (state_stats.q99 - state_stats.q01 + 1e-8) - 1
        scene["state"] = np.clip(normalized, -1, 1)

    # === Save normalized dataset ===
    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    print(f"Saved normalized data to {output_file}")

    # === Save normalization statistics ===
    save(normalizer_path, {"action": action_stats, "state": state_stats})
    print(f"Saved normalizer statistics to {normalizer_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Libero dataset action values.")
    parser.add_argument("--dataset_path", type=str, default="/liujinxin/zhy/ICLR2026/datasets/libero/data", help="Root path to dataset.")
    parser.add_argument("--output_path", type=str, default="/liujinxin/zhy/ICLR2026/datasets/libero/data/meta", help="Path to save normalized data.")
    parser.add_argument("--normalizer_path", type=str, default="/liujinxin/zhy/ICLR2026/configs/normalizer_libero", help="Path to save normalization stats.")
    parser.add_argument("--output_filename", type=str, default="libero_all_norm.pkl", help="Filename for normalized pickle output.")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.normalizer_path, args.output_filename) 