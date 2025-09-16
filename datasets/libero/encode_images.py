from transformers import AutoModel, AutoImageProcessor
import numpy as np
import sys
import torch
from PIL import Image
import os
import re
from tqdm import tqdm
import shutil

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def load_images(folder_path, size, interval=2, augmentation=None):
    """Load and resize all images in the specified folder."""
    image_paths = sorted(os.listdir(folder_path),key=natural_sort_key)[::interval]
    images = [Image.open(os.path.join(folder_path, img)).resize(size) for img in image_paths]

    if augmentation is not None:
        images = augmentation(images)

    return images,image_paths


def image_level_enc(images, model, processor, save_codes_path, batch_size=1):
    """Process images in batches: encode, decode, and save the reconstructed images and codes."""
    os.makedirs(save_codes_path, exist_ok=True)
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size] #保留batch维度：1*512*512
        try:
            with torch.no_grad():
                # Encode the batch of images
                codes = model.encode(batch)
                # Save the encoded codes
                np.save(f'{save_codes_path}/{start_idx:03d}.npy', codes.detach().cpu().numpy())
        except Exception as e:
            print(f"Error processing batch starting at image {start_idx}: {e}")

#====================================
# STAGE 2 Encoding images and wrist images using Emu3 vq tokenizer
#===================================
data_config = {
    'libero': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (200, 200),
        'VIDEO_ROOT': '/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all',
        'VIDEO_CODES_SAVE_1': '/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all_codes_200',
        'VIDEO_CODES_SAVE_2': '/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all_gripper_codes_200'
    }
}

process_data = 'libero'
path = "/liujinxin/zhy/ICLR2026/pretrain/Emu3-VisionTokenizer"
image_encoder = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)

config = data_config.get(process_data)
processor.min_pixels = config['min_pixels']
interval = config['interval'] #sampling rate
SIZE = config['SIZE']
VIDEO_ROOT = config['VIDEO_ROOT']
VIDEO_CODES_SAVE_1 = config['VIDEO_CODES_SAVE_1']
VIDEO_CODES_SAVE_2 = config['VIDEO_CODES_SAVE_2']
os.makedirs(VIDEO_CODES_SAVE_1, exist_ok=True)
os.makedirs(VIDEO_CODES_SAVE_2, exist_ok=True)

#multi-processing
try:
    rank = int(sys.argv[1])
except Exception as e:
    print(f"Error parsing rank: {e}")
videos = sorted(os.listdir(VIDEO_ROOT))[rank::8]

for video in tqdm(videos, desc="Processing videos"):
    images, image_paths = load_images(os.path.join(VIDEO_ROOT, video, 'images'), SIZE, interval)
    gripper_images, gripper_image_paths = load_images(os.path.join(VIDEO_ROOT, video,'gripper_images'), SIZE, interval)

    processed_codes_path_1 = os.path.join(VIDEO_CODES_SAVE_1, video) 
    image_level_enc(images, image_encoder, processor, processed_codes_path_1)
    processed_codes_path_2 = os.path.join(VIDEO_CODES_SAVE_2, video)
    image_level_enc(images, image_encoder, processor, processed_codes_path_2)


# videos = sorted(os.listdir(VIDEO_ROOT))
# for video in tqdm(videos):
#     src_image_code_folder = os.path.join("/liujinxin/zhy/UniVLA/datasets/processed_data/libero_all_codes_200_augshift",video)
#     src_gripper_image_code_folder = os.path.join("/liujinxin/zhy/UniVLA/datasets/processed_data/libero_all_gripper_codes_200_augshift",video)
#     processed_codes_path_1 = os.path.join(VIDEO_CODES_SAVE_1, video)
#     processed_codes_path_2 = os.path.join(VIDEO_CODES_SAVE_2, video)
#     if os.path.exists(processed_codes_path_1) and os.path.exists(processed_codes_path_2):
#         print(f"Skipping video {video} as it has already been processed.")
#         continue
#     os.makedirs(processed_codes_path_1, exist_ok=True)
#     os.makedirs(processed_codes_path_2, exist_ok=True)
#     if os.path.exists(src_image_code_folder):
#         for filename in os.listdir(src_image_code_folder):
#             src_file = os.path.join(src_image_code_folder, filename)
#             dst_file = os.path.join(processed_codes_path_1, filename)
#             if os.path.isfile(src_file):
#                 shutil.copy2(src_file, dst_file)

#     # 复制 src_gripper_image_code_folder 到 processed_codes_path_2
#     if os.path.exists(src_gripper_image_code_folder):
#         for filename in os.listdir(src_gripper_image_code_folder):
#             src_file = os.path.join(src_gripper_image_code_folder, filename)
#             dst_file = os.path.join(processed_codes_path_2, filename)
#             if os.path.isfile(src_file):
#                 shutil.copy2(src_file, dst_file)