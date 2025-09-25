import numpy as np
import pickle

data = "/liujinxin/zhy/ICLR2026/datasets/libero/data/meta/libero_all_norm_patched.pkl"

with open(data, 'rb') as f:
    data = pickle.load(f)

print(data[0]["text"])
print(data[0]["reward"][0])
print(data[0]["reward"][100])
print(data[0]["returnToGo"][-1])