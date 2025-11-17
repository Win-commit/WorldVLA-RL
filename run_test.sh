#!/bin/bash
# 运行 compute_flow_loss 测试脚本

# 设置环境变量
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_BASE_URL="https://api.bandw.top"

# 运行测试
echo "开始运行 compute_flow_loss 测试..."
python models/action_patches/test_compute_flow_loss.py

# 检查退出码
if [ $? -eq 0 ]; then
    echo "✅ 测试成功完成！"
    exit 0
else
    echo "❌ 测试失败！"
    exit 1
fi
