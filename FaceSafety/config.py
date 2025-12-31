import os

# 数据集路径
DATASET_PATH = "path/to/archive/lfw-deepfunneled/lfw-deepfunneled"

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19531"
COLLECTION_NAME = "lfw_blacklist"

# 模型配置 - 强制GPU
MODEL_NAME = "buffalo_l"
DEVICE = "cuda:1"  # 或 "cuda:1" 如果有多个GPU

# 验证GPU是否可用
import torch
if torch.cuda.is_available():
    print(f"🎉 PyTorch检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA版本: {torch.version.cuda}")
else:
    print("⚠️ 警告：PyTorch未检测到GPU")

# 相似度阈值
THRESHOLD = 0.50

# 人脸的det_score阈值（与图片质量有关，图片质量越低该值越低）
DETSCOREBAR = 0.60

"""
======================================================================
开始搜索最优阈值...
2025-12-29 19:23:48,641 - INFO - 开始搜索最优阈值，范围: (0.3, 0.7), 步长: 0.01
2025-12-29 19:23:48,651 - INFO - 预提取所有样本特征...
阈值搜索: 100%|████████████████████████████████████████████████████████████████████████████████| 41/41 [36:34<00:00, 53.52s/it]
2025-12-29 20:03:37,052 - INFO - 最优阈值: 0.340 (F1=97.50%)
"""

# 测试集结果保存路径
TEST_RESULTS_PATH = "test_results.json"

# 批量处理大小（GPU可适当增大）
BATCH_SIZE = 32 if "cuda" in DEVICE else 4

INPUT_PATH = "path/to/FaceSafety/Input/20251230_01"
FACE_LIBRARY_COLLECTION = "face_library_20251230"
