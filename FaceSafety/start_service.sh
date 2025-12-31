#!/bin/bash
# start_service.sh

echo "启动人脸检测服务..."
echo "Milvus集合: face_library_20251230"
echo "服务地址: http://127.0.0.1:9876"
echo "API文档: http://127.0.0.1:9876/docs"
echo "按 Ctrl+C 停止服务"
echo "========================================"

# 激活虚拟环境（如有必要）
# source /path/to/venv/bin/activate

# 启动服务
python face_detection_service.py
