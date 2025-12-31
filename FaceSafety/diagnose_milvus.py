import time
from pymilvus import connections, utility, Collection

print("="*60)
print("Milvus连接诊断")
print("="*60)

# 1. 测试基础连接
try:
    connections.connect("default", host="localhost", port="19531", timeout=5)
    print("✅ 基础连接成功")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# 2. 测试简单操作
try:
    start = time.time()
    has_col = utility.has_collection("test_connection")
    print(f"✅ has_collection操作成功，耗时: {time.time()-start:.2f}s")
    print(f"   test_connection存在: {has_col}")
except Exception as e:
    print(f"❌ has_collection失败: {e}")

# 3. 测试创建临时集合
try:
    start = time.time()
    if utility.has_collection("temp_test"):
        utility.drop_collection("temp_test")
    
    from pymilvus import FieldSchema, CollectionSchema, DataType
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields, "测试集合")
    collection = Collection("temp_test", schema)
    
    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 64}
    }
    collection.create_index("vec", index_params)
    
    print(f"✅ 创建测试集合成功，耗时: {time.time()-start:.2f}s")
    
    # 清理
    utility.drop_collection("temp_test")
    
except Exception as e:
    print(f"❌ 创建集合失败: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
