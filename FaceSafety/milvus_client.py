from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility
)
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME


class MilvusClient:
    def __init__(self):
        """连接Milvus数据库"""
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"已连接到Milvus: {MILVUS_HOST}:{MILVUS_PORT}")

    def create_collection(self):
        """创建黑名单集合"""
        if utility.has_collection(COLLECTION_NAME):
            print(f"集合 {COLLECTION_NAME} 已存在，正在删除...")
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=255)
        ]

        schema = CollectionSchema(fields, "LFW黑名单人脸库")
        collection = Collection(COLLECTION_NAME, schema)

        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("vector", index_params)
        print(f"集合 {COLLECTION_NAME} 创建成功")
        return collection

    def insert_faces(self, face_data):
        """
        批量插入人脸数据
        face_data: list of dict, 每个dict包含 face_id, vector, person_name, image_path
        """
        collection = Collection(COLLECTION_NAME)

        entities = [
            [item["face_id"] for item in face_data],
            [item["vector"].tolist() for item in face_data],
            [item["person_name"] for item in face_data],
            [item["image_path"] for item in face_data]
        ]

        collection.insert(entities)
        collection.flush()
        print(f"成功插入 {len(face_data)} 条人脸记录")

    def search_face(self, query_vector, top_k=1):
        """
        搜索相似人脸
        返回: (is_match, person_name, similarity, face_id)
        """
        collection = Collection(COLLECTION_NAME)
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["person_name", "face_id"]
        )

        if results[0] and len(results[0]) > 0:
            top_result = results[0][0]
            return True, top_result.entity.get('person_name'), top_result.score, top_result.entity.get('face_id')

        return False, None, 0.0, None

    def get_collection_stats(self):
        """获取集合统计信息"""
        collection = Collection(COLLECTION_NAME)
        return collection.num_entities
