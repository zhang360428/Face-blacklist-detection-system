import os
import time
from collections import defaultdict
from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
from config import DEVICE, BATCH_SIZE


class FaceLibraryAppender:
    def __init__(self, input_path, collection_name):
        """
        初始化人脸库追加器 - 用于向现有人脸库增量添加人脸
        :param input_path: 新图片目录路径
        :param collection_name: 已存在的Milvus集合名称（如face_library_20251230）
        """
        print("="*60)
        print("初始化人脸库追加器...")
        print(f"输入目录: {input_path}")
        print(f"目标Milvus集合: {collection_name}")
        print(f"设备配置: {DEVICE}")
        print("="*60)
        
        # 使用已存在的集合名称（不创建新集合）
        import milvus_client
        milvus_client.COLLECTION_NAME = collection_name
        
        self.model = FaceRecognitionModel(device=DEVICE)
        self.milvus = MilvusClient()
        self.input_path = input_path
        self.collection_name = collection_name
        self.person_images = defaultdict(list)

    def scan_directory(self):
        """扫描目录，按人名分组所有图片（复用原有逻辑）"""
        print(f"\n开始扫描目录: {self.input_path}")
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"目录不存在: {self.input_path}")
        
        # 获取所有图片文件
        all_images = [f for f in os.listdir(self.input_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not all_images:
            raise ValueError(f"目录中没有找到图片文件: {self.input_path}")
        
        # 按人名分组（假设文件名格式：personname_0001.jpg）
        for image_file in all_images:
            base_name = os.path.splitext(image_file)[0]
            person_name = base_name.rsplit('_', 1)[0]
            image_path = os.path.join(self.input_path, image_file)
            self.person_images[person_name].append(image_path)
        
        # 统计信息
        total_people = len(self.person_images)
        total_images = sum(len(images) for images in self.person_images.values())
        
        print(f"扫描完成: 共 {total_people} 人, {total_images} 张图片")
        
        # 显示分布情况
        multi_image_people = {k: v for k, v in self.person_images.items() if len(v) > 1}
        
        print(f"有多张图片的人: {len(multi_image_people)} 人")
        for person, images in multi_image_people.items():
            print(f"  - {person}: {len(images)} 张")
        
        print(f"只有一张图片的人: {total_people - len(multi_image_people)} 人")
        
        return self.person_images

    def check_existing_faces(self, face_ids):
        """
        检查face_id是否已存在于Milvus中
        :param face_ids: 待检查的face_id列表
        :return: 已存在的face_id集合
        """
        print(f"\n检查 {len(face_ids)} 个face_id是否已存在...")
        
        existing_ids = set()
        try:
            # 使用 Milvus 的 query 接口批量查询
            expr = f"face_id in {face_ids}"
            results = self.milvus.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["face_id"]
            )
            existing_ids = {item["face_id"] for item in results}
            print(f"发现 {len(existing_ids)} 个已存在的face_id将被跳过")
        except Exception as e:
            print(f"⚠️ 检查重复时出错（可能是集合为空）: {str(e)}")
        
        return existing_ids

    def append_faces(self, person_images):
        """向现有人脸库追加人脸（核心增量添加逻辑）"""
        print(f"\n开始向集合 '{self.collection_name}' 追加人脸数据...")
        print(f"预计处理 {len(person_images)} 人...")
        
        # 重要：不创建新集合，直接复用已存在的集合
        # self.milvus.create_collection()  # ❌ 注释掉这行
        
        total_success = 0
        total_fail = 0
        total_skip = 0
        start_time = time.time()
        
        # 批量处理
        batch_data = []
        
        # 先收集所有待处理的face_id用于去重检查
        all_face_ids = []
        for person_name, image_paths in person_images.items():
            for image_path in image_paths:
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                all_face_ids.append(base_filename)
        
        # 检查已存在的face_id
        existing_face_ids = self.check_existing_faces(all_face_ids)
        
        # 处理每个人
        for person_name, image_paths in person_images.items():
            person_success = 0
            person_fail = 0
            person_skip = 0
            
            for idx, image_path in enumerate(image_paths):
                try:
                    # 生成face_id
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    face_id = base_filename
                    
                    # 检查是否已存在
                    if face_id in existing_face_ids:
                        print(f"⏭️ 跳过重复: {face_id} 已存在于库中")
                        person_skip += 1
                        continue
                    
                    # 提取特征
                    success, feature, _ = self.model.extract_feature(image_path)
                    
                    if success:
                        batch_data.append({
                            "face_id": face_id,
                            "vector": feature,
                            "person_name": person_name,
                            "image_path": image_path
                        })
                        person_success += 1
                        
                        # 批量插入
                        if len(batch_data) >= BATCH_SIZE:
                            self.milvus.insert_faces(batch_data)
                            batch_data = []
                    else:
                        print(f"⚠️ 警告: {image_path} 未检测到人脸，跳过")
                        person_fail += 1
                        
                except Exception as e:
                    print(f"❌ 错误处理 {image_path}: {str(e)}")
                    person_fail += 1
            
            total_success += person_success
            total_fail += person_fail
            total_skip += person_skip
            
            # 每处理完一个人打印一次进度
            status_msg = f"✅ {person_name}: 成功 {person_success} 张"
            if person_skip > 0:
                status_msg += f", 跳过 {person_skip} 张"
            if person_fail > 0:
                status_msg += f", 失败 {person_fail} 张"
            print(status_msg)
        
        # 插入剩余数据
        if batch_data:
            self.milvus.insert_faces(batch_data)
        
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("人脸库追加完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"处理速度: {total_success/total_time:.2f}张/秒")
        print(f"成功添加: {total_success} 张新图片")
        print(f"跳过重复: {total_skip} 张图片")
        print(f"失败跳过: {total_fail} 张图片")
        print(f"涉及人数: {len(person_images)} 人")
        print(f"集合名称: {self.collection_name}")
        print("="*50)
        
        return total_success, total_fail, total_skip

    def run(self):
        """执行完整追加流程"""
        # 1. 扫描目录
        person_images = self.scan_directory()
        
        # 2. 追加人脸到现有库
        success_count, fail_count, skip_count = self.append_faces(person_images)
        
        # 3. 打印统计
        print("\n" + "="*60)
        print("人脸库追加统计摘要")
        print("="*60)
        print(f"总人数: {len(person_images)}")
        print(f"总图片数: {success_count + fail_count + skip_count}")
        print(f"成功入库: {success_count} (新增)")
        print(f"跳过重复: {skip_count}")
        print(f"失败跳过: {fail_count}")
        print(f"Milvus集合: {self.collection_name}")
        print(f"当前总特征向量数: {self.milvus.get_collection_stats()}")
        print("="*60)
        
        return success_count, fail_count, skip_count


def get_current_collection_name():
    """
    自动获取最新的人脸库集合名称
    按时间排序: face_library_20251230 > face_library_20251229
    """
    try:
        from pymilvus import MilvusClient as RawMilvusClient
        
        # 使用原始Milvus客户端查询所有集合
        raw_client = RawMilvusClient(uri="http://localhost:19531")
        collections = raw_client.list_collections()
        
        # 过滤出人脸库集合
        face_collections = [c for c in collections if c.startswith("face_library_")]
        
        if not face_collections:
            print("⚠️ 未找到任何人脸库集合，将使用默认名称")
            return "face_library_20251230"
        
        # 按名称排序（时间倒序）
        face_collections.sort(reverse=True)
        latest = face_collections[0]
        print(f"自动检测到最新人脸库集合: {latest}")
        return latest
        
    except Exception as e:
        print(f"⚠️ 检测集合失败，使用默认名称: {str(e)}")
        return "face_library_20251230"


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    INPUT_PATH = "/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20260108_01"
    
    # 选项1: 自动检测最新的集合（推荐）
    COLLECTION_NAME = get_current_collection_name()
    
    # 选项2: 手动指定集合名称（如果自动检测不准）
    # COLLECTION_NAME = "face_library_20251230"
    # =================================================
    
    try:
        appender = FaceLibraryAppender(input_path=INPUT_PATH, collection_name=COLLECTION_NAME)
        appender.run()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
