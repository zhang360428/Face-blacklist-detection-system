import os
import time
from collections import defaultdict
from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
from config import DEVICE, BATCH_SIZE


class FaceLibraryBuilder:
    def __init__(self, input_path, collection_name):
        """
        初始化人脸库构建器
        :param input_path: 图片目录路径
        :param collection_name: Milvus集合名称（建议使用新名称，如face_library_20251230）
        """
        print("="*60)
        print("初始化人脸库构建器...")
        print(f"输入目录: {input_path}")
        print(f"Milvus集合: {collection_name}")
        print(f"设备配置: {DEVICE}")
        print("="*60)
        
        # 临时修改集合名称（不影响原黑名单配置）
        import milvus_client
        milvus_client.COLLECTION_NAME = collection_name
        
        self.model = FaceRecognitionModel(device=DEVICE)
        self.milvus = MilvusClient()
        self.input_path = input_path
        self.collection_name = collection_name
        self.person_images = defaultdict(list)

    def scan_directory(self):
        """扫描目录，按人名分组所有图片"""
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
            # 提取人名（移除扩展名和序号部分）
            base_name = os.path.splitext(image_file)[0]  # 移除扩展名
            person_name = base_name.rsplit('_', 1)[0]    # 从最后一个_分割，取前半部分
            
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

    def build_face_library(self, person_images):
        """构建人脸库（包含每个人的所有图片）"""
        print(f"\n开始构建人脸库到集合 '{self.collection_name}'...")
        print(f"预计处理 {len(person_images)} 人...")
        
        # 创建Milvus集合
        self.milvus.create_collection()
        
        total_success = 0
        total_fail = 0
        start_time = time.time()
        
        # 批量处理以提高性能
        batch_data = []
        
        for person_name, image_paths in person_images.items():
            person_success = 0
            person_fail = 0
            
            for idx, image_path in enumerate(image_paths):
                try:
                    # 提取特征
                    success, feature, _ = self.model.extract_feature(image_path)
                    
                    if success:
                        # 生成face_id（格式：personname_0001）
                        base_filename = os.path.splitext(os.path.basename(image_path))[0]
                        face_id = base_filename
                        
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
            
            # 每处理完一个人打印一次进度
            print(f"✅ {person_name}: 成功 {person_success} 张, 失败 {person_fail} 张")
        
        # 插入剩余数据
        if batch_data:
            self.milvus.insert_faces(batch_data)
        
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("人脸库构建完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"处理速度: {total_success/total_time:.2f}张/秒")
        print(f"成功添加: {total_success} 张图片")
        print(f"失败跳过: {total_fail} 张图片")
        print(f"涉及人数: {len(person_images)} 人")
        print(f"集合名称: {self.collection_name}")
        print("="*50)
        
        return total_success, total_fail

    def run(self):
        """执行完整流程"""
        # 1. 扫描目录
        person_images = self.scan_directory()
        
        # 2. 构建人脸库
        success_count, fail_count = self.build_face_library(person_images)
        
        # 3. 打印统计
        print("\n" + "="*60)
        print("人脸库统计摘要")
        print("="*60)
        print(f"总人数: {len(person_images)}")
        print(f"总图片数: {success_count + fail_count}")
        print(f"成功入库: {success_count}")
        print(f"失败跳过: {fail_count}")
        print(f"Milvus集合: {self.collection_name}")
        print(f"特征向量数: {self.milvus.get_collection_stats()}")
        print("="*60)
        
        return success_count, fail_count


if __name__ == "__main__":
    # 使用固定路径和新集合名称
    INPUT_PATH = "path/to/FaceSafety/Input/20251230_01"
    COLLECTION_NAME = "face_library_20251230"  # 新集合名称，避免与黑名单冲突
    
    try:
        builder = FaceLibraryBuilder(input_path=INPUT_PATH, collection_name=COLLECTION_NAME)
        builder.run()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
