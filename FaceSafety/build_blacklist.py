import os
import json
import time
from collections import defaultdict
from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
from config import DATASET_PATH, THRESHOLD, BATCH_SIZE, DEVICE


class BlacklistBuilder:
    def __init__(self):
        # 添加设备检测日志
        print("="*60)
        print("初始化黑名单构建器...")
        print(f"数据集路径: {DATASET_PATH}")
        print(f"设备配置: {DEVICE}")
        print("="*60)
        
        self.model = FaceRecognitionModel(device=DEVICE)  # 传递设备参数
        self.milvus = MilvusClient()
        self.person_images = defaultdict(list)

    def scan_dataset(self):
        """扫描数据集，统计每个人的图片"""
        print("开始扫描数据集...")
        for person_name in os.listdir(DATASET_PATH):
            person_dir = os.path.join(DATASET_PATH, person_name)

            if not os.path.isdir(person_dir):
                continue

            images = [img for img in os.listdir(person_dir)
                      if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if images:
                self.person_images[person_name] = sorted(images)

        total_people = len(self.person_images)
        total_images = sum(len(images) for images in self.person_images.values())
        print(f"扫描完成: 共 {total_people} 人, {total_images} 张图片")

        # 统计分布
        multi_image_people = {k: v for k, v in self.person_images.items() if len(v) > 1}
        single_image_people = {k: v for k, v in self.person_images.items() if len(v) == 1}

        print(f"有多张图片的人: {len(multi_image_people)} 人")
        print(f"只有一张图片的人: {len(single_image_people)} 人")

        return multi_image_people, single_image_people

    def build_blacklist(self, multi_image_people):
        """构建黑名单库（只包含多张图片的第一张）"""
        print("\n开始构建黑名单库...")
        print(f"预计处理 {len(multi_image_people)} 人，每处理10人会打印一次进度...")
        
        # 创建Milvus集合
        self.milvus.create_collection()
        
        blacklist_data = []
        success_count = 0
        fail_count = 0
        start_time = time.time()
        
        # 批量处理以提高性能
        batch_data = []
        
        print("======Yes======")
        for idx, (person_name, images) in enumerate(multi_image_people.items()):
            # 只取第一张（_0001.jpg）
            first_image = images[0]
            image_path = os.path.join(DATASET_PATH, person_name, first_image)
            
            # 提取特征
            success, feature, _ = self.model.extract_feature(image_path)
            
            if success:
                face_id = f"{person_name}_0001"
                batch_data.append({
                    "face_id": face_id,
                    "vector": feature,
                    "person_name": person_name,
                    "image_path": image_path
                })
                success_count += 1
                
                # 批量插入
                if len(batch_data) >= BATCH_SIZE:
                    self.milvus.insert_faces(batch_data)
                    batch_data = []
                
                # 每10人打印一次进度
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"✅ 已处理 {idx+1}/{len(multi_image_people)} 人, 成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed:.2f}s")
                    
            else:
                print(f"⚠️ 警告: {image_path} 未检测到人脸，跳过")
                fail_count += 1
        
        # 插入剩余数据
        if batch_data:
            self.milvus.insert_faces(batch_data)
        
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"黑名单库构建完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"处理速度: {len(multi_image_people)/total_time:.2f}人/秒")
        print(f"成功添加: {success_count} 人")
        print(f"失败跳过: {fail_count} 人")
        print("="*50)
        
        return blacklist_data


    def build_test_set(self, multi_image_people, single_image_people):
        """构建测试集"""
        print("\n开始构建测试集...")

        test_set = {
            "positive_samples": [],  # 正样本：多张图片的非第一张（在黑名单中）
            "negative_samples": []  # 负样本：单张图片的人（不在黑名单中）
        }

        # 1. 正样本：多张图片的人（排除第一张）
        positive_count = 0
        for person_name, images in multi_image_people.items():
            for img in images[1:]:  # 跳过第一张
                image_path = os.path.join(DATASET_PATH, person_name, img)
                test_set["positive_samples"].append({
                    "person_name": person_name,
                    "image_path": image_path,
                    "expected_in_blacklist": True
                })
                positive_count += 1

        # 2. 负样本：只有一张图片的人
        negative_count = 0
        for person_name, images in single_image_people.items():
            image_path = os.path.join(DATASET_PATH, person_name, images[0])
            test_set["negative_samples"].append({
                "person_name": person_name,
                "image_path": image_path,
                "expected_in_blacklist": False
            })
            negative_count += 1

        print(f"测试集构建完成！")
        print(f"正样本: {positive_count} 张（应在黑名单中）")
        print(f"负样本: {negative_count} 张（不应在黑名单中）")
        print(f"总计: {positive_count + negative_count} 张")

        # 保存测试集信息
        with open("test_set_info.json", "w") as f:
            json.dump(test_set, f, indent=2)

        return test_set

    def run(self):
        """执行完整流程"""
        # 1. 扫描数据集
        multi_image_people, single_image_people = self.scan_dataset()
        
        # 2. 构建黑名单
        blacklist = self.build_blacklist(multi_image_people)
        
        # 3. 构建测试集
        test_set = self.build_test_set(multi_image_people, single_image_people)
        
        # 4. 打印统计
        print("\n" + "="*60)
        print("数据集统计摘要")
        print("="*60)
        print(f"黑名单人数: {len(blacklist)}")
        print(f"黑名单特征向量数: {self.milvus.get_collection_stats()}")
        print(f"测试集正样本数: {len(test_set['positive_samples'])}")
        print(f"测试集负样本数: {len(test_set['negative_samples'])}")
        print(f"总计测试图片: {len(test_set['positive_samples']) + len(test_set['negative_samples'])}")
        print("="*60)
        print("\n下一步: 运行 python evaluate.py 进行评估")
        
        return blacklist, test_set


if __name__ == "__main__":
    builder = BlacklistBuilder()
    builder.run()
