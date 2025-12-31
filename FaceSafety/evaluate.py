import os
import json
import time
from tqdm import tqdm
from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
from config import THRESHOLD, TEST_RESULTS_PATH


class BlacklistEvaluator:
    def __init__(self):
        self.model = FaceRecognitionModel()
        self.milvus = MilvusClient()

    def evaluate_test_set(self, test_set_path="test_set_info.json"):
        """评估测试集性能"""
        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"测试集文件 {test_set_path} 不存在，请先运行 build_blacklist.py")

        # 加载测试集
        with open(test_set_path, "r") as f:
            test_set = json.load(f)

        print("\n开始评估测试集...")
        start_time = time.time()

        results = {
            "threshold": THRESHOLD,
            "positive_samples": [],
            "negative_samples": [],
            "metrics": {}
        }

        # 1. 评估正样本
        print("\n评估正样本（应在黑名单中）...")
        for sample in tqdm(test_set["positive_samples"], desc="正样本评估"):
            self._evaluate_sample(sample, results["positive_samples"])

        # 2. 评估负样本
        print("\n评估负样本（不应在黑名单中）...")
        for sample in tqdm(test_set["negative_samples"], desc="负样本评估"):
            self._evaluate_sample(sample, results["negative_samples"])

        # 3. 计算指标
        print("\n计算性能指标...")
        metrics = self._calculate_metrics(results)
        results["metrics"] = metrics

        # 4. 保存结果
        with open(TEST_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n评估完成！总耗时: {total_time:.2f}秒")

        self._print_summary(metrics)
        return results

    def _evaluate_sample(self, sample, results_list):
        """评估单个样本"""
        image_path = sample["image_path"]
        expected_in_blacklist = sample["expected_in_blacklist"]

        # 提取特征
        success, feature, _ = self.model.extract_feature(image_path)

        if not success:
            results_list.append({
                "person_name": sample["person_name"],
                "image_path": image_path,
                "expected_in_blacklist": expected_in_blacklist,
                "detected": False,
                "predicted_in_blacklist": False,
                "similarity": 0.0,
                "correct": False,
                "error": "未检测到人脸"
            })
            return

        # 搜索黑名单
        is_match, person_name, similarity, face_id = self.milvus.search_face(feature)

        # 判断是否命中黑名单
        predicted_in_blacklist = is_match and similarity >= THRESHOLD

        # 判断是否正确
        correct = (predicted_in_blacklist == expected_in_blacklist)

        results_list.append({
            "person_name": sample["person_name"],
            "image_path": image_path,
            "expected_in_blacklist": expected_in_blacklist,
            "detected": True,
            "predicted_in_blacklist": predicted_in_blacklist,
            "matched_person": person_name if is_match else None,
            "similarity": similarity,
            "correct": correct
        })

    def _calculate_metrics(self, results):
        """计算性能指标"""
        # 统计结果
        positive_correct = sum(1 for s in results["positive_samples"] if s["correct"])
        positive_total = len(results["positive_samples"])
        negative_correct = sum(1 for s in results["negative_samples"] if s["correct"])
        negative_total = len(results["negative_samples"])

        # 计算TP, FP, TN, FN
        TP = sum(1 for s in results["positive_samples"] if s["predicted_in_blacklist"])  # 真阳性
        FN = positive_total - TP  # 假阴性

        FP = sum(1 for s in results["negative_samples"] if s["predicted_in_blacklist"])  # 假阳性
        TN = negative_total - FP  # 真阴性

        # 计算指标
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0  # 误报率
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0  # 漏报率

        # 计算每张图片平均耗时
        total_images = positive_total + negative_total
        avg_time_per_image = (results.get("total_time", 0) / total_images) if total_images > 0 else 0

        return {
            "total_samples": total_images,
            "positive_samples": positive_total,
            "negative_samples": negative_total,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "accuracy": accuracy * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1_score * 100,
            "false_positive_rate": fpr * 100,
            "false_negative_rate": fnr * 100,
            "avg_time_per_image_ms": avg_time_per_image * 1000
        }

    def _print_summary(self, metrics):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("黑名单系统性能评估报告")
        print("=" * 60)
        print(f"测试样本总数: {metrics['total_samples']}")
        print(f"正样本数 (应在黑名单): {metrics['positive_samples']}")
        print(f"负样本数 (不应在黑名单): {metrics['negative_samples']}")
        print("-" * 60)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.2f}%")
        print(f"精确率 (Precision): {metrics['precision']:.2f}%")
        print(f"召回率 (Recall): {metrics['recall']:.2f}%")
        print(f"F1分数: {metrics['f1_score']:.2f}%")
        print("-" * 60)
        print(f"误报率 (FPR): {metrics['false_positive_rate']:.2f}%")
        print(f"漏报率 (FNR): {metrics['false_negative_rate']:.2f}%")
        print("-" * 60)
        print(f"平均识别时间: {metrics['avg_time_per_image_ms']:.2f}ms/张")
        print("=" * 60)
        print("\n关键指标说明:")
        print("- 误报率: 不在黑名单的人被错误识别为在黑名单的比例，应尽可能低")
        print("- 漏报率: 在黑名单的人未被识别的比例，应尽可能低")
        print("- 召回率: 在黑名单的人被正确识别的比例，应尽可能高")

    def analyze_errors(self, results_path=TEST_RESULTS_PATH):
        """分析错误案例"""
        if not os.path.exists(results_path):
            print("结果文件不存在，无法分析错误")
            return

        with open(results_path, "r") as f:
            results = json.load(f)

        print("\n" + "=" * 50)
        print("错误案例分析")
        print("=" * 50)

        # 假阳性（误报）
        false_positives = [s for s in results["negative_samples"] if not s["correct"]]
        print(f"\n假阳性案例 (误报): {len(false_positives)} 个")
        for i, fp in enumerate(false_positives[:5]):  # 只显示前5个
            print(f"{i + 1}. {fp['person_name']}")
            print(f"   图片: {fp['image_path']}")
            print(f"   匹配到: {fp.get('matched_person', 'Unknown')}")
            print(f"   相似度: {fp['similarity']:.4f}")

        # 假阴性（漏报）
        false_negatives = [s for s in results["positive_samples"] if not s["correct"]]
        print(f"\n假阴性案例 (漏报): {len(false_negatives)} 个")
        for i, fn in enumerate(false_negatives[:5]):  # 只显示前5个
            print(f"{i + 1}. {fn['person_name']}")
            print(f"   图片: {fn['image_path']}")
            print(f"   匹配到: {fn.get('matched_person', 'None')}")
            print(f"   相似度: {fn['similarity']:.4f}")

        # 未检测到人脸的案例
        no_face_samples = [s for s in results["positive_samples"] + results["negative_samples"]
                           if not s.get("detected", True)]
        print(f"\n未检测到人脸: {len(no_face_samples)} 张")

        return false_positives, false_negatives, no_face_samples


if __name__ == "__main__":
    evaluator = BlacklistEvaluator()
    results = evaluator.evaluate_test_set()
    evaluator.analyze_errors()
