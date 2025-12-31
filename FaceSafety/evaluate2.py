import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

from face_model import FaceRecognitionModel
from milvus_client import MilvusClient
from config import THRESHOLD, TEST_RESULTS_PATH

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """评估结果可视化工具类"""
    
    def __init__(self, save_dir: str = "evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            title: str = "matrix", save_path: Optional[str] = None):
        """绘制混淆矩阵热力图"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用 seaborn 绘制热力图
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                    xticklabels=['negtive', 'postive'], yticklabels=['negtive', 'postive'],
                    ax=ax, cbar_kws={'label': 'ratio'})
        
        # 添加数值标注
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                ax.text(j + 0.5, i + 0.7, f'\n({count})', 
                       ha='center', va='center', fontsize=10)
        
        ax.set_xlabel('predict', fontsize=12)
        ax.set_ylabel('truth', fontsize=12)
        ax.set_title(f'{title}\n(threshold={THRESHOLD})', fontsize=14, fontweight='bold')
        
        # 添加图例说明
        plt.figtext(0.5, -0.05, 
                   "Positive samples: should be in the blacklist\nNegative samples: should not be in the blacklist", 
                   ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存至: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curve(self, y_true: List[int], y_scores: List[float], 
                      title: str = "ROC曲线", save_path: Optional[str] = None):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='随机猜测')
        
        # 标记当前阈值点
        current_threshold_idx = np.argmin(np.abs(thresholds - THRESHOLD))
        ax.plot(fpr[current_threshold_idx], tpr[current_threshold_idx], 
                'ro', markersize=10, label=f'当前阈值 ({THRESHOLD})')
        
        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存至: {save_path}")
        
        plt.show()
        plt.close()
        
        return roc_auc
    
    def plot_pr_curve(self, y_true: List[int], y_scores: List[float], 
                     title: str = "精确率-召回率曲线", save_path: Optional[str] = None):
        """绘制PR曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # 计算基准线（正样本比例）
        baseline = np.mean(y_true)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR曲线 (AUC = {pr_auc:.3f})')
        ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', 
                  label=f'基准线 (正样本比例 = {baseline:.3f})')
        
        # 标记当前阈值点
        current_threshold_idx = np.argmin(np.abs(thresholds - THRESHOLD))
        if current_threshold_idx < len(recall):
            ax.plot(recall[current_threshold_idx], precision[current_threshold_idx], 
                    'ro', markersize=10, label=f'当前阈值 ({THRESHOLD})')
        
        ax.set_xlabel('召回率 (Recall)', fontsize=12)
        ax.set_ylabel('精确率 (Precision)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR曲线已保存至: {save_path}")
        
        plt.show()
        plt.close()
        
        return pr_auc
    
    def plot_similarity_distribution(self, positive_scores: List[float], 
                                   negative_scores: List[float],
                                   threshold: float = THRESHOLD,
                                   save_path: Optional[str] = None):
        """绘制相似度分布直方图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制分布图
        ax.hist(positive_scores, bins=50, alpha=0.6, label='正样本', 
                color='green', density=True)
        ax.hist(negative_scores, bins=50, alpha=0.6, label='负样本', 
                color='red', density=True)
        
        # 添加阈值线
        ax.axvline(x=threshold, color='blue', linestyle='--', 
                  linewidth=2, label=f'决策阈值 ({threshold})')
        
        ax.set_xlabel('相似度得分', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title('相似度分布图', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # 添加统计信息
        stats_text = f"正样本: μ={np.mean(positive_scores):.3f}, σ={np.std(positive_scores):.3f}\n"
        stats_text += f"负样本: μ={np.mean(negative_scores):.3f}, σ={np.std(negative_scores):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相似度分布图已保存至: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_threshold_analysis(self, thresholds: List[float], metrics: Dict[str, List[float]],
                              save_path: Optional[str] = None):
        """绘制不同阈值下的指标变化图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘制准确率
        if 'accuracy' in metrics:
            axes[0].plot(thresholds, metrics['accuracy'], 'b-', linewidth=2)
            axes[0].axvline(x=THRESHOLD, color='r', linestyle='--', label='当前阈值')
            axes[0].set_xlabel('阈值', fontsize=11)
            axes[0].set_ylabel('准确率', fontsize=11)
            axes[0].set_title('准确率 vs 阈值', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        # 绘制精确率和召回率
        if 'precision' in metrics and 'recall' in metrics:
            axes[1].plot(thresholds, metrics['precision'], 'b-', linewidth=2, label='精确率')
            axes[1].plot(thresholds, metrics['recall'], 'g-', linewidth=2, label='召回率')
            axes[1].axvline(x=THRESHOLD, color='r', linestyle='--', label='当前阈值')
            axes[1].set_xlabel('阈值', fontsize=11)
            axes[1].set_ylabel('指标值', fontsize=11)
            axes[1].set_title('精确率 & 召回率 vs 阈值', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        # 绘制FPR和FNR
        if 'fpr' in metrics and 'fnr' in metrics:
            axes[2].plot(thresholds, metrics['fpr'], 'r-', linewidth=2, label='FPR')
            axes[2].plot(thresholds, metrics['fnr'], 'm-', linewidth=2, label='FNR')
            axes[2].axvline(x=THRESHOLD, color='r', linestyle='--', label='当前阈值')
            axes[2].set_xlabel('阈值', fontsize=11)
            axes[2].set_ylabel('错误率', fontsize=11)
            axes[2].set_title('误报率 & 漏报率 vs 阈值', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
        
        # 绘制F1分数
        if 'f1_score' in metrics:
            axes[3].plot(thresholds, metrics['f1_score'], 'y-', linewidth=2, label='F1分数')
            axes[3].axvline(x=THRESHOLD, color='r', linestyle='--', label='当前阈值')
            axes[3].set_xlabel('阈值', fontsize=11)
            axes[3].set_ylabel('F1分数', fontsize=11)
            axes[3].set_title('F1分数 vs 阈值', fontsize=12, fontweight='bold')
            axes[3].legend()
            axes[3].grid(alpha=0.3)
        
        plt.suptitle('阈值敏感性分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"阈值分析图已保存至: {save_path}")
        
        plt.show()
        plt.close()


class BlacklistEvaluator:
    """黑名单系统评估器"""
    
    def __init__(self, model: Optional[FaceRecognitionModel] = None, 
                 milvus: Optional[MilvusClient] = None,
                 save_dir: str = "evaluation_results"):
        """
        初始化评估器
        
        Args:
            model: 人脸识别模型实例
            milvus: Milvus客户端实例
            save_dir: 结果保存目录
        """
        self.model = model or FaceRecognitionModel()
        self.milvus = milvus or MilvusClient()
        self.visualizer = EvaluationVisualizer(save_dir)
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "reports"), exist_ok=True)
        
        logger.info(f"评估器初始化完成，结果将保存至: {save_dir}")
    
    def evaluate_test_set(self, test_set_path: str = "test_set_info.json",
                         generate_plots: bool = True) -> Dict[str, Any]:
        """
        评估测试集性能
        
        Args:
            test_set_path: 测试集文件路径
            generate_plots: 是否生成可视化图表
        
        Returns:
            评估结果字典
        """
        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"测试集文件 {test_set_path} 不存在，请先运行 build_blacklist.py")
        
        # 加载测试集
        logger.info("加载测试集数据...")
        with open(test_set_path, "r", encoding='utf-8') as f:
            test_set = json.load(f)
        
        logger.info(f"测试集加载完成: {len(test_set.get('positive_samples', []))} 正样本, "
                   f"{len(test_set.get('negative_samples', []))} 负样本")
        
        start_time = time.time()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "threshold": THRESHOLD,
            "positive_samples": [],
            "negative_samples": [],
            "metrics": {},
            "summary": {}
        }
        
        # 评估样本
        logger.info("开始评估正样本...")
        for sample in tqdm(test_set["positive_samples"], desc="正样本评估"):
            self._evaluate_sample(sample, results["positive_samples"])
        
        logger.info("开始评估负样本...")
        for sample in tqdm(test_set["negative_samples"], desc="负样本评估"):
            self._evaluate_sample(sample, results["negative_samples"])
        
        # 计算指标
        logger.info("计算性能指标...")
        metrics = self._calculate_detailed_metrics(results)
        results["metrics"] = metrics
        
        # 保存结果
        self._save_results(results)
        
        # 生成可视化
        if generate_plots:
            logger.info("生成可视化图表...")
            self._generate_visualizations(results)
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        logger.info(f"评估完成！总耗时: {total_time:.2f}秒")
        self._print_summary(metrics)
        
        return results
    
    def _evaluate_sample(self, sample: Dict[str, Any], results_list: List[Dict[str, Any]]):
        """评估单个样本"""
        image_path = sample["image_path"]
        expected_in_blacklist = sample["expected_in_blacklist"]
        person_name = sample.get("person_name", "Unknown")
        
        # 提取特征
        success, feature, _ = self.model.extract_feature(image_path)
        
        if not success:
            results_list.append({
                "person_name": person_name,
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
        is_match, matched_person, similarity, face_id = self.milvus.search_face(feature)
        
        # 判断是否命中黑名单
        predicted_in_blacklist = is_match and similarity >= THRESHOLD
        
        # 判断是否正确
        correct = (predicted_in_blacklist == expected_in_blacklist)
        
        results_list.append({
            "person_name": person_name,
            "image_path": image_path,
            "expected_in_blacklist": expected_in_blacklist,
            "detected": True,
            "predicted_in_blacklist": predicted_in_blacklist,
            "matched_person": matched_person if is_match else None,
            "matched_face_id": face_id if is_match else None,
            "similarity": similarity,
            "correct": correct
        })
    
    def _calculate_detailed_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算详细的性能指标"""
        # 提取数据
        positive_samples = results["positive_samples"]
        negative_samples = results["negative_samples"]
        
        # 真实标签和预测标签
        y_true = ([1] * len(positive_samples) + 
                 [0] * len(negative_samples))
        y_pred = ([1 if s["predicted_in_blacklist"] else 0 
                  for s in positive_samples] +
                 [1 if s["predicted_in_blacklist"] else 0 
                  for s in negative_samples])
        y_scores = ([s["similarity"] for s in positive_samples] +
                   [s["similarity"] for s in negative_samples])
        
        # 统计基础数据
        TP = sum(1 for s in positive_samples if s["predicted_in_blacklist"])
        FN = len(positive_samples) - TP
        FP = sum(1 for s in negative_samples if s["predicted_in_blacklist"])
        TN = len(negative_samples) - FP
        
        # 计算指标
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
        
        # 计算每张图片平均耗时
        total_images = len(positive_samples) + len(negative_samples)
        avg_time_per_image = results.get("total_time", 0) / total_images if total_images > 0 else 0
        
        # 计算样本统计信息
        positive_similarities = [s["similarity"] for s in positive_samples if s["detected"]]
        negative_similarities = [s["similarity"] for s in negative_samples if s["detected"]]
        
        metrics = {
            # 基本统计
            "total_samples": total_images,
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples),
            "detected_faces": len(positive_samples) + len(negative_samples) - 
                            sum(1 for s in positive_samples + negative_samples if not s["detected"]),
            
            # 混淆矩阵
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            
            # 核心指标
            "accuracy": accuracy * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1_score_val * 100,
            "false_positive_rate": fpr * 100,
            "false_negative_rate": fnr * 100,
            
            # 时间性能
            "avg_time_per_image_ms": avg_time_per_image * 1000,
            
            # 相似度统计
            "positive_similarity_mean": np.mean(positive_similarities) if positive_similarities else 0,
            "positive_similarity_std": np.std(positive_similarities) if positive_similarities else 0,
            "negative_similarity_mean": np.mean(negative_similarities) if negative_similarities else 0,
            "negative_similarity_std": np.std(negative_similarities) if negative_similarities else 0,
        }
        
        # 存储用于可视化的数据
        results["y_true"] = y_true
        results["y_pred"] = y_pred
        results["y_scores"] = y_scores
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        # 保存完整结果
        results_path = os.path.join(self.save_dir, "reports", "evaluation_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存简洁报告
        summary_path = os.path.join(self.save_dir, "reports", "summary.json")
        summary = {
            "timestamp": results["timestamp"],
            "threshold": results["threshold"],
            "metrics": results["metrics"]
        }
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存至: {self.save_dir}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """生成所有可视化图表"""
        plots_dir = os.path.join(self.save_dir, "plots")
        
        # 准备数据
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        y_scores = results["y_scores"]
        
        positive_scores = [s["similarity"] for s in results["positive_samples"] if s["detected"]]
        negative_scores = [s["similarity"] for s in results["negative_samples"] if s["detected"]]
        
        # 1. 混淆矩阵
        self.visualizer.plot_confusion_matrix(
            y_true, y_pred,
            title="黑名单系统混淆矩阵",
            save_path=os.path.join(plots_dir, "confusion_matrix.png")
        )
        
        # 2. ROC曲线
        roc_auc = self.visualizer.plot_roc_curve(
            y_true, y_scores,
            title="黑名单系统ROC曲线",
            save_path=os.path.join(plots_dir, "roc_curve.png")
        )
        results["metrics"]["roc_auc"] = roc_auc
        
        # 3. PR曲线
        pr_auc = self.visualizer.plot_pr_curve(
            y_true, y_scores,
            title="黑名单系统PR曲线",
            save_path=os.path.join(plots_dir, "pr_curve.png")
        )
        results["metrics"]["pr_auc"] = pr_auc
        
        # 4. 相似度分布
        self.visualizer.plot_similarity_distribution(
            positive_scores, negative_scores,
            threshold=THRESHOLD,
            save_path=os.path.join(plots_dir, "similarity_distribution.png")
        )
        
        logger.info("所有可视化图表生成完成")
    
    def search_optimal_threshold(self, test_set_path: str = "test_set_info.json",
                                threshold_range: Tuple[float, float] = (0.3, 0.9),
                                step: float = 0.01) -> Dict[float, Dict[str, float]]:
        """
        搜索最优阈值
        
        Args:
            test_set_path: 测试集路径
            threshold_range: 阈值搜索范围
            step: 阈值步长
        
        Returns:
            不同阈值下的性能指标
        """
        logger.info(f"开始搜索最优阈值，范围: {threshold_range}, 步长: {step}")
        
        # 加载测试集
        with open(test_set_path, "r", encoding='utf-8') as f:
            test_set = json.load(f)
        
        # 提前提取所有特征
        all_samples = []
        all_features = []
        all_labels = []
        
        logger.info("预提取所有样本特征...")
        for sample in test_set["positive_samples"] + test_set["negative_samples"]:
            image_path = sample["image_path"]
            expected = sample["expected_in_blacklist"]
            
            success, feature, _ = self.model.extract_feature(image_path)
            if success:
                all_samples.append(sample)
                all_features.append(feature)
                all_labels.append(1 if expected else 0)
        
        # 搜索阈值
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = {}
        
        for threshold in tqdm(thresholds, desc="阈值搜索"):
            y_pred = []
            
            for feature, label in zip(all_features, all_labels):
                is_match, _, similarity, _ = self.milvus.search_face(feature)
                predicted = is_match and similarity >= threshold
                y_pred.append(1 if predicted else 0)
            
            # 计算指标
            TP = sum(1 for p, t in zip(y_pred, all_labels) if p == 1 and t == 1)
            FP = sum(1 for p, t in zip(y_pred, all_labels) if p == 1 and t == 0)
            TN = sum(1 for p, t in zip(y_pred, all_labels) if p == 0 and t == 0)
            FN = sum(1 for p, t in zip(y_pred, all_labels) if p == 0 and t == 1)
            
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
            fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
            
            results[threshold] = {
                "accuracy": accuracy * 100,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1_score": f1 * 100,
                "fpr": fpr * 100,
                "fnr": fnr * 100
            }
        
        # 找到最佳F1分数对应的阈值
        best_threshold = max(results.keys(), key=lambda k: results[k]["f1_score"])
        best_metrics = results[best_threshold]
        
        logger.info(f"最优阈值: {best_threshold:.3f} (F1={best_metrics['f1_score']:.2f}%)")
        
        # 可视化阈值搜索
        self._plot_threshold_search_results(results, plots_dir=os.path.join(self.save_dir, "plots"))
        
        return results, best_threshold, best_metrics
    
    def _plot_threshold_search_results(self, threshold_results: Dict[float, Dict[str, float]], 
                                     plots_dir: str):
        """绘制阈值搜索结果图"""
        thresholds = sorted(threshold_results.keys())
        metrics_data = {metric: [threshold_results[t][metric] for t in thresholds] 
                       for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'fpr', 'fnr']}
        
        plot_path = os.path.join(plots_dir, "threshold_analysis.png")
        self.visualizer.plot_threshold_analysis(thresholds, metrics_data, save_path=plot_path)
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "=" * 70)
        print("黑名单系统性能评估报告")
        print("=" * 70)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"决策阈值: {metrics.get('threshold', THRESHOLD)}")
        print("-" * 70)
        print(f"测试样本总数: {metrics['total_samples']:,}")
        print(f"正样本数 (应在黑名单): {metrics['positive_samples']:,}")
        print(f"负样本数 (不应在黑名单): {metrics['negative_samples']:,}")
        print(f"成功检测人脸: {metrics['detected_faces']:,}")
        print("-" * 70)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.2f}%")
        print(f"精确率 (Precision): {metrics['precision']:.2f}%")
        print(f"召回率 (Recall): {metrics['recall']:.2f}%")
        print(f"F1分数: {metrics['f1_score']:.2f}%")
        print("-" * 70)
        print(f"误报率 (FPR): {metrics['false_positive_rate']:.2f}%")
        print(f"漏报率 (FNR): {metrics['false_negative_rate']:.2f}%")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        if 'pr_auc' in metrics:
            print(f"PR-AUC: {metrics['pr_auc']:.3f}")
        print("-" * 70)
        print(f"平均识别时间: {metrics['avg_time_per_image_ms']:.2f}ms/张")
        print("=" * 70)
        print("\n关键指标说明:")
        print("- 误报率 (FPR): 不在黑名单的人被错误识别为在黑名单的比例")
        print("- 漏报率 (FNR): 在黑名单的人未被识别的比例")
        print("- 召回率: 在黑名单的人被正确识别的比例")
        print("- 精确率: 被识别为黑名单的人中真正在黑名单的比例")
        print("=" * 70)
    
    def analyze_errors(self, results_path: Optional[str] = None):
        """
        分析错误案例
        
        Args:
            results_path: 结果文件路径，如果为None则使用最新的结果
        """
        if results_path is None:
            results_path = os.path.join(self.save_dir, "reports", "evaluation_results.json")
        
        if not os.path.exists(results_path):
            logger.error("结果文件不存在，无法分析错误")
            return
        
        with open(results_path, "r", encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n" + "=" * 70)
        print("错误案例分析")
        print("=" * 70)
        
        # 假阳性（误报）
        false_positives = [s for s in results["negative_samples"] if not s["correct"]]
        print(f"\n假阳性案例 (误报): {len(false_positives)} 个")
        print("-" * 70)
        for i, fp in enumerate(false_positives[:10]):  # 显示前10个
            print(f"{i + 1:2d}. {fp['person_name']}")
            print(f"    图片: {fp['image_path']}")
            print(f"    匹配到: {fp.get('matched_person', 'Unknown')}")
            print(f"    相似度: {fp['similarity']:.4f}")
        
        # 假阴性（漏报）
        false_negatives = [s for s in results["positive_samples"] if not s["correct"]]
        print(f"\n假阴性案例 (漏报): {len(false_negatives)} 个")
        print("-" * 70)
        for i, fn in enumerate(false_negatives[:10]):  # 显示前10个
            print(f"{i + 1:2d}. {fn['person_name']}")
            print(f"    图片: {fn['image_path']}")
            print(f"    匹配到: {fn.get('matched_person', 'None')}")
            print(f"    相似度: {fn['similarity']:.4f}")
        
        # 未检测到人脸的案例
        no_face_samples = [s for s in results["positive_samples"] + results["negative_samples"]
                          if not s.get("detected", True)]
        print(f"\n未检测到人脸: {len(no_face_samples)} 张")
        if no_face_samples:
            print("-" * 70)
            for i, sample in enumerate(no_face_samples[:5]):
                print(f"{i + 1}. {sample['person_name']}: {sample['image_path']}")
        
        # 统计错误原因
        print("\n" + "=" * 70)
        print("错误统计摘要")
        print("=" * 70)
        print(f"总错误数: {len(false_positives) + len(false_negatives)}")
        print(f"误报率: {len(false_positives) / len(results['negative_samples']) * 100:.2f}%")
        print(f"漏报率: {len(false_negatives) / len(results['positive_samples']) * 100:.2f}%")
        print(f"检测失败率: {len(no_face_samples) / (len(results['positive_samples']) + len(results['negative_samples'])) * 100:.2f}%")
        print("=" * 70)
        
        return false_positives, false_negatives, no_face_samples
    
    def generate_html_report(self, results_path: Optional[str] = None, 
                           output_path: Optional[str] = None):
        """
        生成HTML格式报告
        
        Args:
            results_path: 结果文件路径
            output_path: 输出HTML路径
        """
        if results_path is None:
            results_path = os.path.join(self.save_dir, "reports", "evaluation_results.json")
        
        if not os.path.exists(results_path):
            logger.error("结果文件不存在，无法生成报告")
            return
        
        with open(results_path, "r", encoding='utf-8') as f:
            results = json.load(f)
        
        if output_path is None:
            output_path = os.path.join(self.save_dir, "reports", "evaluation_report.html")
        
        # 生成HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>黑名单系统评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
                .plot-item {{ text-align: center; }}
                .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .error-list {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 黑名单系统性能评估报告</h1>
                <p><strong>评估时间:</strong> {results['timestamp']}</p>
                <p><strong>决策阈值:</strong> {results['threshold']}</p>
                
                <h2>📊 核心指标</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['accuracy']:.2f}%</div>
                        <div class="metric-label">准确率 (Accuracy)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['precision']:.2f}%</div>
                        <div class="metric-label">精确率 (Precision)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['recall']:.2f}%</div>
                        <div class="metric-label">召回率 (Recall)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['f1_score']:.2f}%</div>
                        <div class="metric-label">F1分数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['false_positive_rate']:.2f}%</div>
                        <div class="metric-label">误报率 (FPR)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['metrics']['false_negative_rate']:.2f}%</div>
                        <div class="metric-label">漏报率 (FNR)</div>
                    </div>
                </div>
                
                <div class="summary">
                    <h3>📈 统计摘要</h3>
                    <p><strong>总样本数:</strong> {results['metrics']['total_samples']:,}</p>
                    <p><strong>正样本数 (应在黑名单):</strong> {results['metrics']['positive_samples']:,}</p>
                    <p><strong>负样本数 (不应在黑名单):</strong> {results['metrics']['negative_samples']:,}</p>
                    <p><strong>平均识别时间:</strong> {results['metrics']['avg_time_per_image_ms']:.2f}ms/张</p>
                    {'<p><strong>ROC-AUC:</strong> ' + str(round(results['metrics'].get('roc_auc', 0), 3)) + '</p>' if 'roc_auc' in results['metrics'] else ''}
                    {'<p><strong>PR-AUC:</strong> ' + str(round(results['metrics'].get('pr_auc', 0), 3)) + '</p>' if 'pr_auc' in results['metrics'] else ''}
                </div>
                
                <h2>📉 可视化图表</h2>
                <div class="plot-grid">
                    <div class="plot-item">
                        <h3>混淆矩阵</h3>
                        <img src="../plots/confusion_matrix.png" alt="混淆矩阵">
                    </div>
                    <div class="plot-item">
                        <h3>ROC曲线</h3>
                        <img src="../plots/roc_curve.png" alt="ROC曲线">
                    </div>
                    <div class="plot-item">
                        <h3>PR曲线</h3>
                        <img src="../plots/pr_curve.png" alt="PR曲线">
                    </div>
                    <div class="plot-item">
                        <h3>相似度分布</h3>
                        <img src="../plots/similarity_distribution.png" alt="相似度分布">
                    </div>
                </div>
                
                <h2>⚠️ 错误分析</h2>
                <div class="error-list">
                    <h3>假阳性 (误报)</h3>
                    <p>数量: {sum(1 for s in results['negative_samples'] if not s['correct'])}/{len(results['negative_samples'])}</p>
                </div>
                <div class="error-list">
                    <h3>假阴性 (漏报)</h3>
                    <p>数量: {sum(1 for s in results['positive_samples'] if not s['correct'])}/{len(results['positive_samples'])}</p>
                </div>
                
                <hr>
                <p style="text-align: center; color: #666; font-size: 12px;">
                    报告由黑名单评估系统生成
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {output_path}")


if __name__ == "__main__":
    # 使用示例
    evaluator = BlacklistEvaluator(save_dir="evaluation_results")
    
    # 1. 基本评估
    results = evaluator.evaluate_test_set("test_set_info.json", generate_plots=True)
    
    # 2. 错误分析
    evaluator.analyze_errors()
    
    # 3. 搜索最优阈值
    print("\n" + "="*70)
    print("开始搜索最优阈值...")
    threshold_results, best_threshold, best_metrics = evaluator.search_optimal_threshold(
        threshold_range=(0.3, 0.7), step=0.01
    )
    
    # 4. 生成HTML报告
    evaluator.generate_html_report()
    
    print("\n" + "="*70)
    print("✅ 评估完成！所有结果已保存至 evaluation_results 目录")
    print("📂 目录结构:")
    print("   ├── plots/          # 可视化图表")
    print("   ├── reports/        # 评估报告")
    print("   └── evaluation_report.html  # 交互式HTML报告")
    print("="*70)
