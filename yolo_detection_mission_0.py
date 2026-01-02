import os
import yaml
from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import traceback
import torch

warnings.filterwarnings('ignore')


class YOLOModelEvaluator:
    def __init__(self, data_yaml_path, project_name="model_evaluation"):
        self.data_yaml_path = data_yaml_path
        self.project_name = project_name

        # 读取数据集配置
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        # 获取数据集所有类别
        self.dataset_classes = self.data_config.get('names', [])
        self.dataset_nc = self.data_config.get('nc', 0)
        print(self.dataset_nc)
        print(self.dataset_classes)

        self.model_versions = [
            'yolov8n.pt',
            'yolov8s.pt',
            'yolov8m.pt',
            'yolov8l.pt',
            'yolov8x.pt',

            'yolov11n.pt',
            'yolov11s.pt',
            'yolov11m.pt',
            'yolov11l.pt',
            'yolov11x.pt',
        ]

        self.model_categories = {
            'yolov8n.pt': 'YOLOv8-Nano',
            'yolov8s.pt': 'YOLOv8-Small',
            'yolov8m.pt': 'YOLOv8-Medium',
            'yolov8l.pt': 'YOLOv8-Large',
            'yolov8x.pt': 'YOLOv8-XLarge',
            'yolov11n.pt': 'YOLOv11-Nano',
            'yolov11s.pt': 'YOLOv11-Small',
            'yolov11m.pt': 'YOLOv11-Medium',
            'yolov11l.pt': 'YOLOv11-Large',
            'yolov11x.pt': 'YOLOv11-XLarge',
        }

        # 定义各预训练模型支持的类别（基于COCO 80类）
        self.model_supported_classes = self._get_coco_classes()

        # 创建结果保存目录
        self.results_dir = Path(f"results/{project_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 保存过滤信息
        self.filter_info = {}

    def _get_coco_classes(self):
        """获取COCO数据集的80个类别（YOLO预训练模型的默认类别）"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return coco_classes

    def get_model_classes_indices(self, model_name, dataset_classes):
        # 获取模型支持的类别名称
        model_classes = self.model_supported_classes

        # 找出数据集类别中哪些是模型支持的
        supported_indices = []
        for class_name in model_classes:
            if class_name in dataset_classes:
                idx = dataset_classes.index(class_name)
                supported_indices.append(idx)

        return supported_indices

    def evaluate_model(self, model_name, device='cuda'):
        print(f"\n{'=' * 60}")
        print(f"正在评估模型: {model_name}")
        print(f"{'=' * 60}")

        try:
            model = YOLO(model_name)

            # 获取模型支持的类别索引
            supported_indices = self.get_model_classes_indices(model_name, self.dataset_classes)

            # 记录过滤信息
            self.filter_info[model_name] = {
                'total_classes': self.dataset_nc,
                'supported_classes': len(supported_indices),
                'supported_indices': supported_indices,
                'unsupported_count': self.dataset_nc - len(supported_indices)
            }

            print(f"数据集总类别数: {self.dataset_nc}")
            print(f"模型支持的类别数: {len(supported_indices)}")
            print(f"过滤掉的未知类别数: {self.dataset_nc - len(supported_indices)}")

            if len(supported_indices) == 0:
                print("警告: 模型不支持数据集中的任何已知类别!")
                print(f"模型支持的类别: {self.model_supported_classes}")
                print(f"数据集类别: {self.dataset_classes}")

            # 显示支持的类别
            supported_names = [self.dataset_classes[i] for i in supported_indices]
            print(f"模型支持的类别: {supported_names}")

            # 执行验证 - 关键修改：使用classes参数
            results = model.val(
                data=self.data_yaml_path,
                imgsz=640,
                batch=16,
                save_json=True,
                save_hybrid=False,
                conf=0.001,
                iou=0.6,
                max_det=300,
                half=True,
                device=device,
                verbose=False,
                classes=supported_indices if supported_indices else None  # 关键：只评估这些类别
            )

            # 提取关键指标
            metrics = {
                'model': model_name,
                'model_category': self.model_categories.get(model_name, 'Unknown'),
                'model_family': 'YOLOv8' if 'yolov8' in model_name else 'YOLOv11',
                'model_size': model_name.split('v')[1][0] if len(model_name.split('v')) > 1 else 'Unknown',
                'total_dataset_classes': self.dataset_nc,
                'evaluated_classes': len(supported_indices),
                'filtered_classes': self.dataset_nc - len(supported_indices),
            }

            # 提取数值指标
            try:
                # mAP
                if hasattr(results.box, 'map'):
                    map_val = results.box.map
                    if isinstance(map_val, (list, np.ndarray)):
                        metrics['mAP50-95'] = float(map_val[0]) if len(map_val) > 0 else 0.0
                    else:
                        metrics['mAP50-95'] = float(map_val)
                else:
                    metrics['mAP50-95'] = 0.0

                if hasattr(results.box, 'map50'):
                    map50_val = results.box.map50
                    if isinstance(map50_val, (list, np.ndarray)):
                        metrics['mAP50'] = float(map50_val[0]) if len(map50_val) > 0 else 0.0
                    else:
                        metrics['mAP50'] = float(map50_val)
                else:
                    metrics['mAP50'] = 0.0

                if hasattr(results.box, 'map75'):
                    map75_val = results.box.map75
                    if isinstance(map75_val, (list, np.ndarray)):
                        metrics['mAP75'] = float(map75_val[0]) if len(map75_val) > 0 else 0.0
                    else:
                        metrics['mAP75'] = float(map75_val)
                else:
                    metrics['mAP75'] = 0.0

                # Precision, Recall
                if hasattr(results.box, 'p'):
                    p_val = results.box.p
                    if isinstance(p_val, (list, np.ndarray)):
                        metrics['precision'] = float(np.mean(p_val)) if len(p_val) > 0 else 0.0
                    else:
                        metrics['precision'] = float(p_val)
                else:
                    metrics['precision'] = 0.0

                if hasattr(results.box, 'r'):
                    r_val = results.box.r
                    if isinstance(r_val, (list, np.ndarray)):
                        metrics['recall'] = float(np.mean(r_val)) if len(r_val) > 0 else 0.0
                    else:
                        metrics['recall'] = float(r_val)
                else:
                    metrics['recall'] = 0.0

                # 计算F1分数
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (
                            metrics['precision'] + metrics['recall'])
                else:
                    metrics['f1_score'] = 0.0

                # 推理时间
                if hasattr(results, 'speed') and 'inference' in results.speed:
                    metrics['inference_time_ms'] = float(results.speed['inference'])
                else:
                    metrics['inference_time_ms'] = 0.0

                if hasattr(results, 'speed') and 'preprocess' in results.speed:
                    metrics['preprocess_time_ms'] = float(results.speed['preprocess'])
                else:
                    metrics['preprocess_time_ms'] = 0.0

                if hasattr(results, 'speed') and 'postprocess' in results.speed:
                    metrics['postprocess_time_ms'] = float(results.speed['postprocess'])
                else:
                    metrics['postprocess_time_ms'] = 0.0

            except Exception as e:
                print(f"提取指标时出错: {str(e)}")
                metrics.update({
                    'mAP50-95': 0.0,
                    'mAP50': 0.0,
                    'mAP75': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'inference_time_ms': 0.0,
                    'preprocess_time_ms': 0.0,
                    'postprocess_time_ms': 0.0,
                })

            # 获取各类别的AP值（只获取支持的类别）
            if hasattr(results.box, 'ap') and hasattr(results.box, 'ap_class_index'):
                # 获取实际评估的类别索引
                evaluated_indices = results.box.ap_class_index.tolist() if hasattr(results.box.ap_class_index,
                                                                                   'tolist') else list(
                    results.box.ap_class_index)

                for i, class_idx in enumerate(evaluated_indices):
                    if class_idx < len(self.dataset_classes):
                        class_name = self.dataset_classes[class_idx]
                        ap_val = results.box.ap[i]
                        if isinstance(ap_val, (list, np.ndarray)):
                            metrics[f'AP_{class_name}'] = float(np.mean(ap_val)) if len(ap_val) > 0 else 0.0
                        else:
                            metrics[f'AP_{class_name}'] = float(ap_val)

            # 为未评估的类别设置AP为0
            for i, class_name in enumerate(self.dataset_classes):
                if f'AP_{class_name}' not in metrics:
                    metrics[f'AP_{class_name}'] = 0.0

            print(f"✓ {model_name} 评估完成")
            print(f"  评估类别数: {len(supported_indices)}/{self.dataset_nc}")
            print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"  mAP50: {metrics['mAP50']:.4f}")
            print(f"  推理时间: {metrics['inference_time_ms']:.2f} ms/img")

            return metrics

        except Exception as e:
            print(f"✗ 评估模型 {model_name} 时出错: {str(e)}")
            traceback.print_exc()
            return None

    def evaluate_all_models(self, skip_models=None):
        """评估所有YOLO模型"""
        all_results = []

        print("开始评估YOLO系列模型...")
        print(f"数据集配置: {self.data_yaml_path}")
        print(f"数据集总类别数: {self.dataset_nc}")
        print(f"数据集类别: {self.dataset_classes}")
        print(f"结果将保存到: {self.results_dir}")

        # 过滤要跳过的模型
        if skip_models is None:
            skip_models = []

        models_to_evaluate = [m for m in self.model_versions if m not in skip_models]
        print(f"将评估以下模型: {', '.join([m.replace('.pt', '') for m in models_to_evaluate])}")

        # 逐个评估模型
        for model_version in tqdm(models_to_evaluate, desc="评估进度"):
            result = self.evaluate_model(model_version)
            if result:
                all_results.append(result)

        # 保存结果
        if all_results:
            self.save_results(all_results)
        else:
            print("Error.")

        return all_results

    def save_results(self, results):
        """保存评估结果"""
        try:
            # 转换为DataFrame
            df = pd.DataFrame(results)

            # 保存为CSV
            csv_path = self.results_dir / "model_evaluation_results.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"\n详细结果已保存到: {csv_path}")

            # 保存为Excel
            excel_path = self.results_dir / "model_evaluation_results.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='评估结果', index=False)

                # 添加过滤信息表
                filter_df = pd.DataFrame([
                    {
                        'model': model,
                        'total_classes': info['total_classes'],
                        'evaluated_classes': info['supported_classes'],
                        'filtered_classes': info['unsupported_count'],
                        'supported_indices': str(info['supported_indices'])
                    }
                    for model, info in self.filter_info.items()
                ])
                filter_df.to_excel(writer, sheet_name='过滤信息', index=False)

                # 添加性能汇总表
                summary_cols = ['model', 'model_category', 'model_family', 'model_size',
                                'total_dataset_classes', 'evaluated_classes', 'filtered_classes',
                                'mAP50-95', 'mAP50', 'f1_score', 'inference_time_ms']
                summary_df = df[[c for c in summary_cols if c in df.columns]].copy()
                summary_df.to_excel(writer, sheet_name='性能汇总', index=False)

            print(f"Excel格式结果已保存到: {excel_path}")

            # 生成可视化报告
            self.create_visualizations(df)

        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
            traceback.print_exc()

    def create_visualizations(self, df):
        """创建可视化图表"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('YOLO系列模型性能对比 (过滤未知类别)', fontsize=16, fontweight='bold')

            # 1. mAP对比图
            ax1 = axes[0, 0]
            models = [str(m).replace('.pt', '') for m in df['model']]
            x = np.arange(len(models))
            width = 0.25

            map50_95 = df['mAP50-95'].fillna(0).values
            map50 = df['mAP50'].fillna(0).values

            ax1.bar(x - width / 2, map50_95, width, label='mAP50-95', color='skyblue')
            ax1.bar(x + width / 2, map50, width, label='mAP50', color='lightcoral')

            ax1.set_xlabel('模型版本')
            ax1.set_ylabel('mAP分数')
            ax1.set_title('不同模型的mAP对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 评估类别数对比
            ax2 = axes[0, 1]
            evaluated_classes = df['evaluated_classes'].fillna(0).values
            total_classes = df['total_dataset_classes'].fillna(0).values

            bars1 = ax2.bar(x, total_classes, width, label='总类别数', color='lightgray')
            bars2 = ax2.bar(x, evaluated_classes, width, label='评估类别数', color='lightgreen')

            ax2.set_xlabel('模型版本')
            ax2.set_ylabel('类别数量')
            ax2.set_title('评估类别数对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 推理时间对比
            ax3 = axes[0, 2]
            inference_times = df['inference_time_ms'].fillna(0).values
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax3.bar(models, inference_times, color=colors)

            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.1f}', ha='center', va='bottom')

            ax3.set_xlabel('模型版本')
            ax3.set_ylabel('推理时间 (ms/img)')
            ax3.set_title('推理速度对比')
            ax3.set_xticklabels(models, rotation=45)
            ax3.grid(True, alpha=0.3)

            # 4. 各类别AP热力图（只显示支持的类别）
            ax4 = axes[1, 0]
            class_cols = [col for col in df.columns if col.startswith('AP_')]

            if class_cols and len(class_cols) > 0:
                class_data = df[['model'] + class_cols].copy()
                class_data['model'] = [str(m).replace('.pt', '') for m in class_data['model']]

                for col in class_cols:
                    class_data[col] = pd.to_numeric(class_data[col], errors='coerce')

                class_data.set_index('model', inplace=True)

                if not class_data.empty:
                    im = ax4.imshow(class_data.values, cmap='YlOrRd', aspect='auto')

                    ax4.set_xlabel('车辆类别')
                    ax4.set_ylabel('模型版本')
                    ax4.set_title('各类别AP值热力图')
                    ax4.set_xticks(range(len(class_cols)))
                    ax4.set_xticklabels([col.replace('AP_', '') for col in class_cols], rotation=45)
                    ax4.set_yticks(range(len(models)))
                    ax4.set_yticklabels([str(m).replace('.pt', '') for m in models])

                    plt.colorbar(im, ax=ax4)
                else:
                    ax4.text(0.5, 0.5, '无类别AP数据', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, '无类别AP数据', ha='center', va='center', transform=ax4.transAxes)

            # 5. 模型系列对比
            ax5 = axes[1, 1]
            v8_models = df[df['model_family'] == 'YOLOv8']
            v11_models = df[df['model_family'] == 'YOLOv11']

            if len(v8_models) > 0 and len(v11_models) > 0:
                v8_sizes = ['n', 's', 'm', 'l', 'x'][:len(v8_models)]
                v11_sizes = ['n', 's', 'm', 'l', 'x'][:len(v11_models)]

                ax5.plot(v8_sizes, v8_models['mAP50-95'].values, 'o-', label='YOLOv8', linewidth=2, markersize=8)
                ax5.plot(v11_sizes, v11_models['mAP50-95'].values, 's-', label='YOLOv11', linewidth=2, markersize=8)

                ax5.set_xlabel('模型尺寸')
                ax5.set_ylabel('mAP50-95')
                ax5.set_title('YOLOv8 vs YOLOv11 性能对比')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, '缺少模型系列数据', ha='center', va='center', transform=ax5.transAxes)

            # 6. 性能汇总表格
            ax6 = axes[1, 2]
            ax6.axis('off')

            if 'model_category' in df.columns:
                performance_data = df[['model_category', 'evaluated_classes', 'mAP50-95',
                                       'mAP50', 'f1_score', 'inference_time_ms']].copy()
            else:
                performance_data = df[['model', 'evaluated_classes', 'mAP50-95',
                                       'mAP50', 'f1_score', 'inference_time_ms']].copy()
                performance_data['model'] = [str(m).replace('.pt', '') for m in performance_data['model']]

            text_content = "模型性能综合对比\n\n"
            text_content += f"{'模型':<15} {'评估类别':<8} {'mAP50-95':<10} {'mAP50':<10} {'F1':<10} {'速度':<10}\n"
            text_content += "-" * 65 + "\n"

            for idx, row in performance_data.iterrows():
                model_name = row['model_category'] if 'model_category' in row else row['model']
                text_content += f"{model_name:<15} {row['evaluated_classes']:<8} "
                text_content += f"{row['mAP50-95']:.4f}     {row['mAP50']:.4f}     "
                text_content += f"{row['f1_score']:.4f}     {row['inference_time_ms']:.1f}ms\n"

            ax6.text(0.05, 0.5, text_content, fontfamily='monospace', fontsize=8,
                     verticalalignment='center', transform=ax6.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            plot_path = self.results_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"可视化图表已保存到: {plot_path}")

            # 生成文本报告
            self.generate_text_report(df)

        except Exception as e:
            print(f"创建可视化图表时出错: {str(e)}")
            traceback.print_exc()

    def generate_text_report(self, df):
        """生成文本格式的性能报告"""
        try:
            report_path = self.results_dir / "performance_report.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("YOLO车辆检测模型性能评估报告 (过滤未知类别)\n")
                f.write("=" * 80 + "\n\n")

                f.write("1. 数据集信息\n")
                f.write("-" * 50 + "\n")
                f.write(f"配置文件: {self.data_yaml_path}\n")
                f.write(f"总类别数: {self.dataset_nc}\n")
                f.write(f"类别列表: {', '.join(self.dataset_classes)}\n\n")

                f.write("2. 评估设置\n")
                f.write("-" * 50 + "\n")
                f.write("评估模式: 只评估预训练模型已知的类别\n")
                f.write("模型支持的类别: COCO 80类\n")
                f.write(
                    f"匹配到的类别: {len([c for c in self.dataset_classes if c in self.model_supported_classes])}\n\n")

                f.write("3. 模型性能总排名 (按mAP50-95)\n")
                f.write("-" * 50 + "\n")

                if 'mAP50-95' in df.columns:
                    ranked_df = df.sort_values('mAP50-95', ascending=False)
                    for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
                        model_name = str(row['model']).replace('.pt', '')
                        f.write(f"{i:2d}. {model_name:<12} mAP50-95: {row['mAP50-95']:.4f}  "
                                f"评估类别: {row['evaluated_classes']}/{row['total_dataset_classes']}  "
                                f"速度: {row['inference_time_ms']:.1f}ms/img\n")

                f.write("\n4. 最佳模型推荐\n")
                f.write("-" * 50 + "\n")

                if len(df) > 0:
                    best_map_idx = df['mAP50-95'].idxmax()
                    best_map_model = df.loc[best_map_idx]

                    best_speed_idx = df['inference_time_ms'].idxmin()
                    best_speed_model = df.loc[best_speed_idx]

                    best_f1_idx = df['f1_score'].idxmax()
                    best_f1_model = df.loc[best_f1_idx]

                    f.write(f"最佳精度模型: {best_map_model['model']}\n")
                    f.write(f"  - mAP50-95: {best_map_model['mAP50-95']:.4f}\n")
                    f.write(
                        f"  - 评估类别: {best_map_model['evaluated_classes']}/{best_map_model['total_dataset_classes']}\n")
                    f.write(f"  - 推理速度: {best_map_model['inference_time_ms']:.1f} ms/img\n\n")

                    f.write(f"最快速度模型: {best_speed_model['model']}\n")
                    f.write(f"  - 推理速度: {best_speed_model['inference_time_ms']:.1f} ms/img\n")
                    f.write(f"  - mAP50-95: {best_speed_model['mAP50-95']:.4f}\n\n")

                    f.write(f"最佳平衡模型: {best_f1_model['model']}\n")
                    f.write(f"  - F1分数: {best_f1_model['f1_score']:.4f}\n")
                    f.write(f"  - mAP50-95: {best_f1_model['mAP50-95']:.4f}\n\n")

                f.write("5. 注意事项\n")
                f.write("-" * 50 + "\n")
                f.write("• 评估时只考虑了模型预训练过的类别\n")
                f.write("• 数据集中的未知类别已被自动过滤\n")
                f.write("• 如果要检测所有类别，需要对模型进行微调训练\n")
                f.write("• COCO预训练模型支持80个类别，详见代码中的model_supported_classes\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("报告生成完成\n")
                f.write("=" * 80 + "\n")

            print(f"详细性能报告已保存到: {report_path}")

        except Exception as e:
            print(f"生成文本报告时出错: {str(e)}")
            traceback.print_exc()


def main():
    """主函数"""
    # 设置路径
    data_yaml_path = "VehiclesDetectionDataset/dataset.yaml"

    # 检查文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"错误: 找不到YAML文件: {data_yaml_path}")
        print("请确保路径正确，或者使用绝对路径")
        return

    # 创建评估器
    print("初始化YOLO模型评估器...")
    evaluator = YOLOModelEvaluator(data_yaml_path, project_name="vehicle_detection_eval_filtered")

    # 执行评估
    print("\n开始评估所有YOLO预训练模型...")
    print("注意: 将只评估预训练模型已知的类别，未知类别会被自动过滤")

    results = evaluator.evaluate_all_models()

    if results:
        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)
        print(f"已评估模型数量: {len(results)}")
        print(f"结果保存目录: {evaluator.results_dir}")

        # 显示过滤信息
        df = pd.DataFrame(results)
        if len(df) > 0:
            print(f"\n过滤统计:")
            for model_name, info in evaluator.filter_info.items():
                print(f"  {model_name}: {info['supported_classes']}/{info['total_classes']} 类别被评估")

            # 显示最佳模型
            best_model = df.loc[df['mAP50-95'].idxmax()]
            fastest_model = df.loc[df['inference_time_ms'].idxmin()]

            print(f"\n最佳精度模型: {best_model['model']}")
            print(f"  - mAP50-95: {best_model['mAP50-95']:.4f}")
            print(f"  - 评估类别: {best_model['evaluated_classes']}/{best_model['total_dataset_classes']}")
            print(f"\n最快推理模型: {fastest_model['model']}")
            print(f"  - 推理速度: {fastest_model['inference_time_ms']:.1f} ms/img")
    else:
        print("评估过程中出现错误，请检查日志")


if __name__ == "__main__":
    # 检查ultralytics是否安装
    try:
        import torch
        from ultralytics import YOLO

        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print("缺少必要的依赖库，请安装：")
        print("pip install ultralytics torch torchvision pandas matplotlib seaborn openpyxl tqdm pyyaml")
        exit(1)

    main()