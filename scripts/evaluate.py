
#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config
from src.models.autoencoder import MLPAutoencoder

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_test_data(processed_data_dir):
    print("正在加载测试数据...")
    X_test = np.load(os.path.join(processed_data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_data_dir, 'y_test.npy'))
    
    print(f"测试集大小: {len(X_test)}")
    print(f"正常样本: {np.sum(y_test == 0)}")
    print(f"异常样本: {np.sum(y_test == 1)}")
    
    return X_test, y_test

def load_model(checkpoints_dir, input_dim, config):
    print("正在加载模型...")
    hidden_dims = config['model']['hidden_dims']
    activation = config['model']['activation']
    dropout = config['model']['dropout']
    
    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout
    )
    
    model_path = os.path.join(checkpoints_dir, 'autoencoder.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

def compute_anomaly_scores(model, X_test, device):
    print("正在计算异常分数...")
    model.to(device)
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        recon = model(X_tensor)
        mse = torch.mean((recon - X_tensor) ** 2, dim=1)
    
    anomaly_scores = mse.cpu().numpy()
    return anomaly_scores

def evaluate_metrics(y_true, anomaly_scores, threshold_percentile=95):
    print("\n计算评估指标...")
    
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    
    precision, recall, thresholds_pr = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    y_pred = (anomaly_scores > threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n评估结果:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision_val:.4f}")
    print(f"  Recall: {recall_val:.4f}")
    print(f"  阈值 ({threshold_percentile}%分位数): {threshold:.6f}")
    print(f"  混淆矩阵:")
    print(f"    TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"    FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'precision': precision_val,
        'recall': recall_val,
        'threshold': threshold,
        'confusion_matrix': cm,
        'anomaly_scores': anomaly_scores,
        'y_true': y_true
    }

def plot_results(metrics, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    
    y_true = metrics['y_true']
    anomaly_scores = metrics['anomaly_scores']
    
    normal_scores = anomaly_scores[y_true == 0]
    abnormal_scores = anomaly_scores[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.5, label='正常', density=True)
    plt.hist(abnormal_scores, bins=50, alpha=0.5, label='异常', density=True)
    plt.xlabel('重构误差 (异常分数)')
    plt.ylabel('密度')
    plt.title('正常与异常样本的异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'anomaly_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 异常分数分布图已保存")
    
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ ROC 曲线已保存")
    
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR-AUC = {metrics["pr_auc"]:.4f}')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('PR 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ PR 曲线已保存")
    
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['预测正常', '预测异常'],
                yticklabels=['实际正常', '实际异常'])
    plt.title('混淆矩阵')
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 混淆矩阵图已保存")

def generate_report(metrics, config, reports_dir):
    os.makedirs(reports_dir, exist_ok=True)
    
    report = []
    report.append("="*60)
    report.append("网络流量异常检测系统 - 评估报告")
    report.append("="*60)
    report.append("")
    report.append("1. 模型配置:")
    report.append(f"   模型类型: {config['model']['type']}")
    report.append(f"   隐藏层维度: {config['model']['hidden_dims']}")
    report.append(f"   激活函数: {config['model']['activation']}")
    report.append(f"   Dropout: {config['model']['dropout']}")
    report.append("")
    report.append("2. 训练配置:")
    report.append(f"   Epochs: {config['training']['epochs']}")
    report.append(f"   Batch Size: {config['training']['batch_size']}")
    report.append(f"   学习率: {config['training']['learning_rate']}")
    report.append("")
    report.append("3. 评估结果:")
    report.append(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    report.append(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    report.append(f"   F1 Score: {metrics['f1']:.4f}")
    report.append(f"   Precision: {metrics['precision']:.4f}")
    report.append(f"   Recall: {metrics['recall']:.4f}")
    report.append("")
    report.append("4. 混淆矩阵:")
    cm = metrics['confusion_matrix']
    report.append(f"   TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    report.append(f"   FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    report.append("")
    report.append("5. 结论:")
    if metrics['roc_auc'] > 0.7:
        report.append("   ✓ 模型表现良好，异常检测能力明显优于随机水平")
    elif metrics['roc_auc'] > 0.5:
        report.append("   ⚠️  模型有一定效果，但仍有提升空间")
    else:
        report.append("   ✗ 模型效果不佳，需要进一步调试")
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    print(f"\n{report_text}")
    
    with open(os.path.join(reports_dir, 'evaluation_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n评估报告已保存至: {os.path.join(reports_dir, 'evaluation_report.md')}")
    
    return metrics['roc_auc'] > 0.5

def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='配置文件路径')
    parser.add_argument('--override', type=str, default='config/local_cpu.yaml', help='覆盖配置文件路径')
    args = parser.parse_args()
    
    print("="*60)
    print("  网络流量异常检测系统 - 模型评估")
    print("="*60)
    
    config = load_config(args.config, args.override)
    
    device = get_device()
    
    processed_data_dir = config['paths']['processed_data']
    checkpoints_dir = config['paths']['checkpoints']
    figures_dir = config['paths']['figures']
    reports_dir = config['paths']['reports']
    
    X_test, y_test = load_test_data(processed_data_dir)
    input_dim = X_test.shape[1]
    
    model = load_model(checkpoints_dir, input_dim, config)
    anomaly_scores = compute_anomaly_scores(model, X_test, device)
    
    normal_scores = anomaly_scores[y_test == 0]
    abnormal_scores = anomaly_scores[y_test == 1]
    print(f"\n正常样本平均重构误差: {np.mean(normal_scores):.6f}")
    print(f"异常样本平均重构误差: {np.mean(abnormal_scores):.6f}")
    
    threshold_percentile = config['evaluation']['percentile']
    metrics = evaluate_metrics(y_test, anomaly_scores, threshold_percentile)
    
    plot_results(metrics, figures_dir)
    success = generate_report(metrics, config, reports_dir)
    
    if success:
        print("\n🎉 模型评估完成，效果良好！")
    else:
        print("\n⚠️  模型评估完成，但效果有待提升")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
