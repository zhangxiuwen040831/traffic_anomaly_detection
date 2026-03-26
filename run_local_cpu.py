
#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ {description} 完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败！错误码: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='运行完整的本地 CPU 验证流程')
    parser.add_argument('--skip-env-check', action='store_true', help='跳过环境检查')
    parser.add_argument('--skip-download', action='store_true', help='跳过数据下载')
    args = parser.parse_args()
    
    print("="*60)
    print("  网络流量异常检测系统 - 本地 CPU 完整验证流程")
    print("="*60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    steps = []
    
    if not args.skip_env_check:
        steps.append(("环境检查", "python environment_check.py"))
    
    if not args.skip_download:
        steps.append(("数据集下载", "python scripts/download_dataset.py"))
    
    steps.extend([
        ("数据预处理", "python scripts/preprocess.py"),
        ("模型训练", "python scripts/train_autoencoder.py"),
        ("模型评估", "python scripts/evaluate.py")
    ])
    
    for description, cmd in steps:
        success = run_command(cmd, description)
        if not success:
            if "数据集下载" in description:
                print("\n使用合成数据集模式...")
                success = run_command("python scripts/download_dataset.py --synthetic", "生成合成数据集")
                if not success:
                    print("\n" + "="*60)
                    print("  流程执行失败！")
                    print("="*60)
                    return 1
            else:
                print("\n" + "="*60)
                print("  流程执行失败！")
                print("="*60)
                return 1
    
    print("\n" + "="*60)
    print("  本地 CPU 最小化验证已完成，可进入云 GPU 第二阶段")
    print("="*60)
    print("\n生成的文件:")
    print("  - 环境检查报告: outputs/logs/environment_check_report.txt")
    print("  - 训练模型: outputs/checkpoints/autoencoder.pth")
    print("  - 评估报告: outputs/reports/evaluation_report.md")
    print("  - 可视化图表: outputs/figures/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
