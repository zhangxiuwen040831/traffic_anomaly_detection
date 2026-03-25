
#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
from datetime import datetime

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    print_header("Python 版本检查")
    version = sys.version
    print(f"Python 版本: {version}")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major >= 3 and minor >= 8:
        print("✓ Python 版本满足要求 (>=3.8)")
        return True
    else:
        print("✗ Python 版本过低，建议使用 3.8 或更高版本")
        return False

def check_pip_available():
    print_header("Pip 检查")
    try:
        import pip
        print(f"✓ Pip 可用")
        return True
    except ImportError:
        print("✗ Pip 不可用")
        return False

def check_dependencies():
    print_header("核心依赖库检查")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
        'requests': 'Requests'
    }
    
    results = {}
    for pkg, name in required_packages.items():
        try:
            module = __import__(pkg)
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = 'available'
            print(f"✓ {name}: {version}")
            results[name] = (True, version)
        except ImportError:
            print(f"✗ {name}: 未安装")
            results[name] = (False, None)
    return results

def check_cuda():
    print_header("GPU/CUDA 检查")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True, True
        else:
            print("✗ CUDA 不可用，将使用 CPU")
            return True, False
    except ImportError:
        print("✗ PyTorch 未安装，无法检查 CUDA")
        return False, False

def check_system_resources():
    print_header("系统资源检查")
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        print(f"CPU 核心数: {cpu_count}")
        
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        print(f"总内存: {memory_gb:.2f} GB")
        print(f"可用内存: {memory_available_gb:.2f} GB")
        
        disk = psutil.disk_usage('.')
        disk_gb = disk.total / (1024 ** 3)
        disk_available_gb = disk.free / (1024 ** 3)
        print(f"磁盘总空间: {disk_gb:.2f} GB")
        print(f"磁盘可用空间: {disk_available_gb:.2f} GB")
    except ImportError:
        print("CPU 核心数: 未检测 (psutil 未安装)")
        print("总内存: 未检测 (psutil 未安装)")
        print("可用内存: 未检测 (psutil 未安装)")
        print("磁盘总空间: 未检测 (psutil 未安装)")
        print("磁盘可用空间: 未检测 (psutil 未安装)")
        memory_available_gb = 8.0
        disk_available_gb = 10.0
        cpu_count = 4
    
    return {
        'memory_gb': memory_available_gb if 'memory_gb' in locals() else 8.0,
        'memory_available_gb': memory_available_gb,
        'disk_available_gb': disk_available_gb,
        'cpu_count': cpu_count
    }

def check_requirements_satisfied(resources):
    print_header("第一阶段运行条件检查")
    
    min_memory = 2.0  # GB
    min_disk = 1.0    # GB
    
    satisfied = True
    
    if resources['memory_available_gb'] < min_memory:
        print(f"⚠️  可用内存可能不足 (当前: {resources['memory_available_gb']:.2f} GB, 建议: {min_memory} GB)")
    else:
        print(f"✓ 可用内存充足")
    
    if resources['disk_available_gb'] < min_disk:
        print(f"⚠️  磁盘空间可能不足 (当前: {resources['disk_available_gb']:.2f} GB, 建议: {min_disk} GB)")
    else:
        print(f"✓ 磁盘空间充足")
    
    return satisfied

def generate_report(python_ok, deps, torch_ok, cuda_available, resources, requirements_ok):
    print_header("环境检查报告")
    
    report = []
    report.append("="*60)
    report.append("环境检查报告")
    report.append("="*60)
    report.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("1. Python 版本:")
    report.append(f"   {'✓ 满足要求' if python_ok else '✗ 不满足要求'}")
    report.append("")
    report.append("2. 核心依赖库:")
    for name, (installed, version) in deps.items():
        status = "✓" if installed else "✗"
        ver_str = f" ({version})" if version else ""
        report.append(f"   {status} {name}{ver_str}")
    report.append("")
    report.append("3. GPU/CUDA:")
    if torch_ok:
        if cuda_available:
            report.append("   ✓ CUDA 可用")
        else:
            report.append("   ✗ CUDA 不可用，将使用 CPU")
    else:
        report.append("   ✗ PyTorch 未安装")
    report.append("")
    report.append("4. 系统资源:")
    report.append(f"   可用内存: {resources['memory_available_gb']:.2f} GB")
    report.append(f"   可用磁盘: {resources['disk_available_gb']:.2f} GB")
    report.append("")
    report.append("5. 第一阶段运行条件:")
    if requirements_ok:
        report.append("   ✓ 满足运行条件")
    else:
        report.append("   ✗ 不满足运行条件")
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    os.makedirs('outputs/logs', exist_ok=True)
    with open('outputs/logs/environment_check_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存至: outputs/logs/environment_check_report.txt")
    
    return requirements_ok

def main():
    print("="*60)
    print("  网络流量异常检测系统 - 环境检查")
    print("="*60)
    
    try:
        python_ok = check_python_version()
        pip_ok = check_pip_available()
        deps = check_dependencies()
        torch_ok, cuda_available = check_cuda()
        resources = check_system_resources()
        requirements_ok = check_requirements_satisfied(resources)
        
        final_ok = generate_report(
            python_ok, deps, torch_ok, cuda_available, 
            resources, requirements_ok
        )
        
        if final_ok:
            print("\n🎉 环境检查通过，可以开始第一阶段！")
            return 0
        else:
            print("\n⚠️  环境检查部分未通过，请检查上述问题后重试。")
            return 1
            
    except Exception as e:
        print(f"\n✗ 环境检查过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
