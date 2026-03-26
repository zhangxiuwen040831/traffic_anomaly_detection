"""
系统验证脚本 - 检查部署包完整性和功能
"""
from pathlib import Path
import sys

def verify_deploy_bundle():
    """验证deploy_bundle目录结构"""
    print("=" * 60)
    print("验证 deploy_bundle 目录结构")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    deploy_bundle = project_root / "deploy_bundle"
    
    if not deploy_bundle.exists():
        print("❌ deploy_bundle 目录不存在!")
        return False
    
    # 检查必要的目录
    required_dirs = [
        "model",
        "inference",
        "reports",
        "figures"
    ]
    
    all_good = True
    for dir_name in required_dirs:
        dir_path = deploy_bundle / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ 目录存在")
        else:
            print(f"❌ {dir_name}/ 目录不存在!")
            all_good = False
    
    # 检查模型文件
    print("\n检查模型文件:")
    model_dir = deploy_bundle / "model"
    model_files = [
        "best.ckpt",
        "model_config.yaml",
        "threshold_config.yaml",
        "preprocessing.json"
    ]
    
    for file_name in model_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在!")
            all_good = False
    
    # 检查推理模块
    print("\n检查推理模块:")
    inference_dir = deploy_bundle / "inference"
    inference_files = [
        "model_loader.py",
        "infer.py",
        "batch_infer.py"
    ]
    
    for file_name in inference_files:
        file_path = inference_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在!")
            all_good = False
    
    # 检查报告
    print("\n检查报告:")
    reports_dir = deploy_bundle / "reports"
    report_files = [
        "final_model_summary.md",
        "metrics.json"
    ]
    
    for file_name in report_files:
        file_path = reports_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在!")
            all_good = False
    
    # 检查图表
    print("\n检查图表:")
    figures_dir = deploy_bundle / "figures"
    figure_files = [
        "roc.png",
        "pr.png",
        "score_dist.png",
        "confusion.png"
    ]
    
    for file_name in figure_files:
        file_path = figures_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"⚠️  {file_name} 不存在 (可选)")
    
    return all_good

def verify_frontend():
    """验证前端应用"""
    print("\n" + "=" * 60)
    print("验证前端应用")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend_app"
    
    if not frontend_dir.exists():
        print("❌ frontend_app 目录不存在!")
        return False
    
    print("✅ frontend_app/ 目录存在")
    
    frontend_files = [
        "app.py",
        "run_frontend.bat"
    ]
    
    all_good = True
    for file_name in frontend_files:
        file_path = frontend_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在!")
            all_good = False
    
    return all_good

def verify_documentation():
    """验证文档"""
    print("\n" + "=" * 60)
    print("验证文档")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    doc_files = [
        "README_DEPLOY.md",
        "README_FRONTEND.md"
    ]
    
    all_good = True
    for file_name in doc_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在!")
            all_good = False
    
    return all_good

def test_model_import():
    """测试模型加载器导入"""
    print("\n" + "=" * 60)
    print("测试模型加载器导入")
    print("=" * 60)
    
    try:
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from deploy_bundle.inference.model_loader import TrafficAnomalyModel
        print("✅ TrafficAnomalyModel 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("\n" + "=" * 60)
    print("网络流量异常检测系统 - 完整性验证")
    print("=" * 60)
    
    results = []
    
    results.append(("deploy_bundle 目录结构", verify_deploy_bundle()))
    results.append(("前端应用", verify_frontend()))
    results.append(("文档", verify_documentation()))
    results.append(("模型加载器导入", test_model_import()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有验证通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 运行前端: cd frontend_app ; run_frontend.bat")
        print("2. 或查看文档: README_DEPLOY.md, README_FRONTEND.md")
    else:
        print("⚠️  部分验证失败，请检查上述错误信息。")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
