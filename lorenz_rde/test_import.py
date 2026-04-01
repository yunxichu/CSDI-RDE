#!/usr/bin/env python3
# 测试Lorenz系统代码导入
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lorenz_dir = os.path.join(base_dir, 'lorenz')
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(lorenz_dir, 'models'))
sys.path.insert(0, os.path.join(lorenz_dir, 'data'))
sys.path.insert(0, os.path.join(lorenz_dir, 'inference'))

print("测试Lorenz系统模块导入...")

try:
    from dataset_lorenz import generate_coupled_lorenz
    print("✓ dataset_lorenz 导入成功")
    
    from main_model import CSDI_Lorenz
    print("✓ main_model 导入成功")
    
    from gpr_module import GaussianProcessRegressor
    print("✓ gpr_module 导入成功")
    
    from diff_models import *
    print("✓ diff_models 导入成功")
    
    from test import load_model, impute
    print("✓ test 导入成功")
    
    from test2 import predict
    print("✓ test2 导入成功")
    
    print("\n所有模块导入成功！")
    
    print("\n测试数据生成...")
    lorenz_data, full_data = generate_coupled_lorenz(N=5, L=50, stepsize=8)
    print(f"✓ 数据生成成功！lorenz_data 形状: {lorenz_data.shape}")
    print(f"✓ full_data 形状: {full_data.shape}")
    
    print("\n✓ 所有测试通过！Lorenz系统代码整理完成！")
    
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
