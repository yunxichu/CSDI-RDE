#!/usr/bin/env python3
# 测试导入模块
import os
import sys

print("测试 test.py 的模块导入...")
sys.path.insert(0, '/home/rhl/Github/rde_gpr')
from test import load_model, impute
print("✓ test.py 模块导入成功")

print("\n测试 test2.py 的模块导入...")
from test2 import predict
print("✓ test2.py 模块导入成功")

print("\n所有模块导入成功！")
