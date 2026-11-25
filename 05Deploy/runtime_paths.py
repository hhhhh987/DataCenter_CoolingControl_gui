"""
runtime_paths.py
统一路径出口: 全项目唯一的 frozen 判断处
打包后(frozen): exe 放在"05Deploy"层, DEPLOY_DIR = exe 所在目录
开发时: DEPLOY_DIR = 本文件所在目录(05Deploy)
"""
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    DEPLOY_DIR = Path(sys.executable).resolve().parent
else:
    DEPLOY_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = DEPLOY_DIR.parent