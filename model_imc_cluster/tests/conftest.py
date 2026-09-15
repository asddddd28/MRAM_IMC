"""统一测试导入环境；各测试分组无需重复修改 sys.path。"""
from pathlib import Path
import sys

# 此文件留在 tests/ 根目录，使验收/API/CLI 分组共享相同项目根路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
