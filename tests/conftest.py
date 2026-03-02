from __future__ import annotations

import sys
from pathlib import Path

# pytest 启动时，把项目根目录加入 sys.path，
# 这样测试可以直接 import src.xxx 模块。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
