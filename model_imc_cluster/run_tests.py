"""兼容原有测试入口；实际实现位于 scripts/run_tests.py。"""
from scripts.run_tests import main


if __name__ == "__main__":
    raise SystemExit(main())
