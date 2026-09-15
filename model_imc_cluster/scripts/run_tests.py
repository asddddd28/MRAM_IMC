"""运行全部分层测试，保留退出码及 JUnit、JSON、原始文本报告。"""
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
# 使用脚本位置定位项目，避免从仓库根目录/其他工作目录启动时路径漂移。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster_model.trace import write_json


def main():
    """执行 pytest 并汇总本次运行；不把历史报告当成本次结果。"""
    report = ROOT / "reports" / "acceptance"
    report.mkdir(parents=True, exist_ok=True)
    # 如果 pytest 提前失败，没有产生新 XML，不能继续读取上一次的成功计数。
    (report / "junit.xml").unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests"), f"--junitxml={report / 'junit.xml'}"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    text = result.stdout + result.stderr
    print(text, end="")
    (report / "pytest-output.txt").write_text(text, encoding="utf-8")
    data = dict(command=command, exit_code=result.returncode, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), tested_scope="T01-T13 normalized behavior, mapping, APIs and CLI",
                not_implemented=["T14/P4 PyTorch gradient wrapper and training", "P5 calibration/multi-chip characterization",
                                 "arbitrary switch-event topology", "state/weight memory fault models", "three-mode controller"],
                calibrated=False)
    if (report / "junit.xml").exists():
        root = ET.parse(report / "junit.xml").getroot()
        suites = list(root.iter("testsuite"))
        data.update({k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")})
        data["seconds"] = sum(float(s.get("time", "0")) for s in suites)
        data["passed"] = data["tests"]-data["failures"]-data["errors"]-data["skipped"]
    write_json(report / "summary.json", data)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
