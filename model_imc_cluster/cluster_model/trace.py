"""JSON-safe structured traces and first-divergence comparison."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import numpy as np


def json_safe(value):
    """递归转换 dataclass/NumPy 类型，保留 Python 大整数而不转成浮点。"""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_jsonl(path, results):
    """每个 StepResult 写一行，包含候选/提交状态，便于逐步定位分歧。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False)+"\n")


def first_divergence(reference, actual):
    """Compare candidate AND committed five-plane states before looking at spikes."""
    for step, (ref, hw) in enumerate(zip(reference, actual)):
        # 只比较 spike 会遗漏局部历史分歧；最终发放还可能把候选差异清零。
        for state_name in ("candidate", "state"):
            for field in ("scu", "mr"):
                a, b = getattr(getattr(ref, state_name), field), getattr(getattr(hw, state_name), field)
                where = np.argwhere(a != b)
                if len(where):
                    m, bit = map(int, where[0])
                    return dict(step=step, field=f"{state_name}.{field}", macro=m, bit=bit,
                                reference=int(a[m, bit]), actual=int(b[m, bit]))
        if ref.spike != hw.spike:
            return dict(step=step, field="spike", reference=ref.spike, actual=hw.spike)
    if len(reference) != len(actual):
        return dict(step=min(len(reference), len(actual)), field="length")
    return None
