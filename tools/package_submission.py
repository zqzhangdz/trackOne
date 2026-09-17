"""按比赛规范生成 submission_code.zip 与 result.zip，并跑合规 checklist.

用法:
    python tools/package_submission.py [--workdir .] [--skip_predict]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

SPEC_FILES = [
    "README.md",
    "predict.py",
    "requirements.txt",
    "config.yaml",
    "train.py",
    "src/model.py",
    "src/feature.py",
    "src/utils.py",
    "models/model.pkl",
    "models/scaler.pkl",
    "examples/result_example.json",
]


def build_code_zip(workdir: str, out_zip: str) -> list[str]:
    tmp = tempfile.mkdtemp(prefix="subpkg_")
    root = os.path.join(tmp, "submission_code")
    os.makedirs(root)
    copied = []
    for rel in SPEC_FILES:
        src = os.path.join(workdir, rel)
        dst = os.path.join(root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"缺少规范文件: {rel}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    # src 包需要 __init__.py 才能 import（规范树未列但 Python 包必需，置于 src/ 内）
    init = os.path.join(root, "src", "__init__.py")
    if not os.path.exists(init):
        with open(init, "w", encoding="utf-8") as f:
            f.write('"""算法模块包."""\n')
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(copied + ["src/__init__.py"]):
            z.write(os.path.join(root, rel), rel)
    shutil.rmtree(tmp)
    return copied


def validate_code_zip(zip_path: str, workdir: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    missing = [f for f in SPEC_FILES if f not in names]
    if missing:
        raise RuntimeError(f"zip 缺少规范文件: {missing}")
    extra_top = {n.split("/")[0] for n in names} - {
        "README.md", "predict.py", "requirements.txt", "config.yaml", "train.py", "src", "models", "examples"
    }
    if extra_top:
        raise RuntimeError(f"zip 顶层存在规范外目录/文件: {extra_top}")
    print(f"[check] 规范文件齐备: {len(SPEC_FILES)} 个; 顶层无多余项")


def smoke_predict(zip_path: str, workdir: str, python_bin: str) -> None:
    tmp = tempfile.mkdtemp(prefix="subrun_")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    data_dir = os.path.join(workdir, "test")
    out_json = os.path.join(tmp, "result_smoke.json")
    cmd = [python_bin, "predict.py", "--data_dir", data_dir, "--output", out_json]
    print("[check] 烟雾运行:", " ".join(cmd))
    subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)
    with open(out_json, encoding="utf-8") as f:
        res = json.load(f)
    validate_result_schema(res)
    print(f"[check] 烟雾预测成功: {len(res['resultData'])} 井")
    shutil.rmtree(tmp)


def validate_result_schema(res: dict) -> None:
    assert set(res.keys()) >= {"modelId", "modelName", "version", "resultData"}, "result 顶层字段缺失"
    n = 0
    for item in res["resultData"]:
        assert "logId" in item and "predictions" in item
        depths = [p["depth"] for p in item["predictions"]]
        assert depths == sorted(depths), f"{item['logId']} depth 未升序"
        for p in item["predictions"]:
            assert set(p.keys()) == {"depth", "POR", "PERM", "SW"}, f"预测字段错误: {p.keys()}"
            for k in ("POR", "PERM", "SW"):
                v = p[k]
                assert isinstance(v, (int, float)) and v == v, f"非法数值 {k}={v}"
        n += len(item["predictions"])
    print(f"[check] result schema 合法: {len(res['resultData'])} 井, {n:,} 点, depth 升序")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_predict", type=int, default=0)
    args = ap.parse_args()
    workdir = os.path.abspath(args.workdir)

    code_zip = os.path.join(workdir, "submission_code.zip")
    build_code_zip(workdir, code_zip)
    print("已生成", code_zip)
    validate_code_zip(code_zip, workdir)
    if not args.skip_predict:
        smoke_predict(code_zip, workdir, args.python)

    result_json = os.path.join(workdir, "result.json")
    if os.path.exists(result_json):
        with open(result_json, encoding="utf-8") as f:
            validate_result_schema(json.load(f))
        result_zip = os.path.join(workdir, "result.zip")
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(result_json, "result.json")
        print("已生成", result_zip)
    else:
        print("警告: 未找到 result.json, 跳过 result.zip（先运行 predict.py）")


if __name__ == "__main__":
    main()
