"""把 v6 融合配置写入已部署的 models/model.pkl（不重训，成员不变）。

配置来源：experiments/tune_v6_grid.py 在 screen 上逐目标精调 (blend_w, threshold, clf_tcn_w)，
并经 LOO 复核（固定配置、不在留出井重调）确认方向未被 screen 过拟合反转。

要点：
  - 只改 blend_w / thresholds / clf_tcn_w 三个字段；成员（classifiers/regressors/tcn_sds）不动。
  - clf_tcn_w 写为 dict（逐目标），src/model.py::predict 已支持 dict 或标量（见 :151）。
  - 写前备份为 models/model.pkl.v5bak，可回滚。
用法: python experiments/apply_v6_config.py [--dry-run]
"""
from __future__ import annotations
import argparse, os, shutil, sys, warnings
warnings.filterwarnings("ignore")
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import TARGETS

V5 = {"blend_w": {"POR": [0.45, 0.18333, 0.18333, 0.18333],
                  "PERM": [0.05, 0.31667, 0.31667, 0.31667],
                  "SW": [0.15, 0.28333, 0.28333, 0.28333]},
      "thresholds": {"POR": 0.12, "PERM": 0.07, "SW": 0.04},
      "clf_tcn_w": 0.55}
# 部署模型 blend_w 是长度 4 的列表: [树权重, TCN成员1, TCN成员2, TCN成员3]
# 逐目标 TCN 总权重 w 在三成员间等分（与 tune_v6_grid.py 的单一路径集成等价）
TCN_W = {"POR": 0.54, "PERM": 0.96, "SW": 0.80}
V6 = {"blend_w": {c: [1.0 - w] + [w / 3.0] * 3 for c, w in TCN_W.items()},
      "thresholds": {"POR": 0.115, "PERM": 0.065, "SW": 0.040},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.75, "SW": 0.30}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = "models/model.pkl"
    bak = path + ".v5bak"
    m = joblib.load(path)
    n_tcn = len(m["tcn_sds"])
    print("=== model.pkl 现状 ===")
    for k in ("blend_w", "thresholds", "clf_tcn_w", "target_transform"):
        print(f"  {k:16s} = {m.get(k)!r}")
    print(f"  tcn_sds            = {n_tcn} 个状态字典")
    assert n_tcn == 3, f"tcn_sds 应为 3 个，实为 {n_tcn}"
    assert "tcn2_sd" not in m, "存在 tcn2_sd：pass2 被误拼（见 CLAUDE.md §5.8）"
    # blend_w 必须是长度 n_tcn+1 的列表；若被写坏则先从备份恢复
    bw_bad = any(not isinstance(m["blend_w"][c], list) or len(m["blend_w"][c]) != n_tcn + 1
                 for c in TARGETS)
    if bw_bad:
        assert os.path.exists(bak), "blend_w 已损坏且无备份可恢复"
        print(f"\n  blend_w 格式错误（应为长度 {n_tcn+1} 的列表）→ 从 {bak} 恢复")
        shutil.copy2(bak, path)
        m = joblib.load(path)

    if args.dry_run:
        print("\n--dry-run：未写入")
        return

    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    for k, v in V6.items():
        m[k] = v
    joblib.dump(m, path)

    m2 = joblib.load(path)
    ok = (all(m2["blend_w"][c] == V6["blend_w"][c] for c in TARGETS)
          and m2["thresholds"] == V6["thresholds"]
          and m2["clf_tcn_w"] == V6["clf_tcn_w"]
          and len(m2["tcn_sds"]) == 3 and "tcn2_sd" not in m2)
    print(f"\n写回校验: {ok}")
    if not ok:
        shutil.copy2(bak, path)
        raise SystemExit("校验失败，已回滚到 .v5bak")
    print("已写 models/model.pkl")


if __name__ == "__main__":
    main()
