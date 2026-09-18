"""numpy 前向 <-> torch 前向 parity 测试（改动网络后必须跑，误差 <1e-4）.

注意: 必须 m.eval() 关闭 dropout，否则随机性导致误报。
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    from src.model import get_tcn_classes, numpy_forward_tcn
    cls = get_tcn_classes()
    TCNNet = cls["TCNNet"]
    torch.manual_seed(0); np.random.seed(0)
    F = 23
    m = TCNNet(F, ch=32, dilations=(1, 2, 4), repeats=1).eval()
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    x = np.random.randn(200, F).astype(np.float32)
    with torch.no_grad():
        o = m(torch.from_numpy(x).unsqueeze(0))
    np_o = numpy_forward_tcn(x, sd, [1, 2, 4])
    ok = True
    for c in ("POR", "PERM", "SW"):
        err = float(np.abs(o[c].squeeze(0).numpy() - np_o[c]).max())
        print(f"{c} max_err={err:.3e}")
        ok = ok and err < 1e-4
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
