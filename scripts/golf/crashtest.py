#!/usr/bin/env python3
"""採用 net を全 H×W 入力で crashtest する。

私的ベンチは任意サイズのグリッドを投げてくる。ある形状で実行時例外を出す net は
grader ERROR を誘発し、提出全体を 0 点にする(task324/79 で実証)。require_correct
の arc-gen 検証は全形状を網羅しないので、1..30 × 1..30 を別途総当たりする。

usage:
  crashtest.py SUBMISSION.zip TASK [TASK ...]   # 指定タスクのみ
  crashtest.py SUBMISSION.zip                    # zip 内の全 taskNNN.onnx(提出前ゲート)
終了コード: crash する net が1件以上で 1。
"""
from __future__ import annotations
import re
import sys
import zipfile

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "scripts")
from lib import scoring  # noqa: E402

# ORT のカーネル実行エラーは stderr に大量に出る。crash 検出は sess.run の Python 例外で
# 行うためログには依存しない。FATAL のみ表示にしてノイズを抑える。
try:
    ort.set_default_logger_severity(4)
except Exception:  # noqa: BLE001
    pass

MAX_EINSUM_OPERANDS = 15  # このオペランド数以上の Einsum はローカル ORT でハング(grader は正常)


def make_input(h: int, w: int, color: int) -> np.ndarray:
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    x[0, color, :h, :w] = 1.0  # グリッド内 one-hot、境界外 zero-hot
    return x


def _is_hang_prone(model: onnx.ModelProto) -> bool:
    """巨大 fused-Einsum はローカル ORT でハングするが grader は正常採点する。"""
    for nd in model.graph.node:
        if nd.op_type == "Einsum" and len(nd.input) >= MAX_EINSUM_OPERANDS:
            return True
    return False


def crashtest(b: bytes):
    # 返り値: None=スキップ(giant fused-Einsum), list=crash した形状([]==clean)
    # grader 同様に sanitize(ノード名重複の解消など)してからロードする
    try:
        model = onnx.load_model_from_string(b)
        if _is_hang_prone(model):
            return None  # ローカル ORT でハングする net はスキップ(grader は正常)
        sm = scoring.sanitize_model(model)
        if sm is not None:
            model = sm
        b = model.SerializeToString()
        so = ort.SessionOptions()
        so.log_severity_level = 4
        sess = ort.InferenceSession(b, so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        return [(0, 0, -1, "LOAD: " + str(e)[:55])]
    iname = sess.get_inputs()[0].name
    crashes = []
    for h in range(1, 31):
        for w in range(1, 31):
            for color in (0, 1, 5):
                try:
                    sess.run(None, {iname: make_input(h, w, color)})
                except Exception as e:  # noqa: BLE001
                    crashes.append((h, w, color, str(e)[:60]))
                    break  # この形状で crash 確定 → 次の形状へ
    return crashes


def main() -> int:
    zf = sys.argv[1]
    z = zipfile.ZipFile(zf)
    # 引数はスペース区切りでも個別でも受ける(zsh の word-split 差異に頑健)。
    # タスク未指定なら zip 内の全 taskNNN.onnx を対象にする(提出前ゲート)。
    arg_tasks = " ".join(sys.argv[2:]).split()
    if arg_tasks:
        tasks = [int(x) for x in arg_tasks]
    else:
        tasks = sorted(
            int(m.group(1))
            for n in z.namelist()
            if (m := re.match(r"task(\d{3})\.onnx$", n))
        )
    names = set(z.namelist())
    bad = skipped = missing = 0
    for t in tasks:
        name = f"task{t:03d}.onnx"
        if name not in names:
            missing += 1
            continue
        cr = crashtest(z.read(name))
        if cr is None:
            skipped += 1
            continue
        if cr:
            bad += 1
            print(f"🔴 task{t:03d}: {len(cr)} 形状で crash  例: H{cr[0][0]}×W{cr[0][1]} "
                  f"color{cr[0][2]} -> {cr[0][3]}")
    print(f"\ncrashtest {len(tasks)} 件 / crash あり {bad} 件 / "
          f"giant-skip {skipped} 件 / 欠落 {missing} 件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
