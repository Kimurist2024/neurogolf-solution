#!/usr/bin/env python3
"""Conv/ConvTranspose の bias 長 != 出力チャンネル数 を検出する。

bias 長 < out_channels は ORT 未定義動作で、提出を非決定に揺らして私的0点を
引き起こす地雷(task386 で実証、[[conv-bias-length-flip]])。フォーラムでも
「zip メンバー順で 0 点が変わる」現象の真因がこれと特定されている。

usage:
  check_bias.py FILE.onnx ...        # 個別ファイル
  check_bias.py DIR ...              # ディレクトリ内の *.onnx
  check_bias.py SUBMISSION.zip ...   # zip 内の *.onnx
出力: 問題ネットのみ。MINE(<) が危険な地雷、over(>) も非推奨。
終了コード: 地雷(<)が1件以上で 1、なければ 0。
"""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path

import onnx


def check_model(b: bytes, label: str):
    issues = []
    try:
        m = onnx.load_model_from_string(b)
    except Exception as e:  # noqa: BLE001
        return [(label, "LOAD_ERROR", str(e)[:80], True)]
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    for nd in m.graph.node:
        if nd.op_type not in ("Conv", "ConvTranspose"):
            continue
        if len(nd.input) < 3 or not nd.input[2]:
            continue  # bias なし(これは安全)
        wdims = inits.get(nd.input[1])
        bdims = inits.get(nd.input[2])
        if not wdims or not bdims:
            continue  # weight/bias が動的(通常発生しない)
        group = 1
        for a in nd.attribute:
            if a.name == "group":
                group = a.i
        if nd.op_type == "Conv":
            out_ch = wdims[0]
        else:  # ConvTranspose: weight = [in_ch, out_ch/group, kH, kW]
            out_ch = wdims[1] * group
        blen = bdims[0]
        if blen != out_ch:
            is_mine = blen < out_ch
            sev = "MINE(<)" if is_mine else "over(>)"
            issues.append((label, nd.op_type,
                           f"{nd.name or '?'} bias={blen} out_ch={out_ch} [{sev}]",
                           is_mine))
    return issues


def iter_inputs(args):
    for a in args:
        p = Path(a)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in sorted(z.namelist()):
                    if n.endswith(".onnx"):
                        yield f"{p.name}:{n}", z.read(n)
        elif p.suffix == ".onnx":
            yield str(p), p.read_bytes()
        elif p.is_dir():
            for f in sorted(p.glob("*.onnx")):
                yield f.name, f.read_bytes()


def main() -> int:
    total = 0
    mines = 0
    overs = 0
    for label, b in iter_inputs(sys.argv[1:]):
        total += 1
        for lab, op, msg, is_mine in check_model(b, label):
            if is_mine:
                mines += 1
                print(f"🔴 {lab}  {op}  {msg}")
            else:
                overs += 1
                print(f"🟡 {lab}  {op}  {msg}")
    print(f"\n検査 {total} ネット / 地雷(<) {mines} 件 / over(>) {overs} 件")
    return 1 if mines else 0


if __name__ == "__main__":
    raise SystemExit(main())
