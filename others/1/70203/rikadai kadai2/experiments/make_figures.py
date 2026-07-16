#!/usr/bin/env python3
"""実験CSVからレポート用の白黒図(PDF)を生成する."""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)

# 白黒印刷で判別できるスタイル(グレースケール+線種+マーカー)
plt.rcParams.update({
    "font.family": "Hiragino Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
STYLES = {
    "add":     dict(color="0.0", linestyle="-",  marker="o", markersize=4),
    "djb2":    dict(color="0.35", linestyle="--", marker="s", markersize=4),
    "fnv1a":   dict(color="0.55", linestyle="-.", marker="^", markersize=4),
    "stdhash": dict(color="0.7", linestyle=":",  marker="d", markersize=4),
}
LABELS = {
    "add": "加算ハッシュ (hash.c)",
    "djb2": "djb2",
    "fnv1a": "FNV-1a",
    "stdhash": "std::hash",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_collision(m, outname):
    """バケットごとの要素数分布(ソート済み)を4関数で比較."""
    rows = read_csv(DATA / f"collision_m{m}.csv")
    counts = defaultdict(list)
    for r in rows:
        counts[r["hash"]].append(int(r["count"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    n = sum(counts["add"])
    expected = n / m
    for name in ["add", "djb2", "fnv1a", "stdhash"]:
        ys = sorted(counts[name], reverse=True)
        ax.plot(range(len(ys)), ys, label=LABELS[name], linewidth=1.2,
                **{k: v for k, v in STYLES[name].items()},
                markevery=max(1, len(ys) // 25))
    ax.axhline(expected, color="0.2", linewidth=0.8, linestyle=(0, (1, 3)))
    ax.annotate(f"一様分布の期待値 {expected:.1f}", xy=(len(counts['add']) * 0.55, expected),
                xytext=(0, 5), textcoords="offset points", fontsize=9, color="0.2")
    ax.set_xlabel(f"バケット順位（要素数の降順, 全 {m} バケット）")
    ax.set_ylabel("バケット内の要素数 [個]")
    ax.legend()
    fig.savefig(FIGS / outname)
    plt.close(fig)
    print(f"wrote {outname}")


def fig_loadfactor():
    """負荷率と平均比較回数(実測 vs 理論)."""
    rows = read_csv(DATA / "loadfactor.csv")
    series = defaultdict(lambda: ([], [], []))  # hash -> (alpha, succ, unsucc)
    for r in rows:
        a, s, u = series[r["hash"]]
        a.append(float(r["alpha"]))
        s.append(float(r["succ_avg"]))
        u.append(float(r["unsucc_avg"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    alphas = series["add"][0]
    theory_x = [a for a in alphas]
    for ax, idx, title, theory in (
        (axes[0], 1, "成功検索", [1 + a / 2 for a in theory_x]),
        (axes[1], 2, "不成功検索", theory_x),
    ):
        for name in ["add", "fnv1a"]:
            a = series[name][0]
            y = series[name][idx]
            ax.plot(a, y, label=LABELS[name], linewidth=1.2, **STYLES[name])
        ax.plot(theory_x, theory, color="0.0", linestyle=(0, (1, 2)), linewidth=1.8,
                label="理論値" + (" $1+\\alpha/2$" if idx == 1 else " $\\alpha$"))
        ax.set_xlabel("負荷率 $\\alpha = n / m$")
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("平均キー比較回数 [回]")
    fig.savefig(FIGS / "loadfactor.png")
    plt.close(fig)
    print("wrote loadfactor.png")


def fig_timing():
    """要素数と挿入/検索時間(中央値)の両対数プロット."""
    path = DATA / "timing.csv"
    if not path.exists():
        print("timing.csv not found; skip")
        return
    rows = read_csv(path)
    agg = defaultdict(lambda: defaultdict(list))  # structure -> N -> [(ins, lk)]
    for r in rows:
        agg[r["structure"]][int(r["N"])].append(
            (float(r["insert_total_us"]), float(r["lookup_avg_ns"])))
    st_styles = {
        "fixed17":   dict(color="0.0", linestyle="-", marker="o", markersize=4),
        "improved":  dict(color="0.35", linestyle="--", marker="s", markersize=4),
        "unordered_map": dict(color="0.55", linestyle="-.", marker="^", markersize=4),
        "map":       dict(color="0.7", linestyle=":", marker="d", markersize=4),
    }
    st_labels = {
        "fixed17": "提出版 ($m=17$ 固定)",
        "improved": "改良版 (FNV-1a+リハッシュ)",
        "unordered_map": "std::unordered\\_map",
        "map": "std::map",
    }
    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    for idx, (ylabel, outname) in enumerate(
            [("挿入合計時間 [$\\mu$s]", "timing_insert.png"),
             ("1 回あたり平均検索時間 [ns]", "timing_lookup.png")]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for st in ["fixed17", "improved", "unordered_map", "map"]:
            if st not in agg:
                continue
            ns = sorted(agg[st].keys())
            ys = [median([v[idx] for v in agg[st][n]]) for n in ns]
            label = st_labels[st].replace("\\_", "_")
            ax.plot(ns, ys, label=label, linewidth=1.2, **st_styles[st])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("要素数 $N$ [個]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        fig.savefig(FIGS / outname)
        plt.close(fig)
        print(f"wrote {outname}")


def fig_rehash():
    """挿入レイテンシの時系列(リハッシュのスパイク)と累積平均."""
    path = DATA / "rehash.csv"
    if not path.exists():
        print("rehash.csv not found; skip")
        return
    rows = read_csv(path)
    samples = [(int(r["i"]), float(r["value_ns"])) for r in rows if r["kind"] == "sample"]
    spikes = [(int(r["i"]), float(r["value_ns"])) for r in rows if r["kind"] == "spike"]
    rehashes = [int(r["i"]) for r in rows if r["kind"] == "rehash"]
    cumavg = [(int(r["i"]), float(r["value_ns"])) for r in rows if r["kind"] == "cumavg"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot([i for i, _ in samples], [v for _, v in samples],
            color="0.6", linewidth=0.5, label="挿入レイテンシ（100 件ごと）")
    ax.plot([i for i, _ in spikes], [v for _, v in spikes],
            linestyle="none", marker="x", color="0.0", markersize=5,
            label="スパイク（50 $\\mu$s 超）")
    for k, i in enumerate(rehashes):
        ax.axvline(i, color="0.3", linewidth=0.6, linestyle="--",
                   label="リハッシュ発生" if k == 0 else None)
    ax.set_yscale("log")
    ax.set_ylabel("挿入 1 件のレイテンシ [ns]")
    ax.legend(fontsize=9, loc="upper left")

    ax2 = axes[1]
    ax2.plot([i for i, _ in cumavg], [v for _, v in cumavg],
             color="0.0", linewidth=1.2, label="累積平均挿入時間")
    ax2.set_xlabel("挿入番号 $i$ [件目]")
    ax2.set_ylabel("累積平均 [ns]")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9)
    fig.savefig(FIGS / "rehash.png")
    plt.close(fig)
    print("wrote rehash.png")


def fig_memory():
    """1 要素あたりメモリ使用量の比較(棒グラフ)."""
    path = DATA / "memory.csv"
    if not path.exists():
        print("memory.csv not found; skip")
        return
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5:
                rows.append(parts)
    labels_map = {"improved": "改良版", "unordered_map": "std::unordered_map",
                  "map": "std::map"}
    ns = [100000, 1000000]
    structs = ["improved", "unordered_map", "map"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.35
    hatches = {100000: "", 1000000: "///"}
    grays = {100000: "0.55", 1000000: "0.85"}
    for j, n in enumerate(ns):
        vals = []
        for st in structs:
            delta = next(int(r[4]) for r in rows if r[0] == st and int(r[1]) == n)
            vals.append(delta / n)
        xs = [k + (j - 0.5) * width for k in range(len(structs))]
        ax.bar(xs, vals, width=width, color=grays[n], edgecolor="0.0",
               hatch=hatches[n], label=f"$N = 10^{{{len(str(n)) - 1}}}$")
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(structs)))
    ax.set_xticklabels([labels_map[s] for s in structs])
    ax.set_ylabel("1 要素あたりの増分メモリ [バイト/個]")
    ax.legend()
    fig.savefig(FIGS / "memory.png")
    plt.close(fig)
    print("wrote memory.png")


def fig_openaddr():
    """開放番地法(線形走査)とチェイン法のプローブ数比較."""
    path = DATA / "openaddr.csv"
    if not path.exists():
        print("openaddr.csv not found; skip")
        return
    rows = read_csv(path)
    series = defaultdict(lambda: defaultdict(list))  # method -> field -> list
    for r in rows:
        m = r["method"]
        series[m]["alpha"].append(float(r["alpha"]))
        series[m]["succ"].append(float(r["succ_probes"]))
        series[m]["unsucc"].append(float(r["unsucc_probes"]))
    m_styles = {
        "chained": dict(color="0.0", linestyle="-", marker="o", markersize=4),
        "openaddr": dict(color="0.45", linestyle="--", marker="s", markersize=4),
    }
    m_labels = {"chained": "チェイン法", "openaddr": "開放番地法（線形走査）"}
    alphas_t = [0.25 + 0.01 * i for i in range(71)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, key, title, theories in (
        (axes[0], "succ", "成功検索",
         [("チェイン法理論 $1+\\alpha/2$", [1 + a / 2 for a in alphas_t], "0.0"),
          ("線形走査理論 $(1+\\frac{1}{1-\\alpha})/2$",
           [(1 + 1 / (1 - a)) / 2 for a in alphas_t], "0.45")]),
        (axes[1], "unsucc", "不成功検索",
         [("チェイン法理論 $1+\\alpha$", [1 + a for a in alphas_t], "0.0"),
          ("線形走査理論 $(1+\\frac{1}{(1-\\alpha)^2})/2$",
           [(1 + 1 / (1 - a) ** 2) / 2 for a in alphas_t], "0.45")]),
    ):
        for m in ["chained", "openaddr"]:
            ax.plot(series[m]["alpha"], series[m][key], linestyle="none",
                    marker=m_styles[m]["marker"], markersize=5,
                    color=m_styles[m]["color"], label=m_labels[m] + "（実測）")
        for lbl, ys, col in theories:
            ax.plot(alphas_t, ys, color=col, linewidth=1.0,
                    linestyle=(0, (1, 2)), label=lbl)
        ax.set_xlabel("負荷率 $\\alpha$")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("平均プローブ数 [回]")
    fig.savefig(FIGS / "openaddr.png")
    plt.close(fig)
    print("wrote openaddr.png")


def fig_keylen():
    """キー長とハッシュ計算・検索時間の関係."""
    path = DATA / "keylen.csv"
    if not path.exists():
        print("keylen.csv not found; skip")
        return
    rows = read_csv(path)
    ls = [int(r["keylen"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ls, [float(r["fnv1a_hash_ns"]) for r in rows], label="FNV-1a ハッシュ計算",
            **STYLES["fnv1a"], linewidth=1.2)
    ax.plot(ls, [float(r["add_hash_ns"]) for r in rows], label="加算ハッシュ計算",
            **STYLES["add"], linewidth=1.2)
    ax.plot(ls, [float(r["lookup_ns"]) for r in rows], label="成功検索（FNV-1a 表）",
            **STYLES["djb2"], linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("キー長 $L$ [文字]")
    ax.set_ylabel("1 回あたり平均時間 [ns]")
    ax.legend(fontsize=9)
    fig.savefig(FIGS / "keylen.png")
    plt.close(fig)
    print("wrote keylen.png")


if __name__ == "__main__":
    fig_collision(17, "collision_m17.png")
    fig_collision(1031, "collision_m1031.png")
    fig_loadfactor()
    fig_timing()
    fig_rehash()
    fig_memory()
    fig_openaddr()
    fig_keylen()
