# -*- coding: utf-8 -*-
"""exp05: n-gram 構成方式と stop words 設計の定量比較

課題6 で採用した「stop words を除去した後に 2-gram を構成する」設計
(方式 A)について、次の 4 つの追加実験を行う。

  [1] 2-gram 構成方式の比較
      (A) stop words 除去後に 2-gram 構成(提出プログラムの採用手法)
      (B) 原文の単語列から 2-gram を構成した後、どちらかの語が
          stop word である組を除去
      上位 10 の比較と、方式 A のみに現れる「原文に隣接して存在しない
      2-gram(非隣接アーティファクト)」の定量化を行う。
  [2] n-gram の次数 n を 1, 2, 3 と変えたときの上位 10 と統計量
      (異なり数・ハパックス率・上位 10 被覆率)を stop words
      除去あり/なしで集計する。
  [3] Zipf 則の検証: 1-gram と 2-gram の頻度-順位分布を log-log で
      プロットし、最小二乗法で傾きを推定して理論値(約 -1)と比較する。
  [4] stop words 設計の感度: stop words 集合を
      {なし, 基本 30 語, 採用集合, 採用集合+1 文字語 26 語} と変えた
      ときの 2-gram 上位 10 の変化(「e g」の消滅を含む)を調べる。

実行方法:
  cd "/Users/kimura2003/Downloads/rikadai kadai python" && \
  DATASET_ROOT="$PWD" MPLBACKEND=Agg \
  python3 experiments/exp05_ngram_design.py > results/exp05.txt 2>&1
"""

import csv
import os
import platform
import re
import string
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------
ROOT = os.environ.get("DATASET_ROOT", ".")
TEXT_PATH = os.path.join(ROOT, "dataset", "text", "text_english.txt")
DATA_DIR = os.path.join(ROOT, "experiments", "data")
FIG_DIR = os.path.join(ROOT, "experiments", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TOP_N = 10

# 白黒印刷で判別できる描画設定
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "0.7",
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------
# stop words 集合
# ---------------------------------------------------------------------
# STOP_WORDS_ADOPTED は提出プログラム src/6324034_final_exam.py の
# STOP_WORDS と同一の集合である。experiments/ ディレクトリからは
# パスの都合で src/ を import できないため、ここに同一内容をコピーして
# 再掲している(同一性の確認のため語数を出力する)。
STOP_WORDS_ADOPTED = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "and", "or", "but", "on", "at", "by", "for",
    "with", "from", "as", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "you",
    "i", "my", "your", "our", "not", "no", "so", "if", "than", "then",
    "there", "here", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "shall", "should", "may", "might", "must",
    "am", "into", "up", "down", "out", "about", "over", "under", "all",
    "any", "each", "which", "who", "whom", "what", "when", "where",
    "how", "why", "s", "t",
}

# 感度実験用の「基本 30 語」: 英語の代表的な機能語(冠詞・接続詞・
# be 動詞・頻出前置詞・頻出代名詞)のみからなる最小構成の集合
BASIC_STOP_WORDS_30 = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "of", "to", "in", "on", "at", "by", "for", "with", "as",
    "that", "this", "it", "not", "no", "i", "you", "he", "she", "they",
}

# 1 文字語(a-z の 26 語)。トークナイザ [a-z]+ が「e.g.」「1950s」等を
# 分割して生む 1 文字トークンをまとめて除去するための追加集合
SINGLE_LETTERS = set(string.ascii_lowercase)

STOP_WORD_SETS = [
    ("none", "stop words なし", set()),
    ("basic30", "基本 30 語", BASIC_STOP_WORDS_30),
    ("adopted", "採用集合(提出プログラムと同一)", STOP_WORDS_ADOPTED),
    ("adopted+letters", "採用集合 + 1 文字語 26 語",
     STOP_WORDS_ADOPTED | SINGLE_LETTERS),
]


# ---------------------------------------------------------------------
# 共通処理(トークン化・n-gram 構成は提出プログラムと同一仕様)
# ---------------------------------------------------------------------
def tokenize(text):
    """テキストを小文字化し、英単語のみを取り出す(提出プログラムと同一)。"""
    return re.findall(r"[a-z]+", text.lower())


def build_ngrams(words, n):
    """単語列から連続する n 語の組のリストを作る(n=2 は提出プログラムと同一)。"""
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def ranked_items(counter):
    """頻度降順(同数は辞書順)で決定的に並べた (gram, count) のリスト。"""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


def rank_map(counter):
    """gram -> 順位(1 始まり)の辞書。順位は ranked_items の並びで定義する。"""
    return {g: i + 1 for i, (g, c) in enumerate(ranked_items(counter))}


def print_top(counter, top_n=TOP_N, marks=None):
    """上位 top_n を表形式で出力する。marks は gram -> 注記文字列。"""
    for i, (g, c) in enumerate(ranked_items(counter)[:top_n], start=1):
        note = (marks or {}).get(g, "")
        print(f"  {i:>2d}. {g:<32s} {c:>4d}  {note}")


def hbar_top(ax, counter, title, top_n=TOP_N, hatch_grams=None,
             color="0.6", hatch_color="0.9", hatch="//"):
    """横棒グラフで上位 top_n を描く(白黒判別: グレー + ハッチ)。"""
    top = ranked_items(counter)[:top_n]
    labels = [g for g, _ in top]
    values = [c for _, c in top]
    hatch_grams = hatch_grams or set()
    for i, (g, v) in enumerate(zip(labels, values)):
        if g in hatch_grams:
            ax.barh(i, v, color=hatch_color, edgecolor="black",
                    hatch=hatch, height=0.65)
        else:
            ax.barh(i, v, color=color, edgecolor="black", height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------
# [0] コーパスの基本統計
# ---------------------------------------------------------------------
print("=" * 70)
print("exp05: n-gram 構成方式と stop words 設計の定量比較")
print("=" * 70)
print(f"platform : {platform.platform()}")
print(f"python   : {platform.python_version()}")
print(f"numpy    : {np.__version__}")
print(f"text     : {TEXT_PATH}")
print()

with open(TEXT_PATH, encoding="utf-8") as f:
    text = f.read()
words = tokenize(text)
uni_raw = Counter(words)

print("-" * 70)
print("[0] コーパスの基本統計")
print("-" * 70)
print(f"総トークン数(正規表現 [a-z]+): {len(words)}")
print(f"異なり語数                    : {len(uni_raw)}")
print(f"採用 stop words の語数        : {len(STOP_WORDS_ADOPTED)} "
      f"(提出プログラムの STOP_WORDS と同一)")
print(f"基本 stop words の語数        : {len(BASIC_STOP_WORDS_30)}")
print(f"採用+1文字語の語数            : "
      f"{len(STOP_WORDS_ADOPTED | SINGLE_LETTERS)}")
print()

# ---------------------------------------------------------------------
# [1] 2-gram 構成方式の比較(方式 A vs 方式 B)
# ---------------------------------------------------------------------
print("-" * 70)
print("[1] 2-gram 構成方式の比較")
print("    方式 A: stop words 除去後に 2-gram 構成(採用手法)")
print("    方式 B: 2-gram 構成後に stop word を含む組を除去")
print("-" * 70)

filtered = [w for w in words if w not in STOP_WORDS_ADOPTED]
bigrams_A = Counter(build_ngrams(filtered, 2))

raw_bigrams = build_ngrams(words, 2)
raw_bigram_set = set(raw_bigrams)
bigrams_B = Counter(
    g for g in raw_bigrams
    if all(w not in STOP_WORDS_ADOPTED for w in g.split())
)

print(f"stop words 除去後のトークン数        : {len(filtered)}")
print(f"方式 A: 2-gram 総数 {sum(bigrams_A.values())}, "
      f"異なり数 {len(bigrams_A)}")
print(f"方式 B: 2-gram 総数 {sum(bigrams_B.values())}, "
      f"異なり数 {len(bigrams_B)}")

# 方式 A の「非隣接アーティファクト」: 原文の隣接 2-gram に存在しない組
artifact_types = [g for g in bigrams_A if g not in raw_bigram_set]
artifact_tokens = sum(bigrams_A[g] for g in artifact_types)
print(f"方式 A のうち原文に隣接して存在しない 2-gram: "
      f"異なり数 {len(artifact_types)} / {len(bigrams_A)} "
      f"({len(artifact_types) / len(bigrams_A) * 100:.1f} %), "
      f"総数 {artifact_tokens} / {sum(bigrams_A.values())} "
      f"({artifact_tokens / sum(bigrams_A.values()) * 100:.1f} %)")
print()

marks_A = {g: "(非隣接)" for g in artifact_types}
print("方式 A 上位 10(「(非隣接)」は原文に隣接ペアとして存在しない組):")
print_top(bigrams_A, marks=marks_A)
print()
print("方式 B 上位 10:")
print_top(bigrams_B)
print()

# 上位 10 の和集合について順位と頻度を比較
rankA = rank_map(bigrams_A)
rankB = rank_map(bigrams_B)
topA = [g for g, _ in ranked_items(bigrams_A)[:TOP_N]]
topB = [g for g, _ in ranked_items(bigrams_B)[:TOP_N]]
union = sorted(set(topA) | set(topB),
               key=lambda g: (rankA.get(g, 10 ** 6), g))
print("上位 10 の和集合の順位比較(順位 '-' はその方式に出現しないこと):")
print(f"  {'2-gram':<32s} {'A頻度':>5s} {'A順位':>5s} "
      f"{'B頻度':>5s} {'B順位':>5s}  原文に隣接")
for g in union:
    ra = str(rankA[g]) if g in rankA else "-"
    rb = str(rankB[g]) if g in rankB else "-"
    ca = bigrams_A.get(g, 0)
    cb = bigrams_B.get(g, 0)
    adj = "yes" if g in raw_bigram_set else "no"
    print(f"  {g:<32s} {ca:>5d} {ra:>5s} {cb:>5d} {rb:>5s}  {adj}")
print()

focus = "generation natural"
print(f"確認: 「{focus}」 方式 A 頻度 {bigrams_A.get(focus, 0)} "
      f"(順位 {rankA.get(focus, '-')}), "
      f"方式 B 頻度 {bigrams_B.get(focus, 0)}, "
      f"原文に隣接ペアとして存在: {focus in raw_bigram_set}")
print()

csv_methods = os.path.join(DATA_DIR, "exp05_methods_top10.csv")
with open(csv_methods, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "method_A_gram", "method_A_count", "A_nonadjacent",
                "method_B_gram", "method_B_count"])
    ra10 = ranked_items(bigrams_A)[:TOP_N]
    rb10 = ranked_items(bigrams_B)[:TOP_N]
    for i in range(TOP_N):
        ga, ca = ra10[i]
        gb, cb = rb10[i]
        w.writerow([i + 1, ga, ca,
                    "yes" if ga in artifact_types else "no", gb, cb])
print(f"CSV 保存: {csv_methods}")

csv_union = os.path.join(DATA_DIR, "exp05_methods_union.csv")
with open(csv_union, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gram", "count_A", "rank_A", "count_B", "rank_B",
                "adjacent_in_raw"])
    for g in union:
        w.writerow([g, bigrams_A.get(g, 0), rankA.get(g, ""),
                    bigrams_B.get(g, 0), rankB.get(g, ""),
                    "yes" if g in raw_bigram_set else "no"])
print(f"CSV 保存: {csv_union}")

# 図1: 方式 A / B の上位 10(非隣接組はハッチで区別)
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
hbar_top(axes[0], bigrams_A,
         "(A) remove stop words, then 2-grams (adopted)",
         hatch_grams=set(artifact_types))
hbar_top(axes[1], bigrams_B,
         "(B) 2-grams first, then remove stop-word pairs")
axes[0].text(0.98, 0.02, "hatched = not adjacent in raw text",
             transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)
fig.tight_layout()
fig_methods = os.path.join(FIG_DIR, "exp05_methods.pdf")
fig.savefig(fig_methods)
plt.close(fig)
print(f"図 保存: {fig_methods}")
print()

# ---------------------------------------------------------------------
# [2] n-gram の次数 n = 1, 2, 3 の比較
# ---------------------------------------------------------------------
print("-" * 70)
print("[2] n-gram の次数 n = 1, 2, 3 の比較(stop words 除去あり/なし)")
print("-" * 70)

variants = [("raw", "除去なし", words), ("filtered", "除去あり", filtered)]
ngram_stats = []
ngram_top_rows = []
for vkey, vlabel, seq in variants:
    for n in (1, 2, 3):
        counter = Counter(build_ngrams(seq, n))
        tokens = sum(counter.values())
        types = len(counter)
        hapax = sum(1 for c in counter.values() if c == 1)
        top10 = ranked_items(counter)[:TOP_N]
        top10_tokens = sum(c for _, c in top10)
        coverage = top10_tokens / tokens * 100
        ngram_stats.append((vkey, vlabel, n, tokens, types, hapax,
                            hapax / types * 100, coverage, counter))
        for i, (g, c) in enumerate(top10, start=1):
            ngram_top_rows.append([vkey, n, i, g, c])
        print(f"{vlabel} {n}-gram: 総数 {tokens}, 異なり数 {types}, "
              f"ハパックス {hapax} ({hapax / types * 100:.1f} %), "
              f"上位 10 被覆率 {coverage:.1f} %")
        print_top(counter)
        print()

csv_ngram_top = os.path.join(DATA_DIR, "exp05_ngram_top10.csv")
with open(csv_ngram_top, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "n", "rank", "gram", "count"])
    w.writerows(ngram_top_rows)
print(f"CSV 保存: {csv_ngram_top}")

csv_ngram_stats = os.path.join(DATA_DIR, "exp05_ngram_stats.csv")
with open(csv_ngram_stats, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "n", "tokens", "types", "hapax",
                "hapax_percent", "top10_coverage_percent"])
    for vkey, vlabel, n, tokens, types, hapax, hpct, cov, _ in ngram_stats:
        w.writerow([vkey, n, tokens, types, hapax,
                    f"{hpct:.3f}", f"{cov:.3f}"])
print(f"CSV 保存: {csv_ngram_stats}")

# 図2: 2 行(除去なし/あり) x 3 列(n=1,2,3) の上位 10 棒グラフ
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
for row, (vkey, vlabel, seq) in enumerate(variants):
    for col, n in enumerate((1, 2, 3)):
        counter = next(st[8] for st in ngram_stats
                       if st[0] == vkey and st[2] == n)
        en_label = "without stop word removal" if vkey == "raw" \
            else "with stop word removal"
        color = "0.75" if vkey == "raw" else "0.45"
        hbar_top(axes[row][col], counter,
                 f"{n}-gram ({en_label})", color=color)
fig.tight_layout()
fig_ngrams = os.path.join(FIG_DIR, "exp05_ngram_bars.pdf")
fig.savefig(fig_ngrams)
plt.close(fig)
print(f"図 保存: {fig_ngrams}")
print()

# ---------------------------------------------------------------------
# [3] Zipf 則の検証(1-gram / 2-gram、除去なし)
# ---------------------------------------------------------------------
print("-" * 70)
print("[3] Zipf 則の検証(頻度-順位分布の log-log 最小二乗フィット)")
print("-" * 70)

bi_raw = Counter(raw_bigrams)
zipf_fit_rows = []
zipf_points = {}
for skey, counter in [("1gram", uni_raw), ("2gram", bi_raw)]:
    freqs = np.array(sorted(counter.values(), reverse=True),
                     dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    zipf_points[skey] = (ranks, freqs)
    for fit_key, mask in [
        ("all", np.ones_like(freqs, dtype=bool)),
        ("freq>=2", freqs >= 2),
    ]:
        x = np.log10(ranks[mask])
        y = np.log10(freqs[mask])
        slope, icpt = np.polyfit(x, y, 1)
        pred = slope * x + icpt
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        zipf_fit_rows.append([skey, fit_key, int(mask.sum()),
                              f"{slope:.4f}", f"{icpt:.4f}", f"{r2:.4f}"])
        print(f"{skey:<6s} fit={fit_key:<8s} 点数 {int(mask.sum()):>4d}  "
              f"傾き {slope:+.3f}  切片 {icpt:.3f}  R^2 {r2:.4f}")
print("(Zipf 則の理論値: 傾き -1 付近)")
print()

csv_zipf = os.path.join(DATA_DIR, "exp05_zipf.csv")
with open(csv_zipf, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "rank", "frequency"])
    for skey, (ranks, freqs) in zipf_points.items():
        for r, fr in zip(ranks, freqs):
            w.writerow([skey, int(r), int(fr)])
print(f"CSV 保存: {csv_zipf}")

csv_zipf_fit = os.path.join(DATA_DIR, "exp05_zipf_fit.csv")
with open(csv_zipf_fit, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "fit_range", "n_points", "slope", "intercept",
                "r_squared"])
    w.writerows(zipf_fit_rows)
print(f"CSV 保存: {csv_zipf_fit}")

# 図3: log-log プロット(白黒判別: 黒丸 / グレー四角 + 破線フィット)
fig, ax = plt.subplots(figsize=(6.6, 4.8))
styles = {"1gram": ("o", "black", "1-gram"),
          "2gram": ("s", "0.55", "2-gram")}
for skey, (ranks, freqs) in zipf_points.items():
    marker, color, label = styles[skey]
    slope = float([row[3] for row in zipf_fit_rows
                   if row[0] == skey and row[1] == "all"][0])
    icpt = float([row[4] for row in zipf_fit_rows
                  if row[0] == skey and row[1] == "all"][0])
    ax.loglog(ranks, freqs, marker, color=color, markersize=3.5,
              linestyle="none", markerfacecolor="none",
              label=f"{label} (fitted slope {slope:.2f})")
    ax.loglog(ranks, 10 ** icpt * ranks ** slope, "--", color=color,
              linewidth=1.2)
# 理論値: 傾き -1 の基準線(1-gram の最大頻度を通る)
r0 = zipf_points["1gram"][0]
f0 = zipf_points["1gram"][1][0]
ax.loglog(r0, f0 * r0 ** -1.0, ":", color="0.3",
          label="slope = -1 (Zipf reference)")
ax.set_xlabel("Rank")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
fig.tight_layout()
fig_zipf = os.path.join(FIG_DIR, "exp05_zipf.pdf")
fig.savefig(fig_zipf)
plt.close(fig)
print(f"図 保存: {fig_zipf}")
print()

# ---------------------------------------------------------------------
# [4] stop words 設計の感度
# ---------------------------------------------------------------------
print("-" * 70)
print("[4] stop words 設計の感度(2-gram は方式 A で構成)")
print("-" * 70)

target = "e g"
sensitivity = []
for skey, slabel, sw in STOP_WORD_SETS:
    filt = [w for w in words if w not in sw]
    counter = Counter(build_ngrams(filt, 2))
    rmap = rank_map(counter)
    eg_count = counter.get(target, 0)
    eg_rank = rmap.get(target, None)
    sensitivity.append((skey, slabel, sw, filt, counter, eg_count, eg_rank))
    print(f"集合 {skey}({slabel}): 語数 {len(sw)}, "
          f"除去後トークン数 {len(filt)}, 異なり 2-gram 数 {len(counter)}")
    print(f"  「{target}」の頻度 {eg_count}"
          + (f", 順位 {eg_rank}" if eg_rank else "(消滅)"))
    print_top(counter)
    print()

csv_sw = os.path.join(DATA_DIR, "exp05_stopwords_top10.csv")
with open(csv_sw, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set", "set_size", "tokens_after_filter",
                "eg_count", "eg_rank", "rank", "gram", "count"])
    for skey, slabel, sw, filt, counter, eg_count, eg_rank in sensitivity:
        for i, (g, c) in enumerate(ranked_items(counter)[:TOP_N], start=1):
            w.writerow([skey, len(sw), len(filt), eg_count,
                        eg_rank if eg_rank else "", i, g, c])
print(f"CSV 保存: {csv_sw}")

# 図4: 4 集合の上位 10(「e g」はハッチで強調)
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
en_titles = {
    "none": "no stop words (0 words)",
    "basic30": "basic 30 words",
    "adopted": f"adopted set ({len(STOP_WORDS_ADOPTED)} words)",
    "adopted+letters":
        f"adopted + single letters "
        f"({len(STOP_WORDS_ADOPTED | SINGLE_LETTERS)} words)",
}
for ax, (skey, slabel, sw, filt, counter, eg_count, eg_rank) in zip(
        axes.ravel(), sensitivity):
    hbar_top(ax, counter, en_titles[skey], hatch_grams={target},
             hatch="xx")
fig.tight_layout()
fig_sw = os.path.join(FIG_DIR, "exp05_stopwords.pdf")
fig.savefig(fig_sw)
plt.close(fig)
print(f"図 保存: {fig_sw}")
print()

print("=" * 70)
print("exp05 完了")
print("=" * 70)
