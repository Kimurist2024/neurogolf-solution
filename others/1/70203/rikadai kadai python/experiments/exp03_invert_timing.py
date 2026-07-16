# -*- coding: utf-8 -*-
"""exp03: 色反転処理の実装方式と実行時間のスケーリング計測

課題4 で採用した numpy ベクトル演算 (255 - frame) による色反転について、
実装方式ごとの実行時間と、画素数に対するスケーリングを計測する。

計測する実装方式(1 フレーム 1280x720x3 uint8):
  (a) numpy ベクトル演算 255 - frame            … 提出プログラムの採用手法
  (b) np.subtract(255, frame, out=buf) in-place … 出力バッファ再利用版
  (c) Python 3 重ループ                          … 160x90 のみ実測し、
                                                   画素数比でフルサイズへ換算
  (d) cv2.bitwise_not(frame)                     … 参考計測。課題の制約で
                                                   提出プログラムでは使用禁止

さらに、
  - 解像度 {160x90, 320x180, 640x360, 1280x720, 2560x1440} で
    (a)(d) の時間を計測し、log-log の傾きを最小二乗で推定する。
  - 動画全体(294 フレーム)の処理時間を
    読み込み / 反転 / リサイズ / 書き出し の 4 区間に分けて計測する。

実行方法:
  cd "/Users/kimura2003/Downloads/rikadai kadai python" && \
  DATASET_ROOT="$PWD" MPLBACKEND=Agg \
  python3 experiments/exp03_invert_timing.py > results/exp03.txt 2>&1
"""

import csv
import os
import platform
import statistics
import time
import timeit

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------
ROOT = os.environ.get("DATASET_ROOT", ".")
VIDEO_PATH = os.path.join(ROOT, "dataset", "videos", "Cat - 66004.mp4")
DATA_DIR = os.path.join(ROOT, "experiments", "data")
FIG_DIR = os.path.join(ROOT, "experiments", "figures")
OUT_VIDEO = os.path.join(ROOT, "dataset", "videos", "output",
                         "exp03_invert_0.3x.mp4")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)

# 計測条件
REPEAT = 7            # timeit.repeat の繰り返し回数(中央値を採用)
TARGET_TOTAL = 0.20   # 1 回の repeat の目標総時間 [s](number を自動決定)
MIN_NUMBER = 5        # 1 repeat あたりの最小実行回数(高速な処理の安定化)
MAX_NUMBER = 2000
WARMUP_CALLS = 3      # 計測前の空実行回数(キャッシュ・遅延初期化の除去)
LOOP_SIZE = (160, 90)  # Python 3 重ループの実測サイズ (width, height)
SCALES = [(160, 90), (320, 180), (640, 360), (1280, 720), (2560, 1440)]
RESIZE_SCALE = 0.3    # 課題4 と同じ縮小率

# 白黒印刷で判別できる描画設定
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "0.7",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------
# 色反転の各実装
# ---------------------------------------------------------------------
def invert_vec(frame):
    """(a) numpy ベクトル演算(提出プログラムの採用手法)。"""
    return 255 - frame


def invert_inplace(frame, buf):
    """(b) 出力バッファを再利用する in-place 版。"""
    np.subtract(255, frame, out=buf)
    return buf


def invert_loop(frame):
    """(c) Python 3 重ループ(画素ごとに 255 - p を計算)。"""
    h, w, c = frame.shape
    out = np.empty_like(frame)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                out[y, x, ch] = 255 - frame[y, x, ch]
    return out


def invert_cv2(frame):
    """(d) cv2.bitwise_not(参考計測。提出プログラムでは使用禁止)。"""
    return cv2.bitwise_not(frame)


# ---------------------------------------------------------------------
# 計測ユーティリティ
# ---------------------------------------------------------------------
def measure(fn, target_total=TARGET_TOTAL, repeat=REPEAT, min_number=MIN_NUMBER):
    """fn の 1 回あたり実行時間を timeit.repeat で計測する。

    計測前に WARMUP_CALLS 回の空実行を行い(キャッシュ・遅延初期化の影響を
    除去)、その後の 3 回の単発計測の最小値から、1 repeat の総時間が
    target_total 程度になるよう number を自動決定する(min_number 以上)。
    repeat 回の per-call 時間の中央値・最小値・最大値を返す。
    """
    for _ in range(WARMUP_CALLS):
        fn()
    timer = timeit.Timer(fn)
    cal = min(timer.timeit(number=1) for _ in range(3))  # キャリブレーション
    number = int(max(min_number,
                     min(MAX_NUMBER, round(target_total / max(cal, 1e-9)))))
    raw = timer.repeat(repeat=repeat, number=number)
    per_call = sorted(t / number for t in raw)
    return {
        "median_s": statistics.median(per_call),
        "min_s": per_call[0],
        "max_s": per_call[-1],
        "number": number,
        "repeat": repeat,
    }


def fmt_ms(sec):
    return f"{sec * 1e3:.4f}"


# ---------------------------------------------------------------------
# 準備: フレームの取得
# ---------------------------------------------------------------------
print("=" * 70)
print("exp03: 色反転処理の実装方式と実行時間のスケーリング計測")
print("=" * 70)
print(f"platform : {platform.platform()}")
print(f"machine  : {platform.machine()}")
print(f"python   : {platform.python_version()}")
print(f"numpy    : {np.__version__}")
print(f"opencv   : {cv2.__version__}")
print(f"video    : {VIDEO_PATH}")
print(f"timeit   : repeat={REPEAT}, 1 repeat の目標総時間 {TARGET_TOTAL} s "
      f"(number は自動決定), 統計量は中央値")
print()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"動画を開けません: {VIDEO_PATH}")
ret, frame_full = cap.read()
if not ret:
    raise IOError("先頭フレームを読み込めません")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

h, w, c = frame_full.shape
n_pixels_full = h * w
print(f"先頭フレーム: {w}x{h}x{c} dtype={frame_full.dtype}, "
      f"総フレーム数 {n_frames}, fps {fps:.2f}")
print()

# ---------------------------------------------------------------------
# 正しさの確認(4 方式が同一の結果を返すこと)
# ---------------------------------------------------------------------
print("-" * 70)
print("[0] 各実装の同値性確認")
print("-" * 70)
ref = invert_vec(frame_full)
buf = np.empty_like(frame_full)
same_b = np.array_equal(ref, invert_inplace(frame_full, buf))
same_d = np.array_equal(ref, invert_cv2(frame_full))
frame_small = cv2.resize(frame_full, LOOP_SIZE)
same_c = np.array_equal(invert_vec(frame_small), invert_loop(frame_small))
print(f"(b) in-place  == (a) ベクトル演算 : {same_b}")
print(f"(c) 3重ループ == (a) ベクトル演算 : {same_c}  ({LOOP_SIZE[0]}x{LOOP_SIZE[1]} で確認)")
print(f"(d) bitwise_not == (a) ベクトル演算 : {same_d}")
assert same_b and same_c and same_d
print()

# ---------------------------------------------------------------------
# [1] 1 フレーム(1280x720)に対する 4 方式の実行時間
# ---------------------------------------------------------------------
print("-" * 70)
print("[1] 1 フレーム (1280x720x3 uint8) に対する 4 方式の実行時間")
print("-" * 70)

results_methods = []

r = measure(lambda: invert_vec(frame_full))
results_methods.append(("a_vec", "255 - frame (採用手法)", w, h, r, "measured"))

r = measure(lambda: invert_inplace(frame_full, buf))
results_methods.append(("b_inplace", "np.subtract(out=) in-place", w, h, r,
                        "measured"))

# (c) Python 3 重ループは 160x90 で実測し、画素数比でフルサイズへ換算
r_loop = measure(lambda: invert_loop(frame_small), target_total=0.0,
                 min_number=1)
sw, sh = LOOP_SIZE
scale_factor = (w * h) / (sw * sh)
results_methods.append(("c_loop_small", "Python 3重ループ (160x90 実測)",
                        sw, sh, r_loop, "measured"))
r_loop_est = {
    "median_s": r_loop["median_s"] * scale_factor,
    "min_s": r_loop["min_s"] * scale_factor,
    "max_s": r_loop["max_s"] * scale_factor,
    "number": r_loop["number"],
    "repeat": r_loop["repeat"],
}
results_methods.append(("c_loop_est", "Python 3重ループ (1280x720 換算値)",
                        w, h, r_loop_est, f"extrapolated x{scale_factor:.0f}"))

r = measure(lambda: invert_cv2(frame_full))
results_methods.append(("d_cv2", "cv2.bitwise_not (参考・提出では使用禁止)",
                        w, h, r, "measured"))

print(f"{'方式':<40s} {'サイズ':>10s} {'中央値[ms]':>12s} "
      f"{'最小[ms]':>10s} {'最大[ms]':>10s} {'number':>7s} {'備考':>16s}")
for key, label, mw, mh, r, note in results_methods:
    print(f"{label:<40s} {mw:>5d}x{mh:<4d} {fmt_ms(r['median_s']):>12s} "
          f"{fmt_ms(r['min_s']):>10s} {fmt_ms(r['max_s']):>10s} "
          f"{r['number']:>7d} {note:>16s}")
print()

med = {key: r["median_s"] for key, _, _, _, r, _ in results_methods}
ratio_loop_vec = med["c_loop_est"] / med["a_vec"]
ratio_vec_cv2 = med["a_vec"] / med["d_cv2"]
ratio_vec_inplace = med["a_vec"] / med["b_inplace"]
print(f"倍率: 3重ループ(換算) / ベクトル演算(a)   = {ratio_loop_vec:.1f} 倍")
print(f"倍率: ベクトル演算(a) / in-place(b)       = {ratio_vec_inplace:.2f} 倍")
print(f"倍率: ベクトル演算(a) / bitwise_not(d)    = {ratio_vec_cv2:.2f} 倍")
print()

csv_methods = os.path.join(DATA_DIR, "exp03_methods.csv")
with open(csv_methods, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["key", "label", "width", "height", "pixels",
                   "median_ms", "min_ms", "max_ms", "number", "repeat",
                   "note"])
    for key, label, mw, mh, r, note in results_methods:
        wcsv.writerow([key, label, mw, mh, mw * mh,
                       f"{r['median_s'] * 1e3:.6f}",
                       f"{r['min_s'] * 1e3:.6f}",
                       f"{r['max_s'] * 1e3:.6f}",
                       r["number"], r["repeat"], note])
print(f"CSV 保存: {csv_methods}")

# 図1: 方式別実行時間(横棒、対数軸、白黒判別: グレー濃淡 + ハッチ)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bar_items = [
    ("(a) 255 - frame", med["a_vec"] * 1e3, "0.35", ""),
    ("(b) np.subtract(out=)", med["b_inplace"] * 1e3, "0.55", ""),
    ("(c) Python triple loop\n(extrapolated)", med["c_loop_est"] * 1e3,
     "0.85", "//"),
    ("(d) cv2.bitwise_not\n(reference)", med["d_cv2"] * 1e3, "0.7", "xx"),
]
labels = [b[0] for b in bar_items]
values = [b[1] for b in bar_items]
colors = [b[2] for b in bar_items]
hatches = [b[3] for b in bar_items]
bars = ax.barh(range(len(bar_items)), values, color=colors,
               edgecolor="black", height=0.6)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
for i, v in enumerate(values):
    ax.text(v * 1.15, i, f"{v:.3g} ms", va="center", fontsize=10)
ax.set_yticks(range(len(bar_items)))
ax.set_yticklabels(labels)
ax.set_xscale("log")
ax.set_xlim(right=max(values) * 8)
ax.set_xlabel("Time per frame [ms] (median, 1280x720x3 uint8)")
ax.invert_yaxis()
fig.tight_layout()
fig_methods = os.path.join(FIG_DIR, "exp03_methods.pdf")
fig.savefig(fig_methods)
plt.close(fig)
print(f"図 保存: {fig_methods}")
print()

# ---------------------------------------------------------------------
# [2] 解像度スケーリング: (a) と (d) の時間 vs 画素数
# ---------------------------------------------------------------------
print("-" * 70)
print("[2] 解像度スケーリング ((a) ベクトル演算 と (d) bitwise_not)")
print("-" * 70)

scaling_rows = []
print(f"{'解像度':>12s} {'画素数':>10s} {'(a) 中央値[ms]':>14s} "
      f"{'(d) 中央値[ms]':>14s}")
for sw_, sh_ in SCALES:
    if (sw_, sh_) == (w, h):
        fr = frame_full
    else:
        interp = cv2.INTER_AREA if sw_ < w else cv2.INTER_LINEAR
        fr = cv2.resize(frame_full, (sw_, sh_), interpolation=interp)
    np_ = sw_ * sh_
    ra = measure(lambda: invert_vec(fr))
    rd = measure(lambda: invert_cv2(fr))
    scaling_rows.append((sw_, sh_, np_, ra, rd))
    print(f"{sw_:>6d}x{sh_:<5d} {np_:>10d} {fmt_ms(ra['median_s']):>14s} "
          f"{fmt_ms(rd['median_s']):>14s}")

# log-log の傾きを最小二乗で推定
px = np.array([row[2] for row in scaling_rows], dtype=np.float64)
ta = np.array([row[3]["median_s"] for row in scaling_rows])
td = np.array([row[4]["median_s"] for row in scaling_rows])
slope_a, icpt_a = np.polyfit(np.log10(px), np.log10(ta), 1)
slope_d, icpt_d = np.polyfit(np.log10(px), np.log10(td), 1)
print()
print(f"log-log 傾き (a) 255 - frame     : {slope_a:.3f}")
print(f"log-log 傾き (d) cv2.bitwise_not : {slope_d:.3f}")
print()

csv_scaling = os.path.join(DATA_DIR, "exp03_scaling.csv")
with open(csv_scaling, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["width", "height", "pixels",
                   "a_vec_median_ms", "a_vec_min_ms", "a_vec_number",
                   "d_cv2_median_ms", "d_cv2_min_ms", "d_cv2_number"])
    for sw_, sh_, np_, ra, rd in scaling_rows:
        wcsv.writerow([sw_, sh_, np_,
                       f"{ra['median_s'] * 1e3:.6f}",
                       f"{ra['min_s'] * 1e3:.6f}", ra["number"],
                       f"{rd['median_s'] * 1e3:.6f}",
                       f"{rd['min_s'] * 1e3:.6f}", rd["number"]])
print(f"CSV 保存: {csv_scaling}")

# 図2: log-log プロット(白黒判別: 実線+丸、破線+四角、点線=傾き1 基準線)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.loglog(px, ta * 1e3, "o-", color="black",
          label=f"(a) 255 - frame  (slope {slope_a:.2f})")
ax.loglog(px, td * 1e3, "s--", color="0.45",
          label=f"(d) cv2.bitwise_not  (slope {slope_d:.2f})")
# 傾き 1 の基準線((a) の最終点を通る)
ref_y = ta[-1] * 1e3 * (px / px[-1])
ax.loglog(px, ref_y, ":", color="0.6", label="slope = 1 (reference)")
ax.set_xlabel("Number of pixels per frame")
ax.set_ylabel("Time per frame [ms] (median)")
ax.legend(fontsize=9)
fig.tight_layout()
fig_scaling = os.path.join(FIG_DIR, "exp03_scaling.pdf")
fig.savefig(fig_scaling)
plt.close(fig)
print(f"図 保存: {fig_scaling}")
print()

# ---------------------------------------------------------------------
# [3] 動画全体(294 フレーム)の処理時間内訳
# ---------------------------------------------------------------------
print("-" * 70)
print("[3] 動画全体の処理時間内訳 (読み込み / 反転 / リサイズ / 書き出し)")
print("-" * 70)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"動画を開けません: {VIDEO_PATH}")
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vfps = cap.get(cv2.CAP_PROP_FPS)
new_size = (int(vw * RESIZE_SCALE), int(vh * RESIZE_SCALE))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, vfps, new_size)

t_read = t_invert = t_resize = t_write = 0.0
frame_count = 0
t_total0 = time.perf_counter()
while True:
    t0 = time.perf_counter()
    ret, fr = cap.read()
    t1 = time.perf_counter()
    t_read += t1 - t0
    if not ret:
        break
    inv = 255 - fr
    t2 = time.perf_counter()
    t_invert += t2 - t1
    rs = cv2.resize(inv, new_size)
    t3 = time.perf_counter()
    t_resize += t3 - t2
    writer.write(rs)
    t4 = time.perf_counter()
    t_write += t4 - t3
    frame_count += 1
cap.release()
writer.release()
t_total = time.perf_counter() - t_total0

stages = [
    ("read", "読み込み (cap.read)", t_read),
    ("invert", "反転 (255 - frame)", t_invert),
    ("resize", "リサイズ (cv2.resize)", t_resize),
    ("write", "書き出し (writer.write)", t_write),
]
t_stages = sum(s[2] for s in stages)
print(f"入力: {vw}x{vh}, {frame_count} フレーム, fps {vfps:.2f}")
print(f"出力: {new_size[0]}x{new_size[1]}, {OUT_VIDEO}")
print()
print(f"{'区間':<28s} {'合計[s]':>10s} {'1フレーム平均[ms]':>18s} {'割合[%]':>9s}")
for key, label, t in stages:
    print(f"{label:<28s} {t:>10.4f} {t / frame_count * 1e3:>18.3f} "
          f"{t / t_stages * 100:>9.1f}")
print(f"{'4区間合計':<28s} {t_stages:>10.4f} "
      f"{t_stages / frame_count * 1e3:>18.3f} {100.0:>9.1f}")
print(f"{'全体 (perf_counter)':<28s} {t_total:>10.4f} "
      f"{t_total / frame_count * 1e3:>18.3f}")
print()

csv_pipeline = os.path.join(DATA_DIR, "exp03_pipeline.csv")
with open(csv_pipeline, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["stage", "label", "total_s", "per_frame_ms",
                   "share_percent", "frames"])
    for key, label, t in stages:
        wcsv.writerow([key, label, f"{t:.6f}",
                       f"{t / frame_count * 1e3:.6f}",
                       f"{t / t_stages * 100:.3f}", frame_count])
    wcsv.writerow(["total_stages", "4区間合計", f"{t_stages:.6f}",
                   f"{t_stages / frame_count * 1e3:.6f}", "100.000",
                   frame_count])
    wcsv.writerow(["total_wall", "全体", f"{t_total:.6f}",
                   f"{t_total / frame_count * 1e3:.6f}", "", frame_count])
print(f"CSV 保存: {csv_pipeline}")

# 図3: 内訳の積み上げ横棒(白黒判別: グレー濃淡 + ハッチ)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
stage_style = [
    ("read (decode)", t_read, "0.85", ""),
    ("invert (255 - frame)", t_invert, "0.35", ""),
    ("resize", t_resize, "0.6", "//"),
    ("write (encode)", t_write, "0.95", "xx"),
]
left = 0.0
for label, t, color, hatch in stage_style:
    ax.barh([0], [t], left=left, color=color, edgecolor="black",
            hatch=hatch, height=0.5, label=label)
    if t / t_stages > 0.04:
        ax.text(left + t / 2, 0, f"{t:.2f} s\n({t / t_stages * 100:.1f}%)",
                ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.5))
    left += t
ax.set_yticks([])
ax.set_xlabel(f"Cumulative time for {frame_count} frames [s]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4,
          fontsize=9, frameon=False)
ax.set_xlim(0, t_stages * 1.02)
fig.tight_layout()
fig_pipeline = os.path.join(FIG_DIR, "exp03_pipeline.pdf")
fig.savefig(fig_pipeline, bbox_inches="tight")
plt.close(fig)
print(f"図 保存: {fig_pipeline}")
print()

print("=" * 70)
print("exp03 完了")
print("=" * 70)
