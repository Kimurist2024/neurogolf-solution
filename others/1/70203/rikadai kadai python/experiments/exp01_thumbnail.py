# =====================================================================
# exp01_thumbnail.py
# 課題 2 のサムネイル(代表フレーム)選択手法の定量比較実験
#
# 比較する 4 手法:
#   (a) 中央フレーム法        : フレーム番号 N//2 を選ぶ
#   (b) 平均ヒストグラム法    : 採用手法。グレースケール 64 bin 正規化
#                               ヒストグラムの平均への L2 距離最小
#                               (src/6324034_final_exam.py の
#                                task2_thumbnail_frame と同一ロジック,
#                                sample_step=5)
#   (c) 鮮明度法              : ラプラシアン分散が最大のフレーム
#   (d) カラフルネス法        : Hasler-Suesstrunk 指標が最大のフレーム
#
# 出力:
#   experiments/data/exp01_traces.csv   : 全フレームの各評価値の推移
#   experiments/data/exp01_summary.csv  : 各手法の選択結果・計算時間
#   experiments/figures/exp01_metric_traces.pdf   : 評価値推移 (2x2)
#   experiments/figures/exp01_normalized.pdf      : 正規化評価値の重ね描き
#   experiments/figures/exp01_selected_frames.pdf : 選択フレーム画像 (2x2)
#
# 実行方法:
#   cd "/Users/kimura2003/Downloads/rikadai kadai python"
#   DATASET_ROOT="$PWD" MPLBACKEND=Agg python3 experiments/exp01_thumbnail.py
# =====================================================================
import csv
import os
import time

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATASET_ROOT = os.environ.get("DATASET_ROOT", ".")
VIDEO_PATH = os.path.join(DATASET_ROOT, "dataset", "videos",
                          "People - 84973.mp4")
DATA_DIR = os.path.join(DATASET_ROOT, "experiments", "data")
FIG_DIR = os.path.join(DATASET_ROOT, "experiments", "figures")

HIST_BINS = 64      # 採用手法と同じビン数
SAMPLE_STEP = 5     # 採用手法と同じサンプリング間隔


# ---------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------
def compute_gray_histogram(frame, bins=HIST_BINS):
    """グレースケール正規化ヒストグラム(採用手法と同一ロジック)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def laplacian_variance(frame):
    """鮮明度指標: グレースケール画像のラプラシアンの分散。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def colorfulness_hs(frame):
    """Hasler-Suesstrunk のカラフルネス指標 M。

    rg = R - G, yb = (R + G)/2 - B として
    M = sqrt(sigma_rg^2 + sigma_yb^2) + 0.3 * sqrt(mu_rg^2 + mu_yb^2)
    """
    b = frame[:, :, 0].astype(np.float64)
    g = frame[:, :, 1].astype(np.float64)
    r = frame[:, :, 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma = np.sqrt(rg.var() + yb.var())
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(sigma + 0.3 * mu)


def percentile_rank(values, v):
    """values の中で v 以下の値が占める割合(百分率)。"""
    values = np.asarray(values, dtype=np.float64)
    return float(100.0 * np.mean(values <= v))


def spearman_corr(x, y):
    """Spearman 順位相関係数(順位化してから Pearson)。"""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("exp01: サムネイル選択手法の定量比較")
    print("=" * 70)
    print(f"OpenCV {cv2.__version__} / NumPy {np.__version__} / "
          f"matplotlib {matplotlib.__version__}")
    print(f"video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {VIDEO_PATH}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"size: {width}x{height}, fps: {fps}, frames(prop): {n_frames_prop}")

    # --- 1 パスで全フレームの各評価値と計算時間を収集 ---------------
    hists = []
    lap_vars = []
    colorfulness = []
    t_hist = []
    t_lap = []
    t_col = []
    t_decode = 0.0

    index = 0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_decode += time.perf_counter() - t0
        if not ret:
            break

        t0 = time.perf_counter()
        hists.append(compute_gray_histogram(frame))
        t_hist.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lap_vars.append(laplacian_variance(frame))
        t_lap.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        colorfulness.append(colorfulness_hs(frame))
        t_col.append(time.perf_counter() - t0)

        index += 1
    cap.release()

    n_frames = index
    print(f"frames(read): {n_frames}")
    print(f"decode time total: {t_decode:.4f} s")

    hists = np.array(hists)
    lap_vars = np.array(lap_vars)
    colorfulness = np.array(colorfulness)
    frames_axis = np.arange(n_frames)

    # --- (a) 中央フレーム法 ------------------------------------------
    t0 = time.perf_counter()
    center_index = n_frames // 2
    time_center = time.perf_counter() - t0
    center_dist = np.abs(frames_axis - center_index).astype(np.float64)

    # --- (b) 平均ヒストグラム法(採用手法, sample_step=5) -----------
    t0 = time.perf_counter()
    sampled_idx = np.arange(0, n_frames, SAMPLE_STEP)
    sampled_hists = hists[sampled_idx]
    mean_hist_s5 = sampled_hists.mean(axis=0)
    dists_s5 = np.linalg.norm(sampled_hists - mean_hist_s5, axis=1)
    hist_index_s5 = int(sampled_idx[int(np.argmin(dists_s5))])
    time_hist_select = time.perf_counter() - t0
    time_hist = float(np.sum(np.array(t_hist)[sampled_idx])) + time_hist_select

    # 参考: sample_step=1(全フレーム)での平均ヒストグラム法
    mean_hist_s1 = hists.mean(axis=0)
    dists_s1 = np.linalg.norm(hists - mean_hist_s1, axis=1)
    hist_index_s1 = int(np.argmin(dists_s1))

    # --- (c) 鮮明度法(ラプラシアン分散最大) ------------------------
    t0 = time.perf_counter()
    lap_index = int(np.argmax(lap_vars))
    time_lap = float(np.sum(t_lap)) + (time.perf_counter() - t0)

    # --- (d) カラフルネス法(Hasler-Suesstrunk 指標最大) --------------
    t0 = time.perf_counter()
    col_index = int(np.argmax(colorfulness))
    time_col = float(np.sum(t_col)) + (time.perf_counter() - t0)

    methods = [
        ("center", "中央フレーム法", center_index, time_center),
        ("hist", "平均ヒストグラム法(採用)", hist_index_s5, time_hist),
        ("laplacian", "鮮明度法", lap_index, time_lap),
        ("colorfulness", "カラフルネス法", col_index, time_col),
    ]

    print()
    print("--- 各手法の選択フレームと計算時間 ---")
    for key, name, sel, t in methods:
        print(f"{key:14s} {name:22s} frame={sel:4d}  time={t:.6f} s")
    print(f"(参考) 平均ヒストグラム法 sample_step=1: frame={hist_index_s1}")
    print(f"(参考) sample_step=5 と step=1 の選択差: "
          f"{abs(hist_index_s1 - hist_index_s5)} フレーム")

    # --- 各指標の基本統計 ---------------------------------------------
    print()
    print("--- 各評価値の基本統計(全フレーム) ---")
    stats_rows = [
        ("hist_l2_dist(step=1)", dists_s1),
        ("laplacian_var", lap_vars),
        ("colorfulness", colorfulness),
    ]
    for name, arr in stats_rows:
        print(f"{name:22s} min={arr.min():12.6g} max={arr.max():12.6g} "
              f"mean={arr.mean():12.6g} std={arr.std():12.6g}")

    # --- 指標間の相互評価(クロス評価表) -----------------------------
    print()
    print("--- 各手法の選択フレームにおける他指標の値と百分位 ---")
    print("(百分位はその値以下のフレームが占める割合 [%]。"
          "hist_l2 は小さいほど良い / lap_var, colorfulness は大きいほど良い)")
    header = (f"{'method':14s} {'frame':>5s} "
              f"{'hist_l2':>10s} {'pct':>6s} "
              f"{'lap_var':>10s} {'pct':>6s} "
              f"{'colorful':>10s} {'pct':>6s}")
    print(header)
    cross_rows = []
    for key, name, sel, t in methods:
        h = dists_s1[sel]
        l = lap_vars[sel]
        c = colorfulness[sel]
        ph = percentile_rank(dists_s1, h)
        pl = percentile_rank(lap_vars, l)
        pc = percentile_rank(colorfulness, c)
        print(f"{key:14s} {sel:5d} "
              f"{h:10.6f} {ph:6.1f} "
              f"{l:10.2f} {pl:6.1f} "
              f"{c:10.2f} {pc:6.1f}")
        cross_rows.append([key, sel, h, ph, l, pl, c, pc, t])

    # --- 指標間の相関 ---------------------------------------------------
    print()
    print("--- 指標間の相関係数(全フレーム, Pearson / Spearman) ---")
    pairs = [
        ("hist_l2_dist vs laplacian_var", dists_s1, lap_vars),
        ("hist_l2_dist vs colorfulness", dists_s1, colorfulness),
        ("laplacian_var vs colorfulness", lap_vars, colorfulness),
    ]
    for name, x, y in pairs:
        pear = float(np.corrcoef(x, y)[0, 1])
        spear = spearman_corr(x, y)
        print(f"{name:32s} Pearson={pear:+.4f}  Spearman={spear:+.4f}")

    # --- CSV 出力 -------------------------------------------------------
    traces_csv = os.path.join(DATA_DIR, "exp01_traces.csv")
    with open(traces_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "center_dist", "hist_l2_dist_step1",
                    "laplacian_var", "colorfulness"])
        for i in range(n_frames):
            w.writerow([i, center_dist[i], f"{dists_s1[i]:.8f}",
                        f"{lap_vars[i]:.6f}", f"{colorfulness[i]:.6f}"])
    print()
    print(f"saved: {traces_csv}")

    summary_csv = os.path.join(DATA_DIR, "exp01_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "selected_frame",
                    "hist_l2_dist", "hist_l2_percentile",
                    "laplacian_var", "laplacian_percentile",
                    "colorfulness", "colorfulness_percentile",
                    "time_sec"])
        for row in cross_rows:
            w.writerow([row[0], row[1],
                        f"{row[2]:.8f}", f"{row[3]:.1f}",
                        f"{row[4]:.4f}", f"{row[5]:.1f}",
                        f"{row[6]:.4f}", f"{row[7]:.1f}",
                        f"{row[8]:.6f}"])
    print(f"saved: {summary_csv}")

    # --- 図 1: 評価値の推移(2x2 サブプロット) -----------------------
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))

    panels = [
        (axes[0, 0], center_dist, center_index,
         "(a) Center-frame: $|i - 120|$",
         "distance from center (frames)"),
        (axes[0, 1], dists_s1, hist_index_s5,
         "(b) Mean-histogram (adopted): L2 distance",
         "L2 distance to mean histogram (-)"),
        (axes[1, 0], lap_vars, lap_index,
         "(c) Sharpness: Laplacian variance",
         "Laplacian variance ((gray level)$^2$)"),
        (axes[1, 1], colorfulness, col_index,
         "(d) Colorfulness: Hasler-Suesstrunk $M$",
         "colorfulness $M$ (-)"),
    ]
    for ax, arr, sel, title, ylabel in panels:
        ax.plot(frames_axis, arr, color="black", linewidth=0.9)
        ax.axvline(sel, color="black", linestyle=(0, (5, 2)), linewidth=1.2)
        ymin, ymax = ax.get_ylim()
        ax.text(sel + 3, ymin + 0.88 * (ymax - ymin),
                f"selected: {sel}", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frame index (-)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.85", linewidth=0.5)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "exp01_metric_traces.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"saved: {fig_path}")

    # --- 図 2: 正規化評価値の重ね描き ---------------------------------
    def minmax(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return (arr - arr.min()) / (arr.max() - arr.min())

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    styles = [
        (minmax(center_dist), "center-frame $|i-120|$",
         "-", "o", "white"),
        (minmax(dists_s1), "mean-histogram L2 dist. (adopted)",
         "--", "s", "white"),
        (minmax(lap_vars), "Laplacian variance",
         "-.", "^", "black"),
        (minmax(colorfulness), "colorfulness $M$",
         ":", "v", "black"),
    ]
    for arr, label, ls, marker, mfc in styles:
        ax.plot(frames_axis, arr, linestyle=ls, marker=marker,
                markevery=20, markersize=4, markerfacecolor=mfc,
                color="black", linewidth=0.9, label=label)
    ax.set_xlabel("frame index (-)")
    ax.set_ylabel("min-max normalized metric value (-)")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=1.0)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "exp01_normalized.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"saved: {fig_path}")

    # --- 図 3: 各手法の選択フレーム画像(2x2) ------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0))
    labels = ["(a) Center-frame", "(b) Mean-histogram (adopted)",
              "(c) Sharpness (Laplacian)", "(d) Colorfulness (H-S)"]
    for ax, (key, name, sel, t), label in zip(axes.flat, methods, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sel)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"フレーム {sel} を読み込めません")
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}: frame {sel}", fontsize=9)
        ax.axis("off")
    cap.release()
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "exp01_selected_frames.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"saved: {fig_path}")

    print()
    print("exp01 done.")


if __name__ == "__main__":
    main()
