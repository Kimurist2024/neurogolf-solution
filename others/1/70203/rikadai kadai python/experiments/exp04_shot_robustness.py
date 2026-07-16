# -*- coding: utf-8 -*-
"""exp04: 変わり目検出の頑健性の定量評価

課題7 で採用したグレースケール隣接フレーム差分の総和 D(t) による
変わり目検出について、以下の 3 点を定量評価する。
対象は Detection.mp4(1920x1080、30 fps、322 フレーム)、
正解の変わり目はフレーム {78, 214, 257}(±1 フレーム許容)である。

(1) ノイズ頑健性:
    各フレームにガウスノイズ(標準偏差 sigma ∈ {0, 5, 10, 20, 40}、
    画素値スケール)を加えて D(t) を再計算し、
      方式(1) 平均 + 3σ 閾値
      方式(3) 移動平均基準(window=15, k=4)
      参考   ヒストグラム差分(L1)+ 平均 + 3σ 閾値
    の precision / recall / F1 を計測する。

(2) パラメータ感度:
    方式(1) の係数 k を 0.5〜6.0 まで 0.25 刻みで振り、
    ノイズなしと sigma=20 の 2 条件で検出数・precision・recall・F1 の
    曲線を求め、F1=1 となる「安全な k の範囲」を数値で示す。

(3) 比較手法:
    グレースケールヒストグラム差分法(ビン 64、隣接フレームの
    正規化ヒストグラムの L1 距離)を追加実装し、同じ 3 境界が
    検出できるかを確認する。D(t) 系列との対比図も作成する。

実行方法:
  cd "/Users/kimura2003/Downloads/rikadai kadai python" && \
  DATASET_ROOT="$PWD" MPLBACKEND=Agg \
  python3 experiments/exp04_shot_robustness.py > results/exp04.txt 2>&1
"""

import csv
import math
import os
import platform
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------
ROOT = os.environ.get("DATASET_ROOT", ".")
VIDEO_PATH = os.path.join(ROOT, "dataset", "videos", "Detection.mp4")
DATA_DIR = os.path.join(ROOT, "experiments", "data")
FIG_DIR = os.path.join(ROOT, "experiments", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 実験条件
# ---------------------------------------------------------------------
GROUND_TRUTH = (78, 214, 257)   # 正解の変わり目フレーム番号
TOLERANCE = 1                   # 照合の許容ずれ [フレーム]
NOISE_SIGMAS = (0, 5, 10, 20, 40)   # ガウスノイズ標準偏差(画素値)
K_VALUES = np.arange(0.5, 6.0 + 1e-9, 0.25)  # 方式(1) の k の走査範囲
K_SENS_SIGMAS = (0, 20)         # k 感度実験のノイズ条件
MEANSTD_K = 3.0                 # 方式(1) の既定係数(提出プログラムと同一)
MA_WINDOW = 15                  # 方式(3) の移動平均窓幅(同上)
MA_K = 4.0                      # 方式(3) の係数(同上)
HIST_BINS = 64                  # ヒストグラム差分法のビン数
SEED_BASE = 20260702            # 乱数シード(sigma を加えて系列ごとに固定)

# 白黒印刷で判別できる描画設定
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "0.7",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------
# 系列の計算
# ---------------------------------------------------------------------
def compute_series(video_path, sigma, seed):
    """動画を先頭から走査し、フレームごとにガウスノイズを加えたうえで
    2 種類の隣接フレーム間距離系列を計算する。

    返り値:
      diffs    -- D(t): グレースケール絶対差の総和(提出プログラムと同一の定義)
      hist_l1  -- H(t): 正規化 64 ビンヒストグラムの L1 距離
    いずれも diffs[i] が「フレーム i と i+1 の間の距離」である。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    rng = np.random.default_rng(seed)

    diffs = []
    hist_l1 = []
    prev_gray = None
    prev_hist = None
    shift = int(round(math.log2(256 // HIST_BINS)))  # 64 ビン → 2 bit シフト
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if sigma > 0:
            noise = sigma * rng.standard_normal(gray.shape, dtype=np.float32)
            noisy = np.clip(np.rint(gray.astype(np.float32) + noise),
                            0, 255).astype(np.uint8)
        else:
            noisy = gray
        gray_i32 = noisy.astype(np.int32)
        hist = np.bincount((noisy >> shift).ravel(),
                           minlength=HIST_BINS).astype(np.float64)
        hist /= hist.sum()
        if prev_gray is not None:
            diffs.append(int(np.abs(gray_i32 - prev_gray).sum()))
            hist_l1.append(float(np.abs(hist - prev_hist).sum()))
        prev_gray = gray_i32
        prev_hist = hist
    cap.release()
    return (np.array(diffs, dtype=np.float64),
            np.array(hist_l1, dtype=np.float64))


# ---------------------------------------------------------------------
# 検出方式(提出プログラム src/6324034_final_exam.py と同一ロジック)
# ---------------------------------------------------------------------
def detect_by_mean_std(diffs, k=3.0):
    """方式(1): 閾値 = 平均 + k×標準偏差 で変わり目を検出する。"""
    threshold = diffs.mean() + k * diffs.std()
    frames = np.where(diffs > threshold)[0] + 1
    return threshold, frames


def detect_by_top_peaks(diffs, n_peaks=3):
    """方式(2): 変化量の上位 n_peaks 個のフレームを変わり目とする。"""
    order = np.argsort(diffs)[::-1][:n_peaks]
    return np.sort(order + 1)


def detect_by_moving_average(diffs, window=15, k=4.0):
    """方式(3): 移動平均基準線の k 倍を超えたフレームを検出する。"""
    kernel = np.ones(window) / window
    baseline = np.convolve(diffs, kernel, mode="same")
    frames = np.where(diffs > k * baseline)[0] + 1
    return frames


# ---------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------
def match_detections(detected, truth, tol=TOLERANCE):
    """検出フレームと正解フレームを 1 対 1 で照合する(貪欲最近傍)。

    正解 g に対し |d - g| <= tol の未使用検出のうち最近傍を対応付ける。
    返り値: (tp, fp, fn)
    """
    detected = sorted(int(d) for d in detected)
    used = [False] * len(detected)
    tp = 0
    for g in truth:
        best_j = -1
        best_dist = tol + 1
        for j, d in enumerate(detected):
            if used[j]:
                continue
            dist = abs(d - g)
            if dist <= tol and dist < best_dist:
                best_j = j
                best_dist = dist
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = used.count(False)
    fn = len(truth) - tp
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    """precision / recall / F1 を返す(分母 0 のときは 0 とする)。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def evaluate(detected):
    """検出フレーム列から (tp, fp, fn, precision, recall, f1) を返す。"""
    tp, fp, fn = match_detections(detected, GROUND_TRUTH)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return tp, fp, fn, p, r, f1


def fmt_frames(frames, limit=20):
    """検出フレーム列を表示用文字列にする(多すぎる場合は省略)。"""
    frames = [int(f) for f in frames]
    if len(frames) <= limit:
        return str(frames)
    head = ", ".join(str(f) for f in frames[:limit])
    return f"[{head}, ... 計{len(frames)}個]"


def separation_stats(series):
    """上位 3 ピークと残りの分離度、および平均+kσ 閾値が
    F1=1 を与える k の理論範囲 (k_low, k_high) を返す。"""
    order = np.argsort(series)[::-1]
    top3_min = float(series[order[:3]].min())
    rest_max = float(np.delete(series, order[:3]).max())
    mean = float(series.mean())
    std = float(series.std())
    k_low = (rest_max - mean) / std
    k_high = (top3_min - mean) / std
    return top3_min, rest_max, top3_min / rest_max, mean, std, k_low, k_high


# ---------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("exp04: 変わり目検出の頑健性の定量評価")
    print("=" * 70)
    print(f"platform : {platform.platform()}")
    print(f"machine  : {platform.machine()}")
    print(f"python   : {sys.version.split()[0]}")
    print(f"numpy    : {np.__version__}")
    print(f"opencv   : {cv2.__version__}")
    print(f"video    : {VIDEO_PATH}")
    print(f"正解境界 : {list(GROUND_TRUTH)} (許容ずれ ±{TOLERANCE} フレーム)")
    print(f"乱数シード: {SEED_BASE} + sigma")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_pixels = width * height
    print(f"解像度   : {width}x{height} ({n_pixels} 画素), "
          f"総フレーム数 {n_frames}")

    # -----------------------------------------------------------------
    # 1. ノイズ条件ごとの系列計算と 3 方式の評価
    # -----------------------------------------------------------------
    print()
    print("-" * 70)
    print("実験1: ノイズ頑健性 (sigma ごとの precision / recall / F1)")
    print("-" * 70)

    series_d = {}
    series_h = {}
    noise_rows = []
    sep_rows = []
    ma_kernel = np.ones(MA_WINDOW) / MA_WINDOW
    for sigma in NOISE_SIGMAS:
        diffs, hist_l1 = compute_series(VIDEO_PATH, sigma, SEED_BASE + sigma)
        series_d[sigma] = diffs
        series_h[sigma] = hist_l1

        theory_offset = 2.0 * sigma / math.sqrt(math.pi) * n_pixels
        print(f"\n[sigma = {sigma}]")
        print(f"  D(t) 平均 = {diffs.mean():.4e}, 標準偏差 = {diffs.std():.4e}")
        print(f"  D(t) 平均の対ノイズなし増分 = "
              f"{diffs.mean() - series_d[0].mean():.4e} "
              f"(理論オフセット 2*sigma/sqrt(pi)*画素数 = {theory_offset:.4e})")
        print(f"  H(t) 平均 = {hist_l1.mean():.4e}, "
              f"標準偏差 = {hist_l1.std():.4e}")

        # 正解境界での D(t) ピーク値と移動平均基準線に対する比
        baseline = np.convolve(diffs, ma_kernel, mode="same")
        peak_info = []
        for g in GROUND_TRUTH:
            d_val = diffs[g - 1]           # diffs[g-1] がフレーム g の変化量
            ratio = d_val / baseline[g - 1]
            peak_info.append((g, d_val, ratio))
        print("  境界ピーク: " + ", ".join(
            f"D({g})={v:.4e} (D/移動平均基準線={r:.3f})"
            for g, v, r in peak_info))

        # 分離度統計 (D と H)
        for name, s in (("D", diffs), ("H", hist_l1)):
            t3, rmax, ratio, mean, std, k_lo, k_hi = separation_stats(s)
            print(f"  {name}(t) 分離度: 上位3ピーク最小 = {t3:.4e}, "
                  f"非ピーク最大 = {rmax:.4e}, 分離比 = {ratio:.2f}, "
                  f"F1=1 となる k の理論範囲 = ({k_lo:.3f}, {k_hi:.3f})")
            sep_rows.append([sigma, name, t3, rmax, ratio,
                             mean, std, k_lo, k_hi]
                            + ([f"{r:.4f}" for _, _, r in peak_info]
                               if name == "D" else ["", "", ""]))

        # 方式(1): 平均 + 3σ
        thr1, det1 = detect_by_mean_std(diffs, k=MEANSTD_K)
        tp, fp, fn, p, r, f1 = evaluate(det1)
        print(f"  方式(1) 平均+{MEANSTD_K:.0f}σ: 閾値 = {thr1:.4e}, "
              f"検出 = {fmt_frames(det1)}")
        print(f"    TP={tp} FP={fp} FN={fn} "
              f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
        noise_rows.append([sigma, "D_meanstd_k3", len(det1),
                           tp, fp, fn, p, r, f1])

        # 方式(3): 移動平均基準
        det3 = detect_by_moving_average(diffs, window=MA_WINDOW, k=MA_K)
        tp, fp, fn, p, r, f1 = evaluate(det3)
        print(f"  方式(3) 移動平均基準 (window={MA_WINDOW}, k={MA_K:.0f}): "
              f"検出 = {fmt_frames(det3)}")
        print(f"    TP={tp} FP={fp} FN={fn} "
              f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
        noise_rows.append([sigma, "D_movavg_w15k4", len(det3),
                           tp, fp, fn, p, r, f1])

        # 参考: ヒストグラム差分 + 平均 + 3σ
        thrh, deth = detect_by_mean_std(hist_l1, k=MEANSTD_K)
        tp, fp, fn, p, r, f1 = evaluate(deth)
        print(f"  参考 ヒストグラム差分+平均+{MEANSTD_K:.0f}σ: "
              f"閾値 = {thrh:.4e}, 検出 = {fmt_frames(deth)}")
        print(f"    TP={tp} FP={fp} FN={fn} "
              f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
        noise_rows.append([sigma, "H_meanstd_k3", len(deth),
                           tp, fp, fn, p, r, f1])

    noise_csv = os.path.join(DATA_DIR, "exp04_noise_f1.csv")
    with open(noise_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "method", "n_detected",
                    "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in noise_rows:
            w.writerow(row[:6] + [f"{v:.6f}" for v in row[6:]])
    print(f"\nCSV を保存しました: {noise_csv}")

    sep_csv = os.path.join(DATA_DIR, "exp04_separation.csv")
    with open(sep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "series", "top3_min", "rest_max", "ratio",
                    "mean", "std", "k_low", "k_high",
                    "ma_ratio_78", "ma_ratio_214", "ma_ratio_257"])
        for row in sep_rows:
            w.writerow([row[0], row[1]]
                       + [f"{v:.6e}" for v in row[2:7]]
                       + [f"{v:.4f}" for v in row[7:9]]
                       + row[9:])
    print(f"CSV を保存しました: {sep_csv}")

    # -----------------------------------------------------------------
    # 2. 方式(1) の k 感度
    # -----------------------------------------------------------------
    print()
    print("-" * 70)
    print("実験2: 方式(1) の k 感度 (k = 0.5〜6.0, 0.25 刻み)")
    print("-" * 70)

    k_rows = []
    safe_ranges = {}
    for sigma in K_SENS_SIGMAS:
        diffs = series_d[sigma]
        print(f"\n[sigma = {sigma}]")
        print(f"  {'k':>5} {'閾値':>12} {'検出数':>4} "
              f"{'TP':>3} {'FP':>3} {'FN':>3} "
              f"{'prec':>6} {'rec':>6} {'F1':>6}")
        f1_ok = []
        for k in K_VALUES:
            thr, det = detect_by_mean_std(diffs, k=float(k))
            tp, fp, fn, p, r, f1 = evaluate(det)
            print(f"  {k:5.2f} {thr:12.4e} {len(det):4d} "
                  f"{tp:3d} {fp:3d} {fn:3d} {p:6.3f} {r:6.3f} {f1:6.3f}")
            k_rows.append([sigma, float(k), thr, len(det),
                           tp, fp, fn, p, r, f1])
            if f1 == 1.0:
                f1_ok.append(float(k))
        if f1_ok:
            safe_ranges[sigma] = (min(f1_ok), max(f1_ok))
            print(f"  F1 = 1.000 となる k の範囲: "
                  f"{min(f1_ok):.2f} <= k <= {max(f1_ok):.2f}")
        else:
            safe_ranges[sigma] = None
            print("  F1 = 1.000 となる k は存在しない")

    if all(safe_ranges[s] is not None for s in K_SENS_SIGMAS):
        lo = max(safe_ranges[s][0] for s in K_SENS_SIGMAS)
        hi = min(safe_ranges[s][1] for s in K_SENS_SIGMAS)
        print(f"\n両条件 (sigma=0 と sigma=20) で F1 = 1.000 となる"
              f"安全な k の範囲: {lo:.2f} <= k <= {hi:.2f}")

    k_csv = os.path.join(DATA_DIR, "exp04_k_sensitivity.csv")
    with open(k_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "k", "threshold", "n_detected",
                    "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in k_rows:
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.6e}"]
                       + row[3:7] + [f"{v:.6f}" for v in row[7:]])
    print(f"CSV を保存しました: {k_csv}")

    # -----------------------------------------------------------------
    # 3. ヒストグラム差分法との比較(ノイズなし)
    # -----------------------------------------------------------------
    print()
    print("-" * 70)
    print("実験3: ヒストグラム差分法 (ビン 64, L1 距離) との比較")
    print("-" * 70)

    diffs0 = series_d[0]
    hist0 = series_h[0]

    peaks_d = detect_by_top_peaks(diffs0, n_peaks=3)
    peaks_h = detect_by_top_peaks(hist0, n_peaks=3)
    print(f"D(t) 上位3ピーク: {fmt_frames(peaks_d)}")
    print(f"H(t) 上位3ピーク: {fmt_frames(peaks_h)}")
    tp, fp, fn, p, r, f1 = evaluate(peaks_h)
    print(f"H(t) 上位3ピークの評価: TP={tp} FP={fp} FN={fn} "
          f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

    thrh0, deth0 = detect_by_mean_std(hist0, k=MEANSTD_K)
    tp, fp, fn, p, r, f1 = evaluate(deth0)
    print(f"H(t) 平均+3σ (閾値 {thrh0:.4e}): 検出 = {fmt_frames(deth0)}")
    print(f"  TP={tp} FP={fp} FN={fn} "
          f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

    # ピークとベースラインの分離度(2 位/最下位ピーク比較用の統計)
    for name, s in (("D(t)", diffs0), ("H(t)", hist0)):
        order = np.argsort(s)[::-1]
        top3 = s[order[:3]]
        rest = np.delete(s, order[:3])
        ratio = top3.min() / rest.max()
        print(f"{name}: 上位3ピーク最小値 = {top3.min():.4e}, "
              f"非ピーク最大値 = {rest.max():.4e}, 分離比 = {ratio:.2f}")

    series_csv = os.path.join(DATA_DIR, "exp04_series.csv")
    with open(series_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame"]
        for sigma in NOISE_SIGMAS:
            header += [f"D_sigma{sigma}", f"H_sigma{sigma}"]
        w.writerow(header)
        for i in range(len(diffs0)):
            row = [i + 1]
            for sigma in NOISE_SIGMAS:
                row += [f"{series_d[sigma][i]:.0f}",
                        f"{series_h[sigma][i]:.8f}"]
            w.writerow(row)
    print(f"CSV を保存しました: {series_csv}")

    # -----------------------------------------------------------------
    # 図の作成(白黒判別: 線種・マーカー・グレースケール)
    # -----------------------------------------------------------------
    frames_axis = np.arange(1, len(diffs0) + 1)

    # 図1: sigma に対する F1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {
        "D_meanstd_k3": ("-", "o", "0.0",
                         r"$D(t)$ mean$+3\sigma$ (method 1)"),
        "D_movavg_w15k4": ("--", "s", "0.0",
                           r"$D(t)$ moving average (method 3)"),
        "H_meanstd_k3": (":", "^", "0.45",
                         r"$H(t)$ hist. diff mean$+3\sigma$"),
    }
    for method, (ls, mk, col, label) in styles.items():
        xs = [row[0] for row in noise_rows if row[1] == method]
        ys = [row[8] for row in noise_rows if row[1] == method]
        ax.plot(xs, ys, linestyle=ls, marker=mk, color=col,
                markerfacecolor="white", markersize=7, label=label)
    ax.set_xlabel("Noise standard deviation $\\sigma$ [pixel value]")
    ax.set_ylabel("F1 score")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(list(NOISE_SIGMAS))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp04_f1_vs_sigma.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"図を保存しました: {path}")

    # 図2: k 感度(ノイズなし / sigma=20 の 2 段)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)
    for ax, sigma in zip(axes, K_SENS_SIGMAS):
        ks = [row[1] for row in k_rows if row[0] == sigma]
        ps = [row[7] for row in k_rows if row[0] == sigma]
        rs = [row[8] for row in k_rows if row[0] == sigma]
        f1s = [row[9] for row in k_rows if row[0] == sigma]
        ax.plot(ks, ps, linestyle="--", marker="s", color="0.0",
                markerfacecolor="white", markersize=6, label="precision")
        ax.plot(ks, rs, linestyle=":", marker="^", color="0.45",
                markerfacecolor="white", markersize=6, label="recall")
        ax.plot(ks, f1s, linestyle="-", marker="o", color="0.0",
                markerfacecolor="0.0", markersize=4, label="F1")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"$\\sigma = {sigma}$", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("Coefficient $k$ (threshold $= \\mu + k\\sigma_D$)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp04_k_sensitivity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"図を保存しました: {path}")

    # 図3: D(t) と H(t) の対比(ノイズなし、各系列の最大値で正規化)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    for ax, (s, label) in zip(
            axes,
            ((diffs0, "$D(t)$ / max (pixel difference)"),
             (hist0, "$H(t)$ / max (histogram L1)"))):
        ax.plot(frames_axis, s / s.max(), color="0.0", lw=0.9)
        for g in GROUND_TRUTH:
            ax.axvline(g, color="0.6", linestyle="--", lw=0.9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[-1].set_xlabel("Frame index")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp04_series_compare.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"図を保存しました: {path}")

    # 図4: ノイズ量による D(t) の変化(sigma = 0, 20, 40)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for sigma, ls, col in ((0, "-", "0.0"), (20, "--", "0.45"),
                           (40, ":", "0.7")):
        ax.plot(frames_axis, series_d[sigma], linestyle=ls, color=col,
                lw=1.0, label=f"$\\sigma = {sigma}$")
    for g in GROUND_TRUTH:
        ax.axvline(g, color="0.85", linestyle="-", lw=2.0, zorder=0)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("$D(t)$ (sum of absolute differences)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp04_noise_series.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"図を保存しました: {path}")

    print()
    print("exp04 完了")


if __name__ == "__main__":
    main()
