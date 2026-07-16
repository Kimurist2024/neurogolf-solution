# -*- coding: utf-8 -*-
"""exp06: 二値化閾値の自動決定 --- 固定閾値 128 と大津の方法の比較

課題8 の binarize_frame は固定閾値 128 でフレームを二値化している。
この設計選択を、判別分析法(大津の方法)による閾値の自動決定と比較する。

実験内容:
  [1] 大津の方法を numpy で自作実装し、cv2.threshold(THRESH_OTSU) の
      返す閾値と全 294 フレームで一致するかを検証する
      (cv2 は自作実装の検証用参考としてのみ使用)。
  [2] Cat - 66004.mp4 の 5 フレーム (0, 73, 147, 220, 293) について、
      (a) グレースケールヒストグラム + 固定閾値 128 と Otsu 閾値の位置
          + クラス間分散 sigma_B^2(t) の曲線を重ねた図
      (b) 固定 128 と Otsu の二値化結果の並置図
      を生成する。
  [3] 全 294 フレームで Otsu 閾値の推移を折れ線グラフ化し(固定 128 の
      水平線と重ねる)、平均・標準偏差・範囲を出力する。
  [4] 二値化品質の代理指標として各フレームの「白画素率」を
      固定 128 / フレーム毎 Otsu / 動画一括 Otsu の 3 方式で計算し、
      フレーム間の分散と隣接フレーム間平均変化量(ちらつきの指標)を
      比較する。

実行方法:
  cd "/Users/kimura2003/Downloads/rikadai kadai python" && \
  DATASET_ROOT="$PWD" MPLBACKEND=Agg \
  python3 experiments/exp06_otsu.py > results/exp06.txt 2>&1
"""

import csv
import os
import platform
import time

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
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FIXED_T = 128                       # 課題8 binarize_frame の固定閾値
SAMPLE_FRAMES = [0, 73, 147, 220, 293]  # 図示するフレーム番号

# 白黒印刷で判別できる描画設定
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "0.7",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------
# 大津の方法(判別分析法)の自作実装
# ---------------------------------------------------------------------
def otsu_threshold(gray):
    """大津の方法による閾値の自作実装(numpy のみ使用)。

    グレースケール画像のヒストグラムを正規化した p_i (i = 0..255) に
    対し、閾値 t で 2 クラス C0 = {0..t}, C1 = {t+1..255} に分けたとき、

        omega0(t) = sum_{i<=t} p_i          : C0 の生起確率
        omega1(t) = 1 - omega0(t)           : C1 の生起確率
        mu(t)     = sum_{i<=t} i * p_i      : 累積 1 次モーメント
        mu_T      = mu(255)                 : 全体平均
        mu0(t)    = mu(t) / omega0(t)       : C0 の平均
        mu1(t)    = (mu_T - mu(t)) / omega1(t) : C1 の平均

    のもとで、クラス間分散

        sigma_B^2(t) = omega0(t) * omega1(t) * (mu0(t) - mu1(t))^2
                     = (mu_T * omega0(t) - mu(t))^2
                       / (omega0(t) * (1 - omega0(t)))

    を最大化する t を閾値として返す(全分散 sigma_T^2 は t に
    依存しないため、クラス間分散の最大化はクラス内分散の最小化と
    等価である)。返り値の規約は cv2.threshold(THRESH_OTSU) と同じで、
    二値化は gray > t を白とする。同値の最大が複数ある場合は
    最小の t を採用する(np.argmax は最初の最大位置を返す)。

    返り値: (t, sigma_b2)  t は int、sigma_b2 は長さ 256 の配列
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega0 = np.cumsum(p)                       # omega0(t)
    mu = np.cumsum(p * np.arange(256))          # mu(t)
    mu_t = mu[-1]                               # mu_T
    omega1 = 1.0 - omega0
    valid = (omega0 > 0) & (omega1 > 0)         # 片方のクラスが空なら除外
    sigma_b2 = np.zeros(256)
    sigma_b2[valid] = ((mu_t * omega0[valid] - mu[valid]) ** 2
                       / (omega0[valid] * omega1[valid]))
    t = int(np.argmax(sigma_b2))
    return t, sigma_b2


def white_ratio_from_hist(hist, threshold, inclusive):
    """ヒストグラムから白画素率を計算する。

    inclusive=True  : gray >= threshold を白(課題8 の固定閾値の規約)
    inclusive=False : gray >  threshold を白(Otsu / cv2 の規約)
    """
    total = hist.sum()
    if inclusive:
        return hist[threshold:].sum() / total
    return hist[threshold + 1:].sum() / total


# ---------------------------------------------------------------------
# 動画の全フレーム走査
# ---------------------------------------------------------------------
print("=" * 70)
print("exp06: 二値化閾値の自動決定 --- 固定閾値 128 と大津の方法の比較")
print("=" * 70)
print(f"platform : {platform.platform()}")
print(f"machine  : {platform.machine()}")
print(f"python   : {platform.python_version()}")
print(f"numpy    : {np.__version__}")
print(f"opencv   : {cv2.__version__}")
print(f"video    : {VIDEO_PATH}")
print(f"固定閾値 : {FIXED_T} (課題8 binarize_frame と同一。gray >= 128 が白)")
print()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"動画を開けません: {VIDEO_PATH}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"入力動画: {w}x{h}, fps {fps:.2f}")
print()

hists = []          # 各フレームの 256 bin ヒストグラム
otsu_my = []        # 自作実装の Otsu 閾値
otsu_cv = []        # cv2.threshold(THRESH_OTSU) の閾値(検証用参考)
sample_gray = {}    # 図示用フレームのグレースケール画像
sample_sigma = {}   # 図示用フレームの sigma_B^2 曲線
t_my_total = 0.0    # 自作実装の累積計算時間
t_cv_total = 0.0    # cv2 の累積計算時間(二値化画像の生成込み・参考)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    t_mine, sigma_b2 = otsu_threshold(gray)
    t1 = time.perf_counter()
    t_cv2, _ = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2 = time.perf_counter()
    t_my_total += t1 - t0
    t_cv_total += t2 - t1

    hists.append(np.bincount(gray.ravel(), minlength=256))
    otsu_my.append(t_mine)
    otsu_cv.append(int(t_cv2))
    if idx in SAMPLE_FRAMES:
        sample_gray[idx] = gray.copy()
        sample_sigma[idx] = sigma_b2.copy()
    idx += 1
cap.release()

n_frames = idx
hists = np.array(hists)              # (n_frames, 256)
otsu_my = np.array(otsu_my)
otsu_cv = np.array(otsu_cv)
print(f"読み込んだフレーム数: {n_frames}")
print()

# ---------------------------------------------------------------------
# [1] 自作実装と cv2.threshold(THRESH_OTSU) の一致検証
# ---------------------------------------------------------------------
print("-" * 70)
print("[1] 自作実装と cv2.threshold(THRESH_OTSU) の一致検証")
print("-" * 70)
match = otsu_my == otsu_cv
n_match = int(match.sum())
max_diff = int(np.abs(otsu_my - otsu_cv).max())
print(f"一致フレーム数: {n_match} / {n_frames}")
print(f"閾値差の最大値: {max_diff}")
if n_match < n_frames:
    for i in np.where(~match)[0]:
        print(f"  不一致 frame {i}: 自作 {otsu_my[i]}, cv2 {otsu_cv[i]}")
print(f"自作実装の累積計算時間: {t_my_total:.4f} s "
      f"({t_my_total / n_frames * 1e3:.3f} ms/フレーム)")
print(f"cv2 の累積計算時間(二値化込み・参考): {t_cv_total:.4f} s "
      f"({t_cv_total / n_frames * 1e3:.3f} ms/フレーム)")
print()

# ---------------------------------------------------------------------
# [2] Otsu 閾値の統計と動画一括 Otsu
# ---------------------------------------------------------------------
print("-" * 70)
print("[2] Otsu 閾値の統計 (全フレーム)")
print("-" * 70)
t_mean = otsu_my.mean()
t_std = otsu_my.std()          # 母標準偏差 (ddof=0)
t_min = int(otsu_my.min())
t_max = int(otsu_my.max())
dt = np.abs(np.diff(otsu_my))
print(f"平均   : {t_mean:.2f}")
print(f"標準偏差: {t_std:.2f} (ddof=0)")
print(f"最小   : {t_min}")
print(f"最大   : {t_max}")
print(f"範囲   : {t_max - t_min}")
print(f"隣接フレーム間の閾値変化 |Δt| の平均: {dt.mean():.3f}")
print(f"隣接フレーム間の閾値変化 |Δt| の最大: {int(dt.max())}")

# 動画一括 Otsu: 全フレームのヒストグラムを合算して 1 つの閾値を決める
pooled_hist = hists.sum(axis=0)
pooled_p = pooled_hist.astype(np.float64) / pooled_hist.sum()
pooled_omega0 = np.cumsum(pooled_p)
pooled_mu = np.cumsum(pooled_p * np.arange(256))
pooled_mu_t = pooled_mu[-1]
pooled_valid = (pooled_omega0 > 0) & (pooled_omega0 < 1)
pooled_sigma = np.zeros(256)
pooled_sigma[pooled_valid] = (
    (pooled_mu_t * pooled_omega0[pooled_valid] - pooled_mu[pooled_valid]) ** 2
    / (pooled_omega0[pooled_valid] * (1.0 - pooled_omega0[pooled_valid])))
t_global = int(np.argmax(pooled_sigma))
print(f"動画一括 Otsu 閾値 (全フレーム合算ヒストグラム): {t_global}")
print()

# ---------------------------------------------------------------------
# [3] 白画素率の時系列 (固定 128 / フレーム毎 Otsu / 動画一括 Otsu)
# ---------------------------------------------------------------------
print("-" * 70)
print("[3] 白画素率の時間方向の安定性")
print("-" * 70)
ratio_fixed = np.array([white_ratio_from_hist(hh, FIXED_T, inclusive=True)
                        for hh in hists])
ratio_otsu = np.array([white_ratio_from_hist(hh, tt, inclusive=False)
                       for hh, tt in zip(hists, otsu_my)])
ratio_global = np.array([white_ratio_from_hist(hh, t_global, inclusive=False)
                         for hh in hists])

methods = [
    ("fixed128", "固定 128 (課題8)", ratio_fixed),
    ("otsu_frame", "フレーム毎 Otsu", ratio_otsu),
    ("otsu_global", f"動画一括 Otsu ({t_global})", ratio_global),
]
print(f"{'方式':<24s} {'平均[%]':>9s} {'分散':>12s} {'標準偏差[%]':>12s} "
      f"{'平均|Δr|[pt]':>13s} {'最大|Δr|[pt]':>13s}")
stats_rows = []
for key, label, r in methods:
    dr = np.abs(np.diff(r))
    row = {
        "key": key,
        "label": label,
        "mean": r.mean(),
        "var": r.var(),          # 母分散 (ddof=0)
        "std": r.std(),
        "mean_abs_dr": dr.mean(),
        "max_abs_dr": dr.max(),
    }
    stats_rows.append(row)
    print(f"{label:<24s} {r.mean() * 100:>9.3f} {r.var():>12.3e} "
          f"{r.std() * 100:>12.3f} {dr.mean() * 100:>13.4f} "
          f"{dr.max() * 100:>13.3f}")
var_fixed = ratio_fixed.var()
var_otsu = ratio_otsu.var()
var_global = ratio_global.var()
print()
print(f"白画素率分散の比: フレーム毎 Otsu / 固定 128   = "
      f"{var_otsu / var_fixed:.3f}")
print(f"白画素率分散の比: 動画一括 Otsu / 固定 128     = "
      f"{var_global / var_fixed:.3f}")
dr_fixed = np.abs(np.diff(ratio_fixed)).mean()
dr_otsu = np.abs(np.diff(ratio_otsu)).mean()
dr_global = np.abs(np.diff(ratio_global)).mean()
print(f"平均|Δr| の比:   フレーム毎 Otsu / 固定 128   = "
      f"{dr_otsu / dr_fixed:.3f}")
print(f"平均|Δr| の比:   動画一括 Otsu / 固定 128     = "
      f"{dr_global / dr_fixed:.3f}")
print()

# 閾値が変化した遷移と変化しなかった遷移で |Δr| を条件別に比較する
# (フレーム毎 Otsu の「ちらつき」が閾値の切り替わりに由来するかの確認)
dr_fixed_arr = np.abs(np.diff(ratio_fixed))
dr_otsu_arr = np.abs(np.diff(ratio_otsu))
changed = dt > 0
n_changed = int(changed.sum())
print(f"閾値が変化した遷移数: {n_changed} / {n_frames - 1}")
print(f"{'条件':<26s} {'遷移数':>6s} {'平均|Δr| Otsu[pt]':>18s} "
      f"{'平均|Δr| 固定128[pt]':>20s} {'比':>7s}")
for label, mask in [("閾値変化あり (|Δt|=1)", changed),
                    ("閾値変化なし (Δt=0)", ~changed)]:
    m_otsu = dr_otsu_arr[mask].mean()
    m_fixed = dr_fixed_arr[mask].mean()
    print(f"{label:<26s} {int(mask.sum()):>6d} {m_otsu * 100:>18.4f} "
          f"{m_fixed * 100:>20.4f} {m_otsu / m_fixed:>7.3f}")
print()

# ---------------------------------------------------------------------
# [4] 図示用 5 フレームの詳細
# ---------------------------------------------------------------------
print("-" * 70)
print("[4] 図示用フレームの閾値と白画素率")
print("-" * 70)
print(f"{'frame':>6s} {'Otsu(自作)':>10s} {'Otsu(cv2)':>10s} "
      f"{'白率 固定128[%]':>15s} {'白率 Otsu[%]':>13s}")
for i in SAMPLE_FRAMES:
    print(f"{i:>6d} {otsu_my[i]:>10d} {otsu_cv[i]:>10d} "
          f"{ratio_fixed[i] * 100:>15.2f} {ratio_otsu[i] * 100:>13.2f}")
print()

# ---------------------------------------------------------------------
# CSV 保存
# ---------------------------------------------------------------------
csv_thresholds = os.path.join(DATA_DIR, "exp06_thresholds.csv")
with open(csv_thresholds, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["frame", "otsu_my", "otsu_cv2", "match",
                   "white_ratio_fixed128", "white_ratio_otsu_frame",
                   "white_ratio_otsu_global"])
    for i in range(n_frames):
        wcsv.writerow([i, otsu_my[i], otsu_cv[i], int(match[i]),
                       f"{ratio_fixed[i]:.6f}", f"{ratio_otsu[i]:.6f}",
                       f"{ratio_global[i]:.6f}"])
print(f"CSV 保存: {csv_thresholds}")

csv_summary = os.path.join(DATA_DIR, "exp06_summary.csv")
with open(csv_summary, "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["key", "label", "mean_white_ratio", "var_white_ratio",
                   "std_white_ratio", "mean_abs_diff", "max_abs_diff"])
    for row in stats_rows:
        wcsv.writerow([row["key"], row["label"],
                       f"{row['mean']:.6f}", f"{row['var']:.6e}",
                       f"{row['std']:.6f}", f"{row['mean_abs_dr']:.6f}",
                       f"{row['max_abs_dr']:.6f}"])
    wcsv.writerow(["otsu_threshold_stats",
                   f"mean={t_mean:.3f} std={t_std:.3f} "
                   f"min={t_min} max={t_max} global={t_global}",
                   "", "", "", "", ""])
print(f"CSV 保存: {csv_summary}")
print()

# ---------------------------------------------------------------------
# 図1: ヒストグラム + 閾値位置 + sigma_B^2 曲線 (5 フレーム)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(len(SAMPLE_FRAMES), 1,
                         figsize=(6.8, 2.1 * len(SAMPLE_FRAMES)),
                         sharex=True)
for ax, i in zip(axes, SAMPLE_FRAMES):
    hist_norm = hists[i] / hists[i].sum()
    # 0 と 255 付近のスパイクで分布が潰れないよう縦軸は対数にする
    floor = 2e-7
    ax.fill_between(np.arange(256), np.maximum(hist_norm, floor), floor,
                    step="mid", color="0.75", label="histogram")
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.0)
    ax.axvline(FIXED_T, color="black", linestyle="--", linewidth=1.4,
               label=f"fixed {FIXED_T}")
    ax.axvline(otsu_my[i], color="black", linestyle="-", linewidth=1.4,
               label=f"Otsu {otsu_my[i]}")
    ax2 = ax.twinx()
    sig = sample_sigma[i]
    ax2.plot(np.arange(256), sig / sig.max(), linestyle=":",
             color="0.2", linewidth=1.2,
             label=r"$\sigma_B^2(t)$ (normalized)")
    ax2.set_ylim(0, 1.15)
    ax2.set_yticks([])
    ax.set_ylabel("rel. freq. (log)")
    ax.set_title(f"frame {i}  (Otsu t = {otsu_my[i]})", fontsize=10)
    if i == SAMPLE_FRAMES[0]:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
                  loc="upper center", ncol=4)
axes[-1].set_xlabel("Gray level")
axes[-1].set_xlim(0, 255)
fig.tight_layout()
fig_hist = os.path.join(FIG_DIR, "exp06_hist_thresholds.pdf")
fig.savefig(fig_hist)
plt.close(fig)
print(f"図 保存: {fig_hist}")

# ---------------------------------------------------------------------
# 図2: 固定 128 と Otsu の二値化結果の並置 (5 フレーム x 3 列)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(len(SAMPLE_FRAMES), 3,
                         figsize=(7.0, 1.45 * len(SAMPLE_FRAMES)))
for r, i in enumerate(SAMPLE_FRAMES):
    gray = sample_gray[i]
    bin_fixed = np.where(gray >= FIXED_T, 255, 0).astype(np.uint8)
    bin_otsu = np.where(gray > otsu_my[i], 255, 0).astype(np.uint8)
    for c, (img, title) in enumerate([
            (gray, "grayscale"),
            (bin_fixed, f"fixed {FIXED_T}"),
            (bin_otsu, f"Otsu {otsu_my[i]}")]):
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if r == 0:
            ax.set_title(title.split()[0], fontsize=10)
        if c == 0:
            ax.set_ylabel(f"frame {i}", fontsize=9)
        if c > 0:
            ax.set_xlabel(title, fontsize=8)
fig.tight_layout()
fig_bin = os.path.join(FIG_DIR, "exp06_binarize_compare.pdf")
fig.savefig(fig_bin)
plt.close(fig)
print(f"図 保存: {fig_bin}")

# ---------------------------------------------------------------------
# 図3: Otsu 閾値の推移 (全フレーム)。上段: 固定 128 との比較(全体)、
#       下段: Otsu 閾値付近の拡大(フレーム間変動の確認用)
# ---------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
ax_a.plot(np.arange(n_frames), otsu_my, "-", color="black", linewidth=1.2,
          label="Otsu (per frame)")
ax_a.axhline(FIXED_T, color="black", linestyle="--", linewidth=1.4,
             label=f"fixed {FIXED_T}")
ax_a.axhline(t_global, color="0.45", linestyle="-.", linewidth=1.4,
             label=f"Otsu (whole video) {t_global}")
ax_a.set_ylabel("Threshold")
ax_a.set_ylim(90, 135)
ax_a.legend(fontsize=8, loc="center right")
ax_a.set_title("(a) full range", fontsize=10)
ax_b.plot(np.arange(n_frames), otsu_my, "-", color="black", linewidth=1.2,
          drawstyle="steps-mid", label="Otsu (per frame)")
ax_b.axhline(t_global, color="0.45", linestyle="-.", linewidth=1.4,
             label=f"Otsu (whole video) {t_global}")
ax_b.axhline(t_mean, color="0.6", linestyle=":", linewidth=1.4,
             label=f"mean of per-frame Otsu {t_mean:.1f}")
ax_b.set_xlabel("Frame index")
ax_b.set_ylabel("Threshold")
ax_b.set_ylim(t_min - 0.6, t_max + 0.6)
ax_b.legend(fontsize=8, loc="upper left")
ax_b.set_title("(b) enlarged around per-frame Otsu thresholds", fontsize=10)
ax_b.set_xlim(0, n_frames - 1)
fig.tight_layout()
fig_series = os.path.join(FIG_DIR, "exp06_threshold_series.pdf")
fig.savefig(fig_series)
plt.close(fig)
print(f"図 保存: {fig_series}")

# ---------------------------------------------------------------------
# 図4: 白画素率の推移 (固定 128 / フレーム毎 Otsu / 動画一括 Otsu)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(np.arange(n_frames), ratio_fixed * 100, "-", color="black",
        linewidth=1.2, label=f"fixed {FIXED_T}")
ax.plot(np.arange(n_frames), ratio_otsu * 100, "--", color="0.35",
        linewidth=1.2, label="Otsu (per frame)")
ax.plot(np.arange(n_frames), ratio_global * 100, ":", color="0.5",
        linewidth=1.4, label=f"Otsu (whole video) {t_global}")
ax.set_xlabel("Frame index")
ax.set_ylabel("White pixel ratio [%]")
ax.set_xlim(0, n_frames - 1)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig_ratio = os.path.join(FIG_DIR, "exp06_white_ratio.pdf")
fig.savefig(fig_ratio)
plt.close(fig)
print(f"図 保存: {fig_ratio}")
print()

print("=" * 70)
print("exp06 完了")
print("=" * 70)
