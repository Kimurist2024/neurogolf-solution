# =====================================================================
# exp02: スペクトログラムのパラメータ感度実験
#
# report_audio.wav に対して STFT のパラメータ(窓長 nperseg、窓関数、
# 表示スケール)を系統的に変え、隠されたメッセージ「Good work」の
# 読みやすさを画像コントラスト指標で定量化する。
#
# 実験内容:
#   (1) nperseg in {128, 512, 1024, 4096, 16384}(noverlap = 75%、hann)
#       の 5 枚のスペクトログラムをグリッド図 1 枚に描画
#   (2) 窓関数 {hann, boxcar, blackman}(nperseg = 1024)の 3 枚比較図
#   (3) dB 表示 vs 線形表示の比較図(nperseg = 1024、hann)
#   (4) メッセージ帯域(1--12 kHz)の表示画素値に対する
#       RMS コントラストとミケルソンコントラスト(P5/P95)を全設定で計算
#
# 出力:
#   experiments/figures/exp02_nperseg_grid.pdf
#   experiments/figures/exp02_window.pdf
#   experiments/figures/exp02_scale.pdf
#   experiments/figures/exp02_contrast_vs_nperseg.pdf
#   experiments/data/exp02_contrast.csv
# =====================================================================

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", "/content/drive/MyDrive/Colab Notebooks"
)
AUDIO_PATH = os.path.join(DATASET_ROOT, "dataset", "audio", "report_audio.wav")
FIG_DIR = os.path.join(DATASET_ROOT, "experiments", "figures")
DATA_DIR = os.path.join(DATASET_ROOT, "experiments", "data")

# メッセージが目視で確認できる帯域 [Hz](ベースレポートの図より)
BAND_LOW_HZ = 1000.0
BAND_HIGH_HZ = 12000.0
# dB 表示のダイナミックレンジ(最大値からの下げ幅)[dB]
DYN_RANGE_DB = 80.0

NPERSEG_LIST = [128, 512, 1024, 4096, 16384]
WINDOW_LIST = ["hann", "boxcar", "blackman"]
BASE_NPERSEG = 1024
OVERLAP_RATIO = 0.75


def load_audio(path):
    """wav を読み込み、モノラルの float64 配列と標本化周波数を返す。"""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return rate, data.astype(np.float64)


def compute_spectrogram(data, rate, nperseg, window):
    """STFT スペクトログラムを計算する。noverlap は窓長の 75% とする。"""
    noverlap = int(nperseg * OVERLAP_RATIO)
    freqs, times, sxx = signal.spectrogram(
        data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap
    )
    return freqs, times, sxx


def to_db_image(sxx):
    """パワーを dB 化し、[最大値-DYN_RANGE_DB, 最大値] を [0,1] に正規化する。

    表示(pcolormesh の vmin/vmax)と同じ規則で正規化した「表示画素値」
    を返すことで、図の見た目とコントラスト指標を対応させる。
    """
    sxx_db = 10.0 * np.log10(sxx + 1e-12)
    vmax = sxx_db.max()
    vmin = vmax - DYN_RANGE_DB
    return np.clip((sxx_db - vmin) / DYN_RANGE_DB, 0.0, 1.0), vmin, vmax


def to_linear_image(sxx):
    """パワーを最大値 1 に正規化した線形表示の画素値を返す。"""
    return sxx / sxx.max()


def band_metrics(image, freqs):
    """メッセージ帯域の表示画素値からコントラスト指標を計算する。

    RMS コントラスト: 帯域内画素値の標準偏差
    ミケルソンコントラスト: (P95 - P5) / (P95 + P5)(外れ値に頑健な
    5/95 パーセンタイル版)
    """
    mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    band = image[mask, :]
    p5, p95 = np.percentile(band, [5.0, 95.0])
    rms = float(band.std())
    michelson = float((p95 - p5) / (p95 + p5 + 1e-12))
    return {
        "band_bins": int(mask.sum()),
        "band_cols": int(band.shape[1]),
        "rms_contrast": rms,
        "michelson_p5p95": michelson,
    }


def draw_panel(ax, times, freqs, values, vmin, vmax, title, cbar_label):
    """1 枚のスペクトログラムをグレースケールで描画する。"""
    mesh = ax.pcolormesh(
        times, freqs / 1000.0, values, shading="auto",
        cmap="gray", vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title(title, fontsize=10)
    plt.colorbar(mesh, ax=ax, label=cbar_label)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    rate, data = load_audio(AUDIO_PATH)
    duration = len(data) / rate
    print(f"音声ファイル: {AUDIO_PATH}")
    print(f"標本化周波数: {rate} Hz, サンプル数: {len(data)}, "
          f"長さ: {duration:.2f} 秒")
    print(f"メッセージ帯域: {BAND_LOW_HZ:.0f}-{BAND_HIGH_HZ:.0f} Hz, "
          f"dB 表示ダイナミックレンジ: {DYN_RANGE_DB:.0f} dB")
    print()

    rows = []

    # ----------------------------------------------------------------
    # 実験 1: 窓長 nperseg の感度(hann、dB 表示)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(len(NPERSEG_LIST), 1, figsize=(8.0, 14.0))
    for ax, nperseg in zip(axes, NPERSEG_LIST):
        freqs, times, sxx = compute_spectrogram(data, rate, nperseg, "hann")
        image, vmin, vmax = to_db_image(sxx)
        hop = nperseg - int(nperseg * OVERLAP_RATIO)
        df = rate / nperseg
        dt_ms = hop / rate * 1000.0
        met = band_metrics(image, freqs)
        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        draw_panel(
            ax, times, freqs, sxx_db, vmin, vmax,
            f"nperseg={nperseg} "
            f"($\\Delta f$={df:.1f} Hz, $\\Delta t$={dt_ms:.2f} ms)",
            "Power [dB]",
        )
        rows.append({
            "group": "nperseg", "window": "hann", "scale": "dB",
            "nperseg": nperseg, "noverlap": int(nperseg * OVERLAP_RATIO),
            "hop": hop, "delta_f_hz": round(df, 2),
            "delta_t_ms": round(dt_ms, 3),
            **met,
        })
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp02_nperseg_grid.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"図を保存しました: {out}")

    # ----------------------------------------------------------------
    # 実験 2: 窓関数の感度(nperseg=1024、dB 表示)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(len(WINDOW_LIST), 1, figsize=(8.0, 9.0))
    for ax, window in zip(axes, WINDOW_LIST):
        freqs, times, sxx = compute_spectrogram(
            data, rate, BASE_NPERSEG, window
        )
        image, vmin, vmax = to_db_image(sxx)
        hop = BASE_NPERSEG - int(BASE_NPERSEG * OVERLAP_RATIO)
        met = band_metrics(image, freqs)
        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        draw_panel(
            ax, times, freqs, sxx_db, vmin, vmax,
            f"window={window} (nperseg={BASE_NPERSEG})", "Power [dB]",
        )
        rows.append({
            "group": "window", "window": window, "scale": "dB",
            "nperseg": BASE_NPERSEG,
            "noverlap": int(BASE_NPERSEG * OVERLAP_RATIO),
            "hop": hop, "delta_f_hz": round(rate / BASE_NPERSEG, 2),
            "delta_t_ms": round(hop / rate * 1000.0, 3),
            **met,
        })
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp02_window.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"図を保存しました: {out}")

    # ----------------------------------------------------------------
    # 実験 3: dB 表示 vs 線形表示(nperseg=1024、hann)
    # ----------------------------------------------------------------
    freqs, times, sxx = compute_spectrogram(data, rate, BASE_NPERSEG, "hann")
    hop = BASE_NPERSEG - int(BASE_NPERSEG * OVERLAP_RATIO)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5))

    image_db, vmin, vmax = to_db_image(sxx)
    sxx_db = 10.0 * np.log10(sxx + 1e-12)
    draw_panel(
        axes[0], times, freqs, sxx_db, vmin, vmax,
        "dB scale (nperseg=1024, hann)", "Power [dB]",
    )
    met_db = band_metrics(image_db, freqs)
    rows.append({
        "group": "scale", "window": "hann", "scale": "dB",
        "nperseg": BASE_NPERSEG, "noverlap": int(BASE_NPERSEG * OVERLAP_RATIO),
        "hop": hop, "delta_f_hz": round(rate / BASE_NPERSEG, 2),
        "delta_t_ms": round(hop / rate * 1000.0, 3),
        **met_db,
    })

    image_lin = to_linear_image(sxx)
    draw_panel(
        axes[1], times, freqs, image_lin, 0.0, 1.0,
        "linear scale (nperseg=1024, hann)", "Normalized power",
    )
    met_lin = band_metrics(image_lin, freqs)
    rows.append({
        "group": "scale", "window": "hann", "scale": "linear",
        "nperseg": BASE_NPERSEG, "noverlap": int(BASE_NPERSEG * OVERLAP_RATIO),
        "hop": hop, "delta_f_hz": round(rate / BASE_NPERSEG, 2),
        "delta_t_ms": round(hop / rate * 1000.0, 3),
        **met_lin,
    })
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp02_scale.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"図を保存しました: {out}")

    # ----------------------------------------------------------------
    # 実験 4: コントラスト指標 vs 窓長のプロット(白黒判別可能)
    # ----------------------------------------------------------------
    nperseg_rows = [r for r in rows if r["group"] == "nperseg"]
    xs = [r["nperseg"] for r in nperseg_rows]
    rms_vals = [r["rms_contrast"] for r in nperseg_rows]
    mic_vals = [r["michelson_p5p95"] for r in nperseg_rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(xs, rms_vals, "k-o", label="RMS contrast")
    ax.plot(xs, mic_vals, "k--s", label="Michelson contrast (P5/P95)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("nperseg [samples]")
    ax.set_ylabel("Contrast index")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp02_contrast_vs_nperseg.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"図を保存しました: {out}")

    # ----------------------------------------------------------------
    # CSV 保存と結果表示
    # ----------------------------------------------------------------
    csv_path = os.path.join(DATA_DIR, "exp02_contrast.csv")
    fieldnames = [
        "group", "window", "scale", "nperseg", "noverlap", "hop",
        "delta_f_hz", "delta_t_ms",
        "band_bins", "band_cols", "rms_contrast", "michelson_p5p95",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV を保存しました: {csv_path}")
    print()

    header = (f"{'group':<8} {'window':<9} {'scale':<7} {'nperseg':>7} "
              f"{'df[Hz]':>8} {'dt[ms]':>8} "
              f"{'bins':>5} {'cols':>5} {'RMS':>7} {'Michelson':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['group']:<8} {r['window']:<9} {r['scale']:<7} "
              f"{r['nperseg']:>7} {r['delta_f_hz']:>8.2f} "
              f"{r['delta_t_ms']:>8.3f} "
              f"{r['band_bins']:>5} {r['band_cols']:>5} "
              f"{r['rms_contrast']:>7.4f} {r['michelson_p5p95']:>9.4f}")


if __name__ == "__main__":
    main()
