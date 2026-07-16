# =====================================================================
# 計算機科学応用演習 final exam
# 6324034 木村 竜輝
#
# 課題1: 動画の読み込み(サイズ・FPS・総フレーム数)
# 課題2: サムネイル画像の作成(代表フレーム選択)
# 課題3: ファイル名のパース(タイトルと ID)
# 課題4: 色反転と動画の保存(0.3 倍リサイズ)
# 課題5: 音声の可視化(スペクトログラム)
# 課題6: n-gram(2-gram / stop words 除去)
# 課題7: 動画の変わり目検出(フレーム間差分)
# 課題8: オブジェクト指向(Video クラス・高階関数)
# =====================================================================

import os
import re
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# ---------------------------------------------------------------------
# パス設定
# 課題指定の Colab パスを既定とし、環境変数 DATASET_ROOT があれば
# ローカル実行用に差し替える(Colab では未設定なので指定パスになる)
# ---------------------------------------------------------------------
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT", "/content/drive/MyDrive/Colab Notebooks"
)
VIDEO_DIR = os.path.join(DATASET_ROOT, "dataset", "videos")
AUDIO_DIR = os.path.join(DATASET_ROOT, "dataset", "audio")
TEXT_DIR = os.path.join(DATASET_ROOT, "dataset", "text")
VIDEO_OUT_DIR = os.path.join(VIDEO_DIR, "output")
IMAGE_OUT_DIR = os.path.join(DATASET_ROOT, "dataset", "images", "output")

STUDENT_ID = "6324034"

PEOPLE_VIDEO = os.path.join(VIDEO_DIR, "People - 84973.mp4")
CAT_VIDEO = os.path.join(VIDEO_DIR, "Cat - 66004.mp4")
DETECTION_VIDEO = os.path.join(VIDEO_DIR, "Detection.mp4")
REPORT_AUDIO = os.path.join(AUDIO_DIR, "report_audio.wav")
TEXT_ENGLISH = os.path.join(TEXT_DIR, "text_english.txt")


# =====================================================================
# 課題1: 動画の読み込み
# =====================================================================
def task1_video_info(video_path):
    """動画を読み込み、縦・横のサイズ、FPS、総フレーム数を出力する。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"横のサイズ: {width}")
    print(f"縦のサイズ: {height}")
    print(f"FPS: {fps}")
    print(f"総フレーム数: {frame_count}")
    return width, height, fps, frame_count


# =====================================================================
# 課題2: サムネイル画像の作成
# 中央フレームではなく「代表フレーム」を自作の基準で選ぶ:
# 動画全体の平均ヒストグラムに最も近いヒストグラムを持つフレーム
# =====================================================================
def compute_gray_histogram(frame, bins=64):
    """フレームのグレースケール正規化ヒストグラムを返す。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def task2_thumbnail(video_path, out_path, sample_step=5):
    """動画の代表フレームをサムネイルとして保存する。

    全フレームを sample_step 間隔でサンプリングし、各フレームの
    グレースケールヒストグラムを計算する。全サンプルの平均ヒスト
    グラムに L2 距離が最も近いフレームを「動画全体を最もよく代表
    するフレーム」として選択する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")

    histograms = []
    frame_indices = []
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % sample_step == 0:
            histograms.append(compute_gray_histogram(frame))
            frame_indices.append(index)
        index += 1

    hist_array = np.array(histograms)
    mean_hist = hist_array.mean(axis=0)
    distances = np.linalg.norm(hist_array - mean_hist, axis=1)
    best_index = frame_indices[int(np.argmin(distances))]

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_index)
    ret, best_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"フレーム {best_index} を読み込めません")

    cv2.imwrite(out_path, best_frame)
    print(f"代表フレーム番号: {best_index}")
    print(f"サムネイルを保存しました: {out_path}")
    return best_index, best_frame


# =====================================================================
# 課題3: ファイル名のパース
# =====================================================================
def task3_parse_filename(filename):
    """ファイル名からタイトルと ID を取り出す。

    例: "People - 84973.mp4" -> ("People", "84973")
    拡張子を除いた後、右端の " - " で 1 回だけ分割することで、
    タイトル側に "-" が含まれる場合にも対応する。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.rsplit(" - ", 1)
    if len(parts) != 2:
        raise ValueError(f"'タイトル - ID' 形式ではありません: {filename}")
    title = parts[0].strip()
    video_id = parts[1].strip()
    print(f"Title:  {title}")
    print(f"ID:  {video_id}")
    return title, video_id


# =====================================================================
# 課題4: 色反転と動画の保存
# =====================================================================
def task4_reverse_video(video_path, out_path, scale=0.3):
    """各フレームの色を反転し、縦横 scale 倍にした動画を保存する。

    反転後の画素値 = 画素値の最大値(255) - 元の画素値
    cv2.bitwise_not() は使わず、numpy の配列演算で計算する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    new_size = (int(width * scale), int(height * scale))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, new_size)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        reversed_frame = 255 - frame  # uint8 の最大値から引く
        resized = cv2.resize(reversed_frame, new_size)
        writer.write(resized)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"色反転動画を保存しました: {out_path}")
    print(f"出力サイズ: {new_size[0]}x{new_size[1]}, フレーム数: {frame_count}")
    return out_path


# =====================================================================
# 課題5: 音声の可視化(スペクトログラム)
# =====================================================================
def task5_spectrogram(audio_path, out_path, nperseg=1024, noverlap=768,
                      window="hann", vmin_db=-100, cmap="gray"):
    """音声のスペクトログラムを描画して保存する。

    STFT のパラメータ(窓長 nperseg、オーバーラップ noverlap、窓関数)
    を引数で調整できるようにし、時間-周波数領域に隠されたメッセージを
    可視化する。強度は dB スケールで表示する。
    """
    rate, data = wavfile.read(audio_path)
    if data.ndim > 1:  # ステレオならモノラル化
        data = data.mean(axis=1)
    data = data.astype(np.float64)

    freqs, times, sxx = signal.spectrogram(
        data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap
    )
    sxx_db = 10 * np.log10(sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(times, freqs, sxx_db, shading="auto",
                         cmap=cmap, vmin=vmin_db)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"Spectrogram (nperseg={nperseg}, noverlap={noverlap}, "
                 f"window={window})")
    fig.colorbar(mesh, ax=ax, label="Power [dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"サンプリングレート: {rate} Hz, サンプル数: {len(data)}")
    print(f"スペクトログラムを保存しました: {out_path}")
    return rate, len(data)


# =====================================================================
# 課題6: n-gram
# =====================================================================
# 6-b で用いる stop words(冠詞・be 動詞・前置詞・接続詞・代名詞など、
# 文の内容よりも文法機能を担う高頻度語を自分で設計した)
STOP_WORDS = {
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


def tokenize(text):
    """テキストを小文字化し、英単語のみを取り出す。"""
    return re.findall(r"[a-z]+", text.lower())


def build_2grams(words):
    """単語列から連続する 2 語の組(2-gram)のリストを作る。"""
    return [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]


def plot_top_2grams(counter, top_n, out_path, title):
    """2-gram の出現頻度上位 top_n 個を棒グラフとして保存する。"""
    top = counter.most_common(top_n)
    labels = [pair for pair, _ in top]
    values = [count for _, count in top]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="gray", edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("2-gram")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    for pair, count in top:
        print(f"{pair}: {count}")
    print(f"グラフを保存しました: {out_path}")
    return top


def task6a_2gram(text_path, out_path, top_n=10):
    """テキストの 2-gram 出現頻度上位 top_n をグラフ化する。"""
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    words = tokenize(text)
    counter = Counter(build_2grams(words))
    print(f"総単語数: {len(words)}, 異なり 2-gram 数: {len(counter)}")
    return plot_top_2grams(counter, top_n, out_path,
                           "Top 2-grams (with stop words)")


def task6b_2gram_no_stopwords(text_path, out_path, top_n=10):
    """stop words を除去した 2-gram 出現頻度上位 top_n をグラフ化する。"""
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    words = [w for w in tokenize(text) if w not in STOP_WORDS]
    counter = Counter(build_2grams(words))
    print(f"stop words 除去後の単語数: {len(words)}, "
          f"異なり 2-gram 数: {len(counter)}")
    return plot_top_2grams(counter, top_n, out_path,
                           "Top 2-grams (stop words removed)")


# =====================================================================
# 課題7: 動画の変わり目検出
# =====================================================================
def task7_frame_differences(video_path):
    """全フレームについて前フレームとの画素値変化量を計算する。

    1. 各フレームをグレースケールにする
    2. t 番目と t-1 番目の各画素の差の絶対値を計算する
       (absdiff() は使わず、符号付き整数に変換して numpy で計算)
    3. 全画素の差を足し合わせて変化量とする
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")

    diffs = []
    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int32)
        if prev_gray is not None:
            diffs.append(int(np.abs(gray - prev_gray).sum()))
        prev_gray = gray
    cap.release()
    # diffs[t-1] が「フレーム t-1 と t の間の変化量」
    return np.array(diffs, dtype=np.float64)


def task7a_plot_differences(diffs, out_path):
    """横軸フレーム番号、縦軸変化量のグラフを描画する。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(diffs) + 1), diffs, color="black", lw=0.8)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Sum of absolute differences")
    ax.set_title("Inter-frame change")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"変化量グラフを保存しました: {out_path}")


def detect_by_mean_std(diffs, k=3.0):
    """閾値 = 平均 + k×標準偏差 で変わり目を検出する。"""
    threshold = diffs.mean() + k * diffs.std()
    frames = np.where(diffs > threshold)[0] + 1  # diffs[i] はフレーム i+1
    return threshold, frames


def detect_by_top_peaks(diffs, n_peaks=3):
    """変化量の上位 n_peaks 個のフレームを変わり目とする。"""
    order = np.argsort(diffs)[::-1][:n_peaks]
    return np.sort(order + 1)


def detect_by_moving_average(diffs, window=15, k=4.0):
    """移動平均で平滑化した基準線から大きく外れたフレームを検出する。

    各フレームの変化量を、その近傍 window フレームの移動平均と比較し、
    移動平均の k 倍を超えたフレームを変わり目とする。
    """
    kernel = np.ones(window) / window
    baseline = np.convolve(diffs, kernel, mode="same")
    frames = np.where(diffs > k * baseline)[0] + 1
    return frames


def save_boundary_frames(video_path, frame_indices, out_dir, prefix):
    """検出したフレームとその前フレームの画像を保存して目視確認する。"""
    cap = cv2.VideoCapture(video_path)
    saved = []
    for fi in frame_indices:
        for offset in (-1, 0):
            target = fi + offset
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(out_dir, f"{prefix}_frame{target}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
    cap.release()
    print(f"確認用フレーム画像を {len(saved)} 枚保存しました")
    return saved


# =====================================================================
# 課題8: オブジェクト指向(Video クラス・高階関数)
# =====================================================================
def invert_frame(frame):
    """フレームの色を反転する(課題 4 と同じ処理)。"""
    return 255 - frame


def grayscale_frame(frame):
    """フレームをグレースケール化する(動画保存用に 3ch へ戻す)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def binarize_frame(frame, threshold=128):
    """フレームを二値化する(動画保存用に 3ch へ戻す)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class Video:
    """動画 1 本を表すクラス。

    インスタンス変数:
        path      -- 動画ファイルのパス
        title     -- ファイル名から取り出したタイトル
        video_id  -- ファイル名から取り出した ID
        thumbnail -- 代表フレームの画像(numpy 配列)
    """

    def __init__(self, path):
        self.path = path
        self.title = None
        self.video_id = None
        self.thumbnail = None

    # --- 課題 3 をメソッド化 ---
    def parse_filename(self):
        """ファイル名からタイトルと ID を取り出してインスタンス変数に代入する。"""
        stem = os.path.splitext(os.path.basename(self.path))[0]
        parts = stem.rsplit(" - ", 1)
        if len(parts) != 2:
            raise ValueError(f"'タイトル - ID' 形式ではありません: {self.path}")
        self.title = parts[0].strip()
        self.video_id = parts[1].strip()
        return self.title, self.video_id

    # --- 課題 2 をメソッド化 ---
    def create_thumbnail(self, sample_step=5):
        """代表フレームを選んでインスタンス変数 thumbnail に代入する。"""
        best_index, best_frame = task2_thumbnail_frame(
            self.path, sample_step=sample_step
        )
        self.thumbnail = best_frame
        return best_index

    def get_info(self):
        """4 つのインスタンス変数を辞書型で返す。"""
        return {
            "path": self.path,
            "title": self.title,
            "video_id": self.video_id,
            "thumbnail": self.thumbnail,
        }

    # --- 課題 4 をメソッド化(高階関数版) ---
    def process_video(self, frame_func, out_path, scale=0.3):
        """フレーム処理関数 frame_func を全フレームに適用した動画を作る。

        frame_func: フレーム(numpy 配列)を受け取り、同じ大きさの
                    フレームを返す関数。二値化・グレースケール化・
                    色反転などを差し替えるだけで動画処理ができる。
        """
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"動画を開けません: {self.path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        new_size = (int(width * scale), int(height * scale))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, new_size)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = frame_func(frame)
            writer.write(cv2.resize(processed, new_size))
        cap.release()
        writer.release()
        return out_path

    def create_inverse_video(self, out_path, scale=0.3):
        """色反転動画を作成する(process_video に invert_frame を渡す)。"""
        return self.process_video(invert_frame, out_path, scale=scale)


def task2_thumbnail_frame(video_path, sample_step=5):
    """課題 2 の代表フレーム選択(フレームを返す内部版)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    histograms = []
    frame_indices = []
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % sample_step == 0:
            histograms.append(compute_gray_histogram(frame))
            frame_indices.append(index)
        index += 1
    hist_array = np.array(histograms)
    mean_hist = hist_array.mean(axis=0)
    distances = np.linalg.norm(hist_array - mean_hist, axis=1)
    best_index = frame_indices[int(np.argmin(distances))]
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_index)
    ret, best_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"フレーム {best_index} を読み込めません")
    return best_index, best_frame


# =====================================================================
# メイン処理
# =====================================================================
def main():
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("課題1: 動画の読み込み")
    print("=" * 60)
    task1_video_info(PEOPLE_VIDEO)

    print()
    print("=" * 60)
    print("課題2: サムネイル画像の作成")
    print("=" * 60)
    thumb_path = os.path.join(IMAGE_OUT_DIR, f"{STUDENT_ID}_thumbnail.jpg")
    task2_thumbnail(PEOPLE_VIDEO, thumb_path)

    print()
    print("=" * 60)
    print("課題3: ファイル名のパース")
    print("=" * 60)
    task3_parse_filename("People - 84973.mp4")
    task3_parse_filename("Cat - 66004.mp4")

    print()
    print("=" * 60)
    print("課題4: 色反転と動画の保存")
    print("=" * 60)
    reverse_path = os.path.join(VIDEO_OUT_DIR, f"{STUDENT_ID}_cat_reverse.mp4")
    task4_reverse_video(CAT_VIDEO, reverse_path, scale=0.3)

    print()
    print("=" * 60)
    print("課題5: 音声の可視化")
    print("=" * 60)
    spec_path = os.path.join(IMAGE_OUT_DIR, f"{STUDENT_ID}_spectrogram.jpg")
    task5_spectrogram(REPORT_AUDIO, spec_path)

    print()
    print("=" * 60)
    print("課題6-a: 2-gram")
    print("=" * 60)
    gram_path = os.path.join(IMAGE_OUT_DIR, f"{STUDENT_ID}_2gram_graph.jpg")
    task6a_2gram(TEXT_ENGLISH, gram_path)

    print()
    print("=" * 60)
    print("課題6-b: stop words 除去")
    print("=" * 60)
    gram_sw_path = os.path.join(
        IMAGE_OUT_DIR, f"{STUDENT_ID}_2gram_graph_no_stopwords.jpg"
    )
    task6b_2gram_no_stopwords(TEXT_ENGLISH, gram_sw_path)

    print()
    print("=" * 60)
    print("課題7: 動画の変わり目検出")
    print("=" * 60)
    diffs = task7_frame_differences(DETECTION_VIDEO)
    print(f"フレーム間変化量の数: {len(diffs)}")
    diff_plot_path = os.path.join(IMAGE_OUT_DIR, f"{STUDENT_ID}_framediff.jpg")
    task7a_plot_differences(diffs, diff_plot_path)

    print("--- 方法1: 平均 + 3×標準偏差 ---")
    threshold, frames_std = detect_by_mean_std(diffs, k=3.0)
    print(f"閾値: {threshold:.1f}")
    print(f"検出フレーム: {frames_std.tolist()}")

    print("--- 方法2: 変化量上位3ピーク ---")
    frames_peak = detect_by_top_peaks(diffs, n_peaks=3)
    print(f"検出フレーム: {frames_peak.tolist()}")

    print("--- 方法3: 移動平均基準 ---")
    frames_ma = detect_by_moving_average(diffs, window=15, k=4.0)
    print(f"検出フレーム: {frames_ma.tolist()}")

    save_boundary_frames(DETECTION_VIDEO, frames_peak, IMAGE_OUT_DIR,
                         f"{STUDENT_ID}_boundary")

    print()
    print("=" * 60)
    print("課題8: Video クラスの動作確認")
    print("=" * 60)
    video = Video(PEOPLE_VIDEO)
    video.parse_filename()
    thumb_index = video.create_thumbnail()
    info = video.get_info()
    print(f"path: {info['path']}")
    print(f"title: {info['title']}")
    print(f"video_id: {info['video_id']}")
    print(f"thumbnail: フレーム {thumb_index}, "
          f"形状 {info['thumbnail'].shape}")

    cat = Video(CAT_VIDEO)
    cat.parse_filename()
    print("--- 高階関数による動画処理 ---")
    for func, name in [(binarize_frame, "binary"),
                       (grayscale_frame, "gray"),
                       (invert_frame, "reverse_hof")]:
        out = os.path.join(VIDEO_OUT_DIR, f"{STUDENT_ID}_cat_{name}.mp4")
        cat.process_video(func, out, scale=0.3)
        print(f"{name}: {out}")


if __name__ == "__main__":
    main()
