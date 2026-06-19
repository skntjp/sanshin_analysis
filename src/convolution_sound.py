import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram

# =========================================================
# 設定
# =========================================================
# 1) 入力ファイル
FILE_IMP_NPY = Path("src/sanshin_force_imp_real_obs_pressure.npy")
FILE_IMP_JSON = Path("src/sanshin_force_imp_real_summary.json")
FILE_STRING_SOUND = Path("sound_source/GenSound1.txt")

# 2) 出力ファイル
OUTPUT_WAV = Path("src/sanshin_convolved_44100Hz.wav")
OUTPUT_SPEC_PNG = Path("src/sanshin_convolved_spectrogram_20k.png")

# 3) 中間処理のサンプリングレート（弦データのレートに合わせる）
PROCESSING_FS = 20000

# 4) 最終的な出力WAVファイルのサンプリングレート
FINAL_WAV_FS = 44100

# 5) スペクトログラムのパラメータ
NPERSEG = 512  # 1フレームのサンプル窓幅（周波数解像度と時間解像度のトレードオフ）
NOVERLAP = 384  # フレーム間のオーバーラップサンプル数


def load_simulation_dt(json_path):
    """JSONのメタデータから時間刻み dt を自動取得する"""
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        params = (
            meta.get("summary", {}).get("params", {})
            if "summary" in meta
            else meta.get("params", {})
        )
        return float(params["dt"])


def main():
    # -----------------------------------------------------
    # 1. データのロードと初期サンプリングレートの確認
    # -----------------------------------------------------
    print("Loading raw files...")
    if not FILE_IMP_NPY.exists() or not FILE_STRING_SOUND.exists():
        raise FileNotFoundError(
            "入力ファイルが見つかりません。パスを確認してください。"
        )

    h_raw = np.load(FILE_IMP_NPY).astype(np.float64)
    h_raw -= np.mean(h_raw)

    dt_sim = load_simulation_dt(FILE_IMP_JSON)
    sim_fs = 1.0 / dt_sim

    x_t = np.loadtxt(FILE_STRING_SOUND, dtype=np.float64)
    x_t -= np.mean(x_t)

    # -----------------------------------------------------
    # 2. インパルス応答を 20 kHz にダウンサンプリング
    # -----------------------------------------------------
    print(f"Downsampling Impulse Response to {PROCESSING_FS} Hz...")
    gcd_ir = np.gcd(int(round(sim_fs)), PROCESSING_FS)
    up_ir = int(PROCESSING_FS // gcd_ir)
    down_ir = int(round(sim_fs) // gcd_ir)

    if down_ir > 1000 or up_ir > 1000:
        down_ir = 1000
        up_ir = int(round(PROCESSING_FS / sim_fs * 1000))

    h_t = resample_poly(h_raw, up_ir, down_ir)

    # -----------------------------------------------------
    # 3. 20 kHz の時間軸上で高速FFT畳み込み
    # -----------------------------------------------------
    print(
        f"Applying Frequency Response (at {PROCESSING_FS}Hz Grid via FFT Convolution)..."
    )
    n_fft = len(h_t) + len(x_t) - 1

    H_f = np.fft.rfft(h_t, n=n_fft)
    X_f = np.fft.rfft(x_t, n=n_fft)
    Y_f = H_f * X_f
    y_t = np.fft.irfft(Y_f, n=n_fft)

    # -----------------------------------------------------
    # 【新規追加】 4. アップサンプリング前のスペクトログラム描写
    # -----------------------------------------------------
    print("Generating spectrogram for 20kHz convolved signal...")

    # 短時間フーリエ変換 (STFT) によるスペクトログラム計算
    f_axis, t_axis, Sxx = spectrogram(
        y_t, fs=PROCESSING_FS, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    # パワースペクトル密度をデシベル(dB)表現に変換
    Sxx_db = 10.0 * np.log10(Sxx + 1e-30)

    # 描画処理
    plt.figure(figsize=(11, 5))
    # pcolormeshで時間-周波数平面のマッピングを描画
    pcm = plt.pcolormesh(
        t_axis, f_axis, Sxx_db, shading="gouraud", cmap="magma", vmin=-100, vmax=0
    )

    plt.title(
        f"Spectrogram of Convolved Sound (Before 44.1kHz Resampling)\nSampling Rate: {PROCESSING_FS} Hz",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Frequency [Hz]", fontsize=10)
    plt.xlabel("Time [seconds]", fontsize=10)

    # 縦軸の範囲を20kHzのナイキスト周波数（10kHz）までに設定
    plt.ylim(0, PROCESSING_FS / 2)

    # カラーバーの追加（ダイナミックレンジのインジケータ）
    cbar = plt.colorbar(pcm, pad=0.02)
    cbar.set_label("Intensity [dB]", fontsize=10)

    plt.tight_layout()

    # 画像として保存しつつ画面に表示
    OUTPUT_SPEC_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_SPEC_PNG, dpi=300)
    print(f"  Spectrogram image saved to: {OUTPUT_SPEC_PNG}")
    plt.show()

    # -----------------------------------------------------
    # 5. 指定された 44.1 kHz へ正確にアップサンプリング
    # -----------------------------------------------------
    print(f"Upsampling final output to target audio rate ({FINAL_WAV_FS} Hz)...")
    gcd_out = np.gcd(PROCESSING_FS, FINAL_WAV_FS)
    up_out = int(FINAL_WAV_FS // gcd_out)
    down_out = int(PROCESSING_FS // gcd_out)

    y_final = resample_poly(y_t, up_out, down_out)

    # -----------------------------------------------------
    # 6. オーディオ正規化と16bit WAV書き出し
    # -----------------------------------------------------
    y_final -= np.mean(y_final)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final /= max_val

    audio_data = (y_final * 32767.0).astype(np.int16)

    print(f"Writing final 44.1kHz WAV to {OUTPUT_WAV}...")
    wavfile.write(OUTPUT_WAV, FINAL_WAV_FS, audio_data)
    print("Successfully finished all pipelines!")


if __name__ == "__main__":
    main()