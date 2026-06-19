import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# =========================================================
# 設定
# =========================================================
# 1) 入力ファイル
FILE_IMP_NPY = Path("src/sanshin_force_imp_real_obs_pressure.npy")
FILE_IMP_JSON = Path("src/sanshin_force_imp_real_summary.json")
FILE_STRING_SOUND = Path("sound_source/GenSound1.txt")

# 2) 出力ファイル
OUTPUT_WAV = Path("src/sanshin_convolved_output_20k_base.wav")

# 3) 弦の振動データのサンプリングレートおよびターゲットIRサンプリングレート (Hz)
TARGET_IR_FS = 20000

# 4) 最終的なWAVファイルのサンプリングレート (再生互換用)
FINAL_WAV_FS = 44100


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
        raise FileNotFoundError("入力ファイルが見つかりません。パスを確認してください。")

    # インパルス応答のロードとDCオフセット除去
    h_raw = np.load(FILE_IMP_NPY).astype(np.float64)
    h_raw -= np.mean(h_raw)

    # シミュレーションの本来のサンプリングレート (約296kHz〜350kHz)
    dt_sim = load_simulation_dt(FILE_IMP_JSON)
    sim_fs = 1.0 / dt_sim

    # 弦の入力振動データのロード (20kHz)
    x_t = np.loadtxt(FILE_STRING_SOUND, dtype=np.float64)
    x_t -= np.mean(x_t)

    print(f"  Original Simulation FS: {sim_fs:.2f} Hz")
    print(f"  Original IR Length: {len(h_raw)} samples")
    print(f"  String Input Length: {len(x_t)} samples")

    # -----------------------------------------------------
    # 2. インパルス応答を 20 kHz にダウンサンプリング
    # -----------------------------------------------------
    print(f"Downsampling Impulse Response to {TARGET_IR_FS} Hz...")
    gcd_ir = np.gcd(int(round(sim_fs)), TARGET_IR_FS)
    up_ir = int(TARGET_IR_FS // gcd_ir)
    down_ir = int(round(sim_fs) // gcd_ir)

    # 比率が大きすぎる場合のセーフティガード
    if down_ir > 1000 or up_ir > 1000:
        down_ir = 1000
        up_ir = int(round(TARGET_IR_FS / sim_fs * 1000))

    # ここでIR自体を20kHzに落とす
    h_t = resample_poly(h_raw, up_ir, down_ir)
    print(f"  Downsampled IR Length: {len(h_t)} samples")

    # -----------------------------------------------------
    # 3. 20 kHz 同士での畳み込み（周波数応答特性の適用）
    # -----------------------------------------------------
    print("Applying Frequency Response (20kHz Base FFT Convolution)...")
    # 20kHz同士なので、FFT点数も大幅に削減される
    n_fft = len(h_t) + len(x_t) - 1

    H_f = np.fft.rfft(h_t, n=n_fft)
    X_f = np.fft.rfft(x_t, n=n_fft)

    # 周波数領域での乗算：Y(f) = H(f) * X(f)
    Y_f = H_f * X_f

    # 時間領域に逆変換：y(t) (サンプリングレートは 20kHz)
    y_t = np.fft.irfft(Y_f, n=n_fft)

    # -----------------------------------------------------
    # 4. オーディオプレイヤー互換のため 44.1 kHz へアップサンプリング
    # -----------------------------------------------------
    print(f"Upsampling final output to audio rate ({FINAL_WAV_FS} Hz)...")
    gcd_out = np.gcd(TARGET_IR_FS, FINAL_WAV_FS)
    up_out = int(FINAL_WAV_FS // gcd_out)
    down_out = int(TARGET_IR_FS // gcd_out)

    y_final = resample_poly(y_t, up_out, down_out)

    # -----------------------------------------------------
    # 5. 正規化と16bit WAV書き出し
    # -----------------------------------------------------
    y_final -= np.mean(y_final)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final /= max_val

    audio_data = (y_final * 32767.0).astype(np.int16)

    print(f"Writing WAV file to {OUTPUT_WAV}...")
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(OUTPUT_WAV, FINAL_WAV_FS, audio_data)
    print("Successfully generated output sound!")


if __name__ == "__main__":
    main()