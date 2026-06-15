import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly

# =========================================================
# 設定 (compare_spectrum.py の構造に準拠)
# =========================================================
# 変換したいデータのベース名（条件に合わせて変更してください）
# インパルス駆動の場合: "sanshin_force_imp_real"
# 音源駆動の場合: "sanshin_force_sound_real"
STEM_NAME = "sanshin_force_sound_real" 

DIR_SRC = Path("src")
FILE_NPY = DIR_SRC / f"{STEM_NAME}_obs_pressure.npy"
FILE_JSON = DIR_SRC / f"{STEM_NAME}_summary.json"
OUTPUT_WAV = DIR_SRC / f"{STEM_NAME}_obs.wav"

TARGET_FS = 44100  # 一般的なプレイヤーで再生可能なサンプリングレート (Hz)
NORMALIZE = True   # クリッピングを防ぐために最大振幅を1.0に正規化するか

def main():
    print(f"Loading data from {FILE_NPY}...")
    if not FILE_NPY.exists():
        raise FileNotFoundError(
            f"波形ファイルが見つかりません: {FILE_NPY}\n"
            f"先に sanshin_analysis_pusg.py を実行してデータを生成してください。"
        )
    
    # 1) 音圧データの読み込みとオフセット（直流成分）除去
    wave = np.load(FILE_NPY).astype(np.float64)
    wave -= np.mean(wave)
    
    # 2) シミュレーションのサンプリングレート(fs)をJSONから自動取得
    if not FILE_JSON.exists():
        raise FileNotFoundError(f"メタデータファイルが見つかりません: {FILE_JSON}")
        
    with open(FILE_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
        params = meta.get("summary", {}).get("params", {}) if "summary" in meta else meta.get("params", {})
        dt = float(params.get("dt"))
        
    sim_fs = 1.0 / dt
    print(f"Simulation Sampling Rate: {sim_fs:.2f} Hz (dt: {dt:.3e} s)")
    print(f"Total samples: {len(wave)} ({len(wave)/sim_fs:.3f} seconds)")

    # 3) ダウンサンプリング処理 (約350kHz -> 44.1kHz)
    print(f"Resampling from {sim_fs:.2f} Hz to {TARGET_FS} Hz...")
    
    # 合理的な整数比のアップ/ダウンサンプリング比を計算
    gcd = np.gcd(int(round(sim_fs)), TARGET_FS)
    up = int(TARGET_FS // gcd)
    down = int(round(sim_fs) // gcd)
    
    # 比率が大きすぎてメモリや計算負荷が跳ね上がるのを防ぐためのセーフティ
    if down > 1000 or up > 1000:
        down = 1000
        up = int(round(TARGET_FS / sim_fs * 1000))
        
    wave_resampled = resample_poly(wave, up, down)
    
    # 4) 正規化 (音割れを防ぎ、適切な音量にする)
    if NORMALIZE:
        max_val = np.max(np.abs(wave_resampled))
        if max_val > 0:
            wave_resampled = wave_resampled / max_val
            print("Audio amplitude normalized to max 1.0.")
            
    # 16bit PCM 形式 (-32768 ～ 32767) に変換
    audio_data = (wave_resampled * 32767.0).astype(np.int16)
    
    # 5) WAVファイルの書き出し
    print(f"Writing WAV file to {OUTPUT_WAV}...")
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(OUTPUT_WAV, TARGET_FS, audio_data)
    print("Successfully converted to WAV!")

if __name__ == "__main__":
    main()