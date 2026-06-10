from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.io import wavfile

SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート/sound を出力先に指定
DEFAULT_OUT_DIR = SCRIPT_DIR / "sound"


# =========================================================
# 1) compare_spectrum.py から移植したデータロード・位置計算ロジック
# =========================================================
def load_pkg(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    loaded = np.load(path, allow_pickle=True)
    
    if loaded.shape == ():
        pkg = loaded.item()
        return pkg["data"], pkg.get("params", {})
        
    params = {}
    json_path = path.with_name(path.stem + "_summary.json")
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "summary" in meta and "params" in meta["summary"]:
                    params = meta["summary"]["params"]
                elif "params" in meta:
                    params = meta["params"]
            print(f"Loaded parameters from {json_path.name}")
        except Exception as e:
            print(f"Warning: Failed to load JSON parameters from {json_path}: {e}")
    else:
        print(f"Warning: Summary JSON not found at {json_path}. Parameters will be empty.")
            
    return loaded, params


def get_dx_saved(params, nx_saved):
    nx = int(params.get("nx", 500))
    pml = int(params.get("pml_width", 50))
    dx_sim = float(params.get("dx", 0.002))
    phys_width = dx_sim * (nx - 2 * pml)
    return phys_width / nx_saved


def get_dt_frame(params, nt_saved):
    if "dt" in params and "volume_save_interval" in params:
        return float(params["dt"]) * int(params["volume_save_interval"])
    if "sim_time" in params and "nt" in params and "volume_save_interval" in params:
        dt = float(params["sim_time"]) / int(params["nt"])
        return dt * int(params["volume_save_interval"])
    if "sim_time" in params and nt_saved > 1:
        return float(params["sim_time"]) / float(nt_saved - 1)
    raise ValueError("Cannot determine dt_frame from params.")


def extract_obs_wave(saved_data, params, obs_distance_cm):
    nt, nx, ny, nz = saved_data.shape
    dx = get_dx_saved(params, nx)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    dist_grid = int((obs_distance_cm / 100.0) / dx)
    obs_z = int(np.clip(cz + dist_grid, 0, nz - 1))
    wave = saved_data[:, cx, cy, obs_z].astype(np.float64)
    wave -= np.mean(wave)  # DCオフセット除去
    return wave, dx, (cx, cy, obs_z)


# =========================================================
# 2) コマンドライン引数の設定
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract microphone observation wave from 4D volume data and export as a WAV file."
    )
    p.add_argument("input_path", type=Path, help="Path to the simulation NPY file (e.g., src/sanshin_force_sound_real.npy)")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for audio (default: project_root/sound).")
    p.add_argument("--out-name", type=str, default=None, help="Output WAV file name (default: derived from input file stem).")
    p.add_argument("--distance-cm", type=float, default=30.0, help="Microphone distance from center in cm (default: 30.0).")
    p.add_argument("--sr", type=int, default=44100, help="Target sample rate for output WAV (default: 44100 Hz).")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True, help="Normalize peak amplitude to -0.1 dBFS (0.99).")
    p.add_argument("--gain", type=float, default=1.0, help="Linear gain multiplier applied before normalization.")
    return p.parse_args()


# =========================================================
# 3) メイン処理
# =========================================================
def main() -> None:
    args = parse_args()
    
    # 出力パスの設定
    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_name = args.out_name if args.out_name is not None else f"{args.input_path.stem}_dist{int(args.distance_cm)}cm.wav"
    if not out_name.endswith(".wav"):
        out_name += ".wav"
    output_wav_path = out_dir / out_name

    # データロード
    print(f"Loading package: {args.input_path}")
    saved_data, params = load_pkg(args.input_path)
    
    if saved_data.ndim != 4:
        raise ValueError(f"Expected 4D array (nt, nx, ny, nz), but got shape {saved_data.shape}")

    # パラメータの解析
    nt_saved = saved_data.shape[0]
    dt_frame = get_dt_frame(params, nt_saved)
    fs_sim = 1.0 / dt_frame
    
    # 30cm (指定距離) のマイク位置から音圧波形をシーク
    print(f"Extracting observation wave at distance: {args.distance_cm} cm")
    wave, dx, (cx, cy, obs_z) = extract_obs_wave(saved_data, params, args.distance_cm)
    print(f"-> Grid coordinate resolved: Center=({cx}, {cy}), Obs_Z={obs_z} (dx_saved={dx*1000.0:.2f} mm)")
    print(f"-> Simulation saved sample rate: {fs_sim:.2f} Hz ({nt_saved} frames)")

    # オーディオ標準周波数 (44.1kHz等) への線形補間リサンプリング
    t_src = np.arange(nt_saved, dtype=np.float64) * dt_frame
    duration = t_src[-1] if t_src.size > 0 else 0.0
    
    nt_target = int(math.ceil(duration * args.sr))
    t_target = np.arange(nt_target, dtype=np.float64) / float(args.sr)
    
    print(f"Resampling to target sample rate: {args.sr} Hz (Duration: {duration:.3f} s)")
    wave *= args.gain
    audio_resampled = np.interp(t_target, t_src, wave)

    # ノーマライズ処理
    if args.normalize:
        peak = np.max(np.abs(audio_resampled))
        if peak > 0.0:
            audio_resampled = (audio_resampled / peak) * 0.99
            print(f"-> Peak normalized to -0.1 dBFS (Original peak: {peak:.4g})")
            
    # クライマックスのクリッピング回避と 16-bit PCM へのキャスト
    audio_clipped = np.clip(audio_resampled, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    # WAV保存
    print(f"Writing audio to: {output_wav_path}")
    wavfile.write(output_wav_path, args.sr, audio_int16)
    print("Export completed successfully.")


if __name__ == "__main__":
    main()