# -*- coding: utf-8 -*-
"""
張力比 r を impulse 駆動と sound 駆動（実測弦入力）の両方で掃引するドライバ。sweep_report19.py の派生版。

置き場所: sanshin_analysis_pusg.py と同じフォルダ（gpu1 の /app/src）。本体は一切編集しない。

Jupyter での使い方:
    !python src/sweep_tension_impulse.py --dry              # 実行予定の一覧・出力名・sound 関連の定数を表示
    !python src/sweep_tension_impulse.py                    # 全 run（3 張力比 x impulse/sound = 6 run）を順に実行（前景）
    !python src/sweep_tension_impulse.py --only S08i,S08s   # 指定した run だけ（末尾 i = impulse, s = sound）
    !python src/sweep_tension_impulse.py --no-npy           # obs_pressure.npy も保存しない（SAVE_NPY=False）
    !tail -n 30 src/sweep_tension_impulse.log               # 進捗確認
  背景実行:
    import os; os.system("nohup python src/sweep_tension_impulse.py > src/sweep_tension_impulse.log 2>&1 &")
  止めるとき: !pkill -f sweep_tension_impulse.py

しくみ（sweep_report19.py と同じ）:
  - 1 run ごとに「モジュールを import → 定数を上書き → 出力名とパスを再計算 → main()」を子プロセスで行う。
  - <stem>_summary.json が既にある run は飛ばす。止まっても同じコマンドで続きから走る。
  - 出力名に input_mode が入るので、impulse と sound の run で名前は衝突しない。
  - 入力: run ごとに INPUT_MODE（"impulse" / "sound"）だけを上書きする。インパルス波形・音源ファイル・
    ゲインなどの入力用の設定は本体の既定値のまま（sound は report/14: GenSound1.txt、20 kHz、gain 5e-4、1 N）。
    --dry で本体の SOUND 関連の定数を表示するので、実行前に確認すること。

張力（report/31 スライド 6 の表）:
  全 run で幾何平均 T_g = sqrt(15000*12000) = 13416.4 N/m 固定、T_表 = T_g/sqrt(r)、T_裏 = T_g*sqrt(r)。
  report/31 の表では r = 1.00 だけ T = 13,500 N/m（T_g が 0.6 % 高い）だったが、ここでは r = 1 も
  T_g = 13416.4 N/m に合わせる。ここでは r = 0.80, 1.00, 1.25 の 3 条件を impulse / sound で回す。
  共通: 有限バッフル 8 mm、心 50x40 mm、駒 real、減衰 表裏とも 10（本体の既定値のまま）
"""
import argparse
import importlib.util
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "sanshin_analysis_pusg.py"

TG = math.sqrt(15000.0 * 12000.0)  # 13416.4 N/m

INPUT_MODES = {"i": "impulse", "s": "sound"}  # label の末尾 → INPUT_MODE

RATIOS = [
    ("S03", 0.80),    # 15000.0 / 12000.0
    ("S08", 1.00),    # 13416.4 / 13416.4（report/31 の表の 13,500 ではなく T_g に合わせる）
    ("S09", 1.25),    # 12000.0 / 15000.0
]

# (label, T_front, r, input_mode)。impulse を先に全部回してから sound
RUNS = [(label + suf, round(TG / math.sqrt(r), 1), r, mode)
        for suf, mode in INPUT_MODES.items() for label, r in RATIOS]

# 全 run で固定する本体の定数（本体の既定値と同じ。既定値が変わっていたら止める）
# INPUT_MODE は run ごとに上書きするのでここには入れない
EXPECTED_FIXED = {
    "KOMA_POSITION": "real",
    "BAFFLE_TYPE": "finite",
    "SIDE_WALL_THICKNESS_M": 0.008,
    "ENABLE_SHIN": True,
    "SHIN_X_M": 50e-3,
    "SHIN_Z_M": 40e-3,
    "MEMBRANE_DAMPING": 10.0,
    "BACK_MEMBRANE_DAMPING": 10.0,
    "MEMBRANE_THICKNESS_M": 0.0006,
    "MEMBRANE_DENSITY_VOL": 1150.0,
    "SIM_TIME": 0.30,
    "SAVE_MEMBRANE_HISTORY": True,
    "SAVE_FARFIELD_SURFACE": True,
}

# run ごとに上書きする出力設定
OUTPUT_OVERRIDES = {
    "SAVE_NPY": True,        # obs_pressure.npy を残すため True のまま
    "NUM_VOLUME_SAVES": 1,   # 体積は 1 フレームだけ（15 GB → 約 4 MB）
    "SAVE_JSON": True,
    "SAVE_PNG": True,
}


def load_target():
    spec = importlib.util.spec_from_file_location("sanshin_analysis_pusg", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__ ではないので main() は走らない
    return mod


def check_fixed(mod):
    bad = []
    for k, v in EXPECTED_FIXED.items():
        cur = getattr(mod, k)
        same = math.isclose(cur, v, rel_tol=0, abs_tol=1e-12) if isinstance(v, float) else cur == v
        if not same:
            bad.append(f"{k}: 本体 {cur!r} / 想定 {v!r}")
    if bad:
        raise SystemExit("本体の既定値が掃引の前提と違います（中止）:\n  " + "\n  ".join(bad))


def sound_constants(mod):
    """本体の sound 入力に関係しそうな定数（確認表示用）。"""
    return {k: getattr(mod, k) for k in dir(mod)
            if k.isupper() and "SOUND" in k and not callable(getattr(mod, k))}


def apply_params(mod, tension, ratio, input_mode):
    """定数を上書きし、import 時に計算済みの出力名・パスを作り直す。"""
    mod.INPUT_MODE = input_mode
    mod.MEMBRANE_TENSION = float(tension)
    mod.BACK_TENSION_RATIO = float(ratio)
    for k, v in OUTPUT_OVERRIDES.items():
        setattr(mod, k, v)
    stem = mod.build_output_stem(
        input_mode=mod.INPUT_MODE,
        tension_n_m=mod.MEMBRANE_TENSION,
        back_tension_ratio=mod.BACK_TENSION_RATIO,
        thickness_m=mod.MEMBRANE_THICKNESS_M,
        koma_position=mod.KOMA_POSITION,
        sim_time_sec=mod.SIM_TIME,
        enable_shin=mod.ENABLE_SHIN,
        shin_x_m=mod.SHIN_X_M,
        shin_z_m=mod.SHIN_Z_M,
        baffle_type=mod.BAFFLE_TYPE,
        side_wall_thickness_m=mod.SIDE_WALL_THICKNESS_M,
    )
    out = mod.DEFAULT_OUTPUT_DIR
    mod.OUTPUT_STEM = stem
    mod.RUN_NAME = stem
    mod.OUTPUT_NPY = out / f"{stem}.npy"
    mod.OUTPUT_JSON_PATH = out / f"{stem}_summary.json"
    mod.OUTPUT_PNG_PATH = out / f"{stem}_diagnostics.png"
    mod.OUTPUT_FARFIELD_SURFACE = out / f"{stem}_farfield_surface.json"
    mod.OUTPUT_MEMBRANE_HISTORY = out / f"{stem}_membrane_history.json"
    return stem


def find(label):
    for run in RUNS:
        if run[0] == label:
            return run
    raise SystemExit(f"unknown label: {label}")


def run_one(label, no_npy=False):
    """子プロセス側: 1 run だけ実行する。"""
    _, tension, ratio, input_mode = find(label)
    mod = load_target()
    check_fixed(mod)
    stem = apply_params(mod, tension, ratio, input_mode)
    if no_npy:
        mod.SAVE_NPY = False   # 体積 npy も obs_pressure.npy も保存しない
    print(f"[{label}] input={mod.INPUT_MODE}  T_front={tension:.1f} N/m  T_back={tension*ratio:.1f} N/m  "
          f"r={ratio}  stem={stem}", flush=True)
    mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help=argparse.SUPPRESS)  # 子プロセス用
    ap.add_argument("--only", help="実行する label をカンマ区切りで（例: S08i,S08s）")
    ap.add_argument("--dry", action="store_true", help="一覧と出力名を表示するだけ")
    ap.add_argument("--no-npy", action="store_true", help="SAVE_NPY=False（obs_pressure.npy も保存しない）")
    a = ap.parse_args()

    if a.one:
        run_one(a.one, a.no_npy)
        return

    runs = RUNS
    if a.only:
        want = [s.strip() for s in a.only.split(",") if s.strip()]
        runs = [find(w) for w in want]

    mod = load_target()
    check_fixed(mod)
    out_dir = mod.DEFAULT_OUTPUT_DIR
    default_mode = mod.INPUT_MODE
    stems = [apply_params(mod, t, r, m) for _, t, r, m in RUNS]
    if len(set(stems)) != len(stems):
        raise SystemExit("出力名が重複しています（中止）: " + ", ".join(stems))
    stem_of = {lab: s for (lab, _, _, _), s in zip(RUNS, stems)}

    print(f"INPUT_MODE は run ごとに上書き（本体の既定値 {default_mode!r}）")
    for k, v in sound_constants(mod).items():
        print(f"  {k} = {v!r}")
    print(f"T_g = {TG:.1f} N/m\n")
    print(f"{'label':5s} {'input':7s} {'T_front':>9s} {'T_back':>9s} {'r':>7s} {'sqrt(TfTb)':>10s}  {'status':6s}  stem")
    for lab, t, r, m in runs:
        done = (out_dir / f"{stem_of[lab]}_summary.json").exists()
        print(f"{lab:5s} {m:7s} {t:9.1f} {t*r:9.1f} {r:7.4f} {math.sqrt(t*t*r):10.1f}  "
              f"{'done' if done else 'todo':6s}  {stem_of[lab]}")
    print(f"出力先: {out_dir}", flush=True)
    if a.dry:
        return

    status = {}
    for lab, t, r, m in runs:
        if (out_dir / f"{stem_of[lab]}_summary.json").exists():
            status[lab] = "skip (done)"
            print(f"\n[{lab}] summary.json あり → skip", flush=True)
            continue
        print(f"\n[{lab}] start {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", lab]
        if a.no_npy:
            cmd.append("--no-npy")
        ret = subprocess.run(cmd)
        dt = (time.time() - t0) / 60
        status[lab] = f"ok ({dt:.0f} min)" if ret.returncode == 0 else f"FAILED rc={ret.returncode} ({dt:.0f} min)"
        print(f"[{lab}] {status[lab]}", flush=True)

    print("\n=== 結果 ===")
    for lab, s in status.items():
        print(f"{lab:5s} {s}")


if __name__ == "__main__":
    main()
