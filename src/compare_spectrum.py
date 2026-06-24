import numpy as np
import matplotlib.pyplot as plt
import json
import zarr
from pathlib import Path
from scipy.signal import find_peaks


# =========================================================
# 0) 設定
# =========================================================
CONDITION_NAME = "tension ratio 1.0, -"

FILE_IMPULSE = "src/sanshin_force_imp_btr1_real_s0p003.zarr"
FILE_SOUND = "src/sanshin_force_sound_btr1_real_s0p003.zarr"
DEFAULT_SOUND_FILE = "sound_source/GenSound1.txt"

OBS_DISTANCE_CM = 30.0
FREQ_MAX_HZ = 3000.0

NORM_FMAX_HZ = 3000.0
NORM_OFFSET_DB = -20.0

IGNORE_FREQ = 300.0
IGNORE_TOLERANCE = 30.0
PEAK_MIN_DB = -45.0
PEAK_MIN_SEP_HZ = 100.0
TOP_K_MODES = 4

DELTA_LOCAL_WINDOW_HZ = 150.0
SMOOTH_HZ = 40.0
REL_BASELINE_MIN_HZ = 150.0
REL_BASELINE_MAX_HZ = 3000.0
DELTA_YMIN = -60.0
DELTA_YMAX = 60.0

SAVE_NAME = "fig3_spectrum_analysis.png"
SAVE_PNG = True
SHOW_FIGURE = True
SAVE_DPI = 300
SAVE_PAD_IN = 0.35
FIG_W = 11.5
FIG_H = 8.2
SUPTITLE_Y = 0.985
TIGHT_RECT = (0, 0, 1, 0.95)

SPECTRUM_YMIN = -120
SPECTRUM_YMAX = 20
IMPULSE_FILL_COLOR = "gray"
IMPULSE_FILL_ALPHA = 0.15
INPUT_COLOR = "tab:blue"
OUTPUT_COLOR = "tab:orange"
DELTA_COLOR = "tab:green"
MARKER_COLOR = "tab:red"
GRID_ALPHA = 0.6
LEGEND_LOC = "lower left"
LEGEND_FONTSIZE = 9


# =========================================================
# 1) 入出力・時間/空間刻み
# =========================================================
def load_pkg(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    loaded = np.asarray(zarr.open(str(path), mode="r"))

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
    obs_z = int(np.clip(cz - dist_grid, 0, nz - 1))
    wave = saved_data[:, cx, cy, obs_z].astype(np.float64)
    wave -= np.mean(wave)
    return wave, dx, (cx, cy, obs_z)


def build_sound_input_series(params, nt_saved, dt_frame):
    sound_file = params.get("sound_file", None) or DEFAULT_SOUND_FILE
    src_fs = float(params.get("src_fs", 44100.0))
    gain = float(params.get("sound_gain", 1.0))

    if not Path(sound_file).exists():
        print(f"Warning: Sound source {sound_file} not found. Using silence.")
        return np.zeros(nt_saved)

    src = np.loadtxt(sound_file, dtype=np.float64)
    t_src = np.arange(len(src), dtype=np.float64) / src_fs
    t = np.arange(nt_saved, dtype=np.float64) * dt_frame
    x = np.interp(t, t_src, src, left=0.0, right=0.0)

    mx = np.max(np.abs(x))
    if mx > 0:
        x = x / mx
    x = x * gain
    x -= np.mean(x)
    return x


# =========================================================
# 2) スペクトル・正規化
# =========================================================
def spectrum_mag(wave, dt):
    N = len(wave)
    fs = 1.0 / dt
    w = np.hanning(N)
    X = np.fft.rfft(wave * w)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(X)
    return freqs, mag, fs


def calc_rms_ref(freqs, mag, f_max):
    mask = (freqs <= f_max) & (freqs >= 10.0)
    if not np.any(mask):
        return np.max(mag) if len(mag) > 0 else 1.0
    rms_val = np.sqrt(np.mean(mag[mask] ** 2))
    return rms_val


def moving_average_hz(y, freqs, smooth_hz):
    df = freqs[1] - freqs[0]
    bins = max(1, int(round(smooth_hz / df)))
    if bins <= 1:
        return y.copy(), bins, df
    kernel = np.ones(bins, dtype=np.float64) / bins
    y_sm = np.convolve(y, kernel, mode="same")
    return y_sm, bins, df


def db20(x, ref=1.0):
    eps = 1e-30
    return 20.0 * np.log10((x + eps) / (ref + eps))


# =========================================================
# 3) モード検出・ピーク取得
# =========================================================
def find_impulse_modes_topk(freqs, mag, fs, top_k=4, fmin=0.0, fmax=3000.0):
    mag_db = db20(mag, ref=np.max(mag))
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return []

    freqs_b = freqs[band_mask]
    mag_db_b = mag_db[band_mask]
    df = freqs_b[1] - freqs_b[0]
    min_dist_bins = max(1, int(round(PEAK_MIN_SEP_HZ / df)))

    peaks, _ = find_peaks(mag_db_b, height=PEAK_MIN_DB, distance=min_dist_bins)

    cand = []
    for p in peaks:
        f = float(freqs_b[p])
        if abs(f - IGNORE_FREQ) <= IGNORE_TOLERANCE:
            continue
        cand.append((f, float(mag_db_b[p])))

    cand.sort(key=lambda x: x[1], reverse=True)

    modes = []
    for f, _ in cand:
        if all(abs(f - g) > 30.0 for g in modes):
            modes.append(f)
        if len(modes) >= top_k:
            break

    return sorted(modes)


def pick_local_peak(freqs, y, f0, window_hz):
    fmin = f0 - window_hz
    fmax = f0 + window_hz
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None, None
    idxs = np.where(mask)[0]
    k = idxs[np.argmax(y[idxs])]
    return float(freqs[k]), float(y[k])


def place_label_safely(ax, x, y, text, ypad=3.0):
    ymin, ymax = ax.get_ylim()
    if y > ymax - 10.0:
        yy = y - ypad
        va = "top"
    elif y < ymin + 10.0:
        yy = y + ypad
        va = "bottom"
    else:
        yy = y + ypad
        va = "bottom"
    ax.text(x, yy, text, ha="center", va=va, fontsize=8, clip_on=False)


# =========================================================
# 4) 描画
# =========================================================
def plot_results(f_imp, imp_db_sm, f_in, in_db_sm, f_out, out_db_sm, Delta_rel, modes, fmax_plot):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)

    imp_plot_mask = (f_imp <= fmax_plot)
    axes[0].fill_between(
        f_imp[imp_plot_mask],
        SPECTRUM_YMIN,
        imp_db_sm[imp_plot_mask],
        color=IMPULSE_FILL_COLOR,
        alpha=IMPULSE_FILL_ALPHA,
        label="Impulse Response (Membrane Resonance)",
    )

    axes[0].plot(f_in, in_db_sm, label="Input (sanshin sound)", color=INPUT_COLOR, linewidth=1.5, alpha=0.8)
    axes[0].plot(f_out, out_db_sm, label="Output (radiation from the membrane by FDTD)", color=OUTPUT_COLOR, linewidth=1.5)

    axes[0].set_ylabel("Magnitude [dB]\n(RMS Normalized)")
    axes[0].set_ylim(SPECTRUM_YMIN, SPECTRUM_YMAX)
    axes[0].grid(True, linestyle=":", alpha=GRID_ALPHA)
    axes[0].legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE)

    axes[1].plot(f_out, Delta_rel, label=r"Transfer Function $\Delta_{rel}$ (Output / Input)", color=DELTA_COLOR, linewidth=1.5)
    axes[1].axhline(0.0, linestyle="-", color="gray", linewidth=0.8)
    axes[1].set_ylabel("Relative Enhancement [dB]")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].grid(True, linestyle=":", alpha=GRID_ALPHA)
    axes[1].set_ylim(DELTA_YMIN, DELTA_YMAX)
    axes[1].legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE)

    for i, m in enumerate(modes, start=1):
        if 0 < m < fmax_plot:
            axes[0].axvline(m, linestyle="--", color="gray", linewidth=1.0, alpha=0.5)
            axes[1].axvline(m, linestyle="--", color="gray", linewidth=1.0, alpha=0.5)
            axes[0].text(m, 10, f"M{i}", rotation=90, va="top", ha="right", fontsize=9, color="gray", fontweight="bold")

            f_d, v_d = pick_local_peak(f_out, Delta_rel, m, DELTA_LOCAL_WINDOW_HZ)
            if f_d is not None and f_d < fmax_plot:
                axes[1].plot([f_d], [v_d], marker="o", color=MARKER_COLOR, markersize=5)
                shift_hz = f_d - m
                label_text = f"Peak\nShift: {shift_hz:+.0f}Hz"
                place_label_safely(axes[1], x=f_d, y=v_d, text=label_text)

    axes[0].set_xlim(0, fmax_plot)
    fig.suptitle(f"Spectral Analysis: {CONDITION_NAME}", y=SUPTITLE_Y, fontsize=14)
    plt.tight_layout(rect=TIGHT_RECT)
    return fig


# =========================================================
# 5) メイン処理
# =========================================================
def main():
    print("Loading data...")
    imp_data, imp_params = load_pkg(FILE_IMPULSE)
    snd_data, snd_params = load_pkg(FILE_SOUND)

    dt_imp = get_dt_frame(imp_params, imp_data.shape[0])
    dt_snd = get_dt_frame(snd_params, snd_data.shape[0])

    imp_wave, _, _ = extract_obs_wave(imp_data, imp_params, OBS_DISTANCE_CM)
    out_wave, _, _ = extract_obs_wave(snd_data, snd_params, OBS_DISTANCE_CM)
    in_wave = build_sound_input_series(snd_params, nt_saved=len(out_wave), dt_frame=dt_snd)

    f_imp, mag_imp, fs_imp = spectrum_mag(imp_wave, dt_imp)
    f_out, mag_out, fs_out = spectrum_mag(out_wave, dt_snd)
    f_in, mag_in, _ = spectrum_mag(in_wave, dt_snd)

    fmax_mode = min(FREQ_MAX_HZ, fs_imp / 2)
    modes = find_impulse_modes_topk(f_imp, mag_imp, fs_imp, top_k=TOP_K_MODES, fmin=0.0, fmax=fmax_mode)
    print(f"Modes identified: {[f'{m:.0f} Hz' for m in modes]}")

    print(f"Normalizing by RMS power in range 0-{NORM_FMAX_HZ} Hz...")
    ref_in = calc_rms_ref(f_in, mag_in, NORM_FMAX_HZ)
    ref_out = calc_rms_ref(f_out, mag_out, NORM_FMAX_HZ)
    ref_imp = calc_rms_ref(f_imp, mag_imp, NORM_FMAX_HZ)

    in_db = 20.0 * np.log10((mag_in + 1e-30) / ref_in) + NORM_OFFSET_DB
    out_db = 20.0 * np.log10((mag_out + 1e-30) / ref_out) + NORM_OFFSET_DB
    imp_db = 20.0 * np.log10((mag_imp + 1e-30) / ref_imp) + NORM_OFFSET_DB

    in_db_sm, _, _ = moving_average_hz(in_db, f_in, SMOOTH_HZ)
    out_db_sm, _, _ = moving_average_hz(out_db, f_out, SMOOTH_HZ)
    imp_db_sm, _, _ = moving_average_hz(imp_db, f_imp, SMOOTH_HZ)

    H_db_abs = 20.0 * np.log10((mag_out + 1e-30) / (mag_in + 1e-30))
    H_db_abs_sm, _, _ = moving_average_hz(H_db_abs, f_out, SMOOTH_HZ)

    base_mask = (f_out >= REL_BASELINE_MIN_HZ) & (f_out <= min(REL_BASELINE_MAX_HZ, fs_out / 2))
    base = np.mean(H_db_abs_sm[base_mask]) if np.any(base_mask) else 0.0
    Delta_rel = H_db_abs_sm - base
    print(f"Baseline removed: {base:.2f} dB")

    fmax_plot = min(FREQ_MAX_HZ, fs_out / 2)
    fig = plot_results(f_imp, imp_db_sm, f_in, in_db_sm, f_out, out_db_sm, Delta_rel, modes, fmax_plot)

    if SAVE_PNG:
        plt.savefig(SAVE_NAME, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
        print(f"\nSaved figure to: {SAVE_NAME}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
