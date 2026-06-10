from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import analyze_pu_backmem_npy_farfield as pu


if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd().resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
BUNDLE_DIR = SCRIPT_DIR.parent

DEFAULT_IMP_NPY = DATA_DIR / "sanshin_force_imp_real.npy"
DEFAULT_SOUND_NPY = DATA_DIR / "sanshin_force_sound_real.npy"
DEFAULT_SOUND_FILE = BUNDLE_DIR / "sound_source" / "GenSound1.txt"
DEFAULT_OUT_DIR = SCRIPT_DIR / "spectrum_imp_sound_real"

OBS_RADIUS_M = 0.30
FREQ_MAX_HZ = 3000.0
NORM_FMAX_HZ = 3000.0
NORM_OFFSET_DB = -20.0
ZERO_PAD_FACTOR = 8
SMOOTH_HZ = 40.0
INPUT_SMOOTH_HZ = 5.0
REL_BASELINE_MIN_HZ = 150.0
REL_BASELINE_MAX_HZ = 3000.0
DELTA_YMIN = -30.0
DELTA_YMAX = 30.0

SAVE_DPI = 300
FIG_W = 11.5
FIG_H = 10.2
SAVE_PAD_IN = 0.35
TIGHT_RECT = (0, 0, 1, 1)

FONT_FAMILY = "serif"
FONT_SIZE_BASE = 16
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICK = 14
LEGEND_SCALE = 0.70
LEGEND_FONTSIZE = max(8, int(round(FONT_SIZE_LEGEND * LEGEND_SCALE)))


@dataclass
class VolumeCase:
    path: Path
    data: np.ndarray
    params: dict
    dt_saved: float
    wave: np.ndarray
    freqs: np.ndarray
    mag: np.ndarray
    db: np.ndarray
    db_sm: np.ndarray


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_BASE,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "figure.titlesize": FONT_SIZE_TITLE,
        }
    )
    return plt


def spectrum_mag(wave: np.ndarray, dt: float, zp_factor: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - np.mean(wave)
    n = len(wave)
    nfft = max(n, int(n * max(1, zp_factor)))
    spec = np.fft.rfft(wave * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    return freqs, np.abs(spec), 1.0 / dt


def calc_rms_ref(freqs: np.ndarray, mag: np.ndarray, f_max: float) -> float:
    mask = (freqs >= 10.0) & (freqs <= f_max)
    if not np.any(mask):
        return float(np.max(mag)) if mag.size else 1.0
    return float(np.sqrt(np.mean(np.square(mag[mask]))))


def db20(x: np.ndarray, ref: float = 1.0) -> np.ndarray:
    eps = 1e-30
    return 20.0 * np.log10((x + eps) / (ref + eps))


def moving_average_hz(y: np.ndarray, freqs: np.ndarray, smooth_hz: float) -> tuple[np.ndarray, int, float]:
    if smooth_hz <= 0.0 or len(freqs) < 2:
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        return y.copy(), 1, df
    df = float(freqs[1] - freqs[0])
    bins = max(1, int(round(float(smooth_hz) / df)))
    if bins <= 1:
        return y.copy(), 1, df
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(y, kernel, mode="same"), bins, df


def load_volume_case(path: Path, radius_m: float, zp_factor: int, smooth_hz: float) -> VolumeCase:
    pkg = pu.load_volume_with_metadata(path)
    data = pkg["data"]
    params = pkg["params"]
    dt_saved = pu.saved_dt(params, pkg["t"], int(data.shape[0]))
    wave = pu.observation_wave(data, params, radius_m)
    freqs, mag, _ = spectrum_mag(wave, dt_saved, zp_factor)
    ref = calc_rms_ref(freqs, mag, NORM_FMAX_HZ)
    db = db20(mag, ref) + NORM_OFFSET_DB
    db_sm, _, _ = moving_average_hz(db, freqs, smooth_hz)
    return VolumeCase(path, data, params, dt_saved, wave, freqs, mag, db, db_sm)


def resolve_sound_file(params: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"sound file not found: {explicit}")
        return explicit

    drive_meta = params.get("drive_meta", {})
    candidates: list[Path] = []
    meta_path = drive_meta.get("sound_file")
    if meta_path:
        candidates.append(Path(str(meta_path)))
        candidates.append(BUNDLE_DIR / "sound_source" / Path(str(meta_path)).name)
    candidates.append(DEFAULT_SOUND_FILE)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("GenSound1.txt was not found in expected local paths.")


def build_gensound_drive(params: dict, nt_saved: int, dt_saved: float, sound_file: Path | None) -> np.ndarray:
    drive_meta = params.get("drive_meta", {})
    path = resolve_sound_file(params, sound_file)
    src = np.loadtxt(path, dtype=np.float64)
    if src.ndim != 1:
        src = np.ravel(src)
    peak = float(np.max(np.abs(src))) if src.size else 0.0
    if peak > 0.0:
        src = src / peak

    src_fs = float(drive_meta.get("sound_sample_rate_hz", 20000.0))
    if bool(drive_meta.get("sound_onset_detect", True)):
        thresh = float(drive_meta.get("sound_onset_thresh", 0.02))
        search_n = src.size
        mask = np.abs(src[:search_n]) >= thresh
        if np.any(mask):
            src = src[int(np.argmax(mask)) :]

    t_src = np.arange(src.size, dtype=np.float64) / src_fs
    t_saved = np.arange(nt_saved, dtype=np.float64) * dt_saved
    drive = np.interp(t_saved, t_src, src, left=0.0, right=0.0)
    drive *= float(drive_meta.get("sound_gain", 1.0))

    if str(drive_meta.get("sound_drive_mode", "gated")).lower() == "gated":
        apply_steps = min(nt_saved, max(1, int(round(float(drive_meta.get("sound_apply_sec", 0.03)) / dt_saved))))
        fade_steps = max(0, int(round(float(drive_meta.get("sound_fade_sec", 0.005)) / dt_saved)))
        gated = np.zeros_like(drive)
        envelope = np.ones(apply_steps, dtype=np.float64)
        if fade_steps > 0 and fade_steps * 2 < apply_steps:
            fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_steps, dtype=np.float64)))
            envelope[-fade_steps:] *= fade
        gated[:apply_steps] = drive[:apply_steps] * envelope
        drive = gated

    drive -= np.mean(drive)
    return drive


def build_measured_string_spectrum(params: dict, sound_file: Path | None, smooth_hz: float, zp_factor: int):
    drive_meta = params.get("drive_meta", {})
    path = resolve_sound_file(params, sound_file)
    src = np.loadtxt(path, dtype=np.float64)
    if src.ndim != 1:
        src = np.ravel(src)
    peak = float(np.max(np.abs(src))) if src.size else 0.0
    if peak > 0.0:
        src = src / peak
    if bool(drive_meta.get("sound_onset_detect", True)):
        thresh = float(drive_meta.get("sound_onset_thresh", 0.02))
        mask = np.abs(src) >= thresh
        if np.any(mask):
            src = src[int(np.argmax(mask)) :]
    sample_rate_hz = float(drive_meta.get("sound_sample_rate_hz", 20000.0))
    freqs, mag, _ = spectrum_mag(src, 1.0 / sample_rate_hz, zp_factor)
    ref = float(np.max(mag)) if mag.size else 1.0
    db = db20(mag, ref)
    db_sm, _, _ = moving_average_hz(db, freqs, smooth_hz)
    return freqs, db_sm


def theory_mode_freqs(params: dict, pairs: str) -> list[float]:
    modes: list[float] = []
    for m, n in pu.parse_mode_pairs(pairs):
        f_theory = pu.front_membrane_theory_freq(params, m, n)
        modes.append(float(f_theory))
    return modes


def save_csv(path: Path, f_out: np.ndarray, in_db: np.ndarray, out_db: np.ndarray, imp_db: np.ndarray, delta: np.ndarray, fmax: float) -> None:
    mask = f_out <= fmax
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "gensound_input_db_sm", "sound_output_db_sm", "impulse_db_sm", "delta_rel_db"])
        for row in zip(f_out[mask], in_db[mask], out_db[mask], imp_db[mask], delta[mask]):
            writer.writerow([f"{float(v):.6f}" for v in row])


def draw_spectrum_figure(
    plt,
    f_imp: np.ndarray,
    imp_db: np.ndarray,
    f_input_ref: np.ndarray,
    input_ref_db: np.ndarray,
    f_out: np.ndarray,
    out_db: np.ndarray,
    delta_rel: np.ndarray,
    modes: list[float],
    xlim: tuple[float, float],
):
    lo, hi = float(xlim[0]), float(xlim[1])
    mask_imp = (f_imp >= lo) & (f_imp <= hi)
    mask_input = (f_input_ref >= lo) & (f_input_ref <= hi)
    mask_out = (f_out >= lo) & (f_out <= hi)

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)
    axes[0].plot(
        f_input_ref[mask_input],
        input_ref_db[mask_input],
        label="Measured first-string displacement near the bridge",
        color="tab:blue",
        linewidth=1.1,
    )
    axes[0].set_ylabel("Input [dB re peak]")
    axes[0].set_ylim(-90.0, 3.0)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.85")
    axes[0].set_title("(a) Input Spectrum", pad=8)

    axes[1].fill_between(
        f_imp[mask_imp],
        -120.0,
        imp_db[mask_imp],
        color="gray",
        alpha=0.15,
        label="Impulse Response (Membrane Resonance)",
    )
    axes[1].plot(f_out[mask_out], out_db[mask_out], label="Sound output (FDTD)", color="tab:orange", linewidth=1.5)
    axes[1].set_ylabel("Response [dB]")
    axes[1].set_ylim(-120.0, 0.0)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.85")
    axes[1].set_title("(b) FDTD Response Spectra", pad=8)

    axes[2].plot(f_out[mask_out], delta_rel[mask_out], label=r"Relative enhancement $\Delta_{rel}(f)$", color="tab:green", linewidth=1.5)
    axes[2].axhline(0.0, linestyle="-", color="gray", linewidth=0.8)
    axes[2].set_ylabel(r"$\Delta_{rel}(f)$ [dB]")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].set_ylim(DELTA_YMIN, DELTA_YMAX)
    axes[2].legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
    axes[2].set_title(r"(c) Relative enhancement $\Delta_{rel}(f)$", pad=8)

    for i, mode_freq in enumerate(modes, start=1):
        if lo < mode_freq < hi:
            axes[1].axvline(mode_freq, linestyle="--", color="gray", linewidth=1.0, alpha=0.5)
            axes[2].axvline(mode_freq, linestyle="--", color="gray", linewidth=1.0, alpha=0.5)
            axes[1].text(
                mode_freq,
                0.98,
                f"M{i}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=FONT_SIZE_TICK,
                color="gray",
                fontweight="bold",
                transform=axes[1].get_xaxis_transform(),
            )

    axes[0].set_xlim(lo, hi)
    fig.tight_layout(rect=TIGHT_RECT)
    return fig


def plot_analysis(
    imp: VolumeCase,
    snd: VolumeCase,
    sound_file: Path | None,
    out_dir: Path,
    out_stem: str,
    fmax_hz: float,
    smooth_hz: float,
    zp_factor: int,
    mode_pairs: str,
    dpi: int,
) -> tuple[Path, Path]:
    plt = require_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    in_wave = build_gensound_drive(snd.params, len(snd.wave), snd.dt_saved, sound_file)
    f_in, mag_in, _ = spectrum_mag(in_wave, snd.dt_saved, zp_factor)
    ref_in = calc_rms_ref(f_in, mag_in, NORM_FMAX_HZ)
    in_db = db20(mag_in, ref_in) + NORM_OFFSET_DB
    in_db_sm, _, _ = moving_average_hz(in_db, f_in, INPUT_SMOOTH_HZ)
    f_input_ref, input_ref_db = build_measured_string_spectrum(snd.params, sound_file, INPUT_SMOOTH_HZ, zp_factor)

    fmax_plot = min(fmax_hz, snd.freqs[-1], imp.freqs[-1], f_in[-1])
    f_out = snd.freqs
    f_imp = imp.freqs
    in_db_on_out = np.interp(f_out, f_in, in_db_sm, left=in_db_sm[0], right=in_db_sm[-1])
    imp_db_on_out = np.interp(f_out, f_imp, imp.db_sm, left=imp.db_sm[0], right=imp.db_sm[-1])

    mag_in_on_out = np.interp(f_out, f_in, mag_in, left=mag_in[0], right=mag_in[-1])
    h_db = db20(snd.mag, mag_in_on_out)
    h_db_sm, _, _ = moving_average_hz(h_db, f_out, smooth_hz)
    base_mask = (f_out >= REL_BASELINE_MIN_HZ) & (f_out <= min(REL_BASELINE_MAX_HZ, fmax_plot))
    base = float(np.mean(h_db_sm[base_mask])) if np.any(base_mask) else 0.0
    delta_rel = h_db_sm - base

    modes = theory_mode_freqs(imp.params, mode_pairs)
    print(f"Theoretical mode frequencies: {[f'{m:.0f} Hz' for m in modes]}")
    print(f"Baseline removed from H(f): {base:.2f} dB")

    fig = draw_spectrum_figure(
        plt,
        f_imp,
        imp.db_sm,
        f_input_ref,
        input_ref_db,
        f_out,
        snd.db_sm,
        delta_rel,
        modes,
        (0.0, fmax_plot),
    )

    png_path = out_dir / f"{out_stem}.png"
    csv_path = out_dir / f"{out_stem}.csv"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
    plt.close(fig)
    save_csv(csv_path, f_out, in_db_on_out, snd.db_sm, imp_db_on_out, delta_rel, fmax_plot)
    return png_path, csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare 15kN impulse and 15kN GenSound-driven output to show emphasized frequency bands."
    )
    p.add_argument("--imp-npy", type=Path, default=DEFAULT_IMP_NPY)
    p.add_argument("--sound-npy", type=Path, default=DEFAULT_SOUND_NPY)
    p.add_argument("--sound-file", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out-stem", default="imp_sound_15k_gensound_enhancement")
    p.add_argument("--radius-m", type=float, default=OBS_RADIUS_M)
    p.add_argument("--fmax-hz", type=float, default=FREQ_MAX_HZ)
    p.add_argument("--smooth-hz", type=float, default=SMOOTH_HZ)
    p.add_argument("--zp-factor", type=int, default=ZERO_PAD_FACTOR)
    p.add_argument("--mode-pairs", default="1,1;1,3;1,5;1,7")
    p.add_argument("--dpi", type=int, default=SAVE_DPI)
    args, _unknown = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    print("Loading 15kN impulse and sound data...")
    imp = load_volume_case(args.imp_npy, args.radius_m, args.zp_factor, args.smooth_hz)
    snd = load_volume_case(args.sound_npy, args.radius_m, args.zp_factor, args.smooth_hz)
    png_path, csv_path = plot_analysis(
        imp=imp,
        snd=snd,
        sound_file=args.sound_file,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        fmax_hz=args.fmax_hz,
        smooth_hz=args.smooth_hz,
        zp_factor=args.zp_factor,
        mode_pairs=args.mode_pairs,
        dpi=args.dpi,
    )
    print(f"saved figure: {png_path}")
    print(f"saved csv: {csv_path}")


if __name__ == "__main__":
    main()
