import json
import math
import time
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd().resolve()

DEFAULT_OUTPUT_DIR = SCRIPT_DIR
GPU_DEVICE_IDS = (0,)
DEVICES = tuple(f"cuda:{idx}" for idx in GPU_DEVICE_IDS)

RHO_AIR = 1.205
C_AIR = 343.0
K_AIR = RHO_AIR * C_AIR * C_AIR

NX = 500
NY = 500
NZ = 500
DX = 0.002
DY = 0.002
DZ = 0.002
COURANT = 0.85
SANSHIN_BOX_REFERENCE_COURANT = 0.99
SIM_TIME = 0.30

PML_WIDTH = 50
PML_STRENGTH = 0.04

MEMBRANE_SIZE_M = 0.20
BODY_DEPTH_M = 0.10
MEMBRANE_TENSION = 15000.0
BACK_TENSION_RATIO = 0.8
MEMBRANE_THICKNESS_M = 0.0006
MEMBRANE_DENSITY_VOL = 1150.0
MEMBRANE_DAMPING = 10.0
BACK_MEMBRANE_DAMPING = 10.0

BRIDGE_WIDTH_M = 0.016

# choices: "center", "real"
KOMA_POSITION = "real"

BRIDGE_FORCE_AMPLITUDE_N = 1.0
KOMA_FORCE_SCALE = 1000.0
SOUND_BRIDGE_FORCE_N = 1.0

# choices: "impulse", "sound"
INPUT_MODE = "impulse"

# choices: "ricker", "band_limited_noise", "hann_sine"
IMPULSE_PROFILE = "ricker"
IMPULSE_DISPLACEMENT_M = 0.001
IMPULSE_CENTER_FREQ_HZ = 650.0
IMPULSE_LOW_FREQ_HZ = 50.0
IMPULSE_HIGH_FREQ_HZ = 5000.0
IMPULSE_DELAY_SEC = 0.006
IMPULSE_DURATION_SEC = 0.060
IMPULSE_FADE_SEC = 0.004
IMPULSE_RANDOM_SEED = 1234

SOUND_FILE = SCRIPT_DIR.parent / "sound_source" / "GenSound1.txt"
SOUND_SAMPLE_RATE_HZ = 20000.0
SOUND_GAIN = 0.0005
# choices: True, False
SOUND_ONSET_DETECT = True
SOUND_ONSET_THRESH = 0.02
SOUND_ONSET_MAX_SEC = 0.10
# choices: "continuous", "gated"
SOUND_DRIVE_MODE = "continuous"
SOUND_APPLY_SEC = 0.03
SOUND_FADE_SEC = 0.005

NUM_FIELD_FRAMES = 8
NUM_MEMBRANE_FRAMES = 8
TARGET_RESOLUTION = 100
NUM_VOLUME_SAVES = 2400
# choices: True, False
SAVE_NPY = True
# choices: True, False
SAVE_JSON = True
# choices: True, False
SAVE_NPZ = False
# choices: True, False
SAVE_PNG = True
ENERGY_INTERVAL = 500
# choices: True, False
ENABLE_COOLING_SLEEP = False
COOL_EVERY_STEPS = 5
COOL_SLEEP_SEC = 0.03
# choices: True, False
SAVE_FARFIELD_SURFACE = True
FARFIELD_SURFACE_HALF_EXTENT_M = 0.32
FARFIELD_SURFACE_STRIDE = 2
FARFIELD_SURFACE_SAVE_INTERVAL = 16


def _compact_number(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace(".", "p").replace("-", "m")


def _source_tag(input_mode: str) -> str:
    mode = str(input_mode).lower()
    if mode == "impulse":
        return "imp"
    if mode == "sound":
        return "sound"
    raise ValueError(f"unknown input mode for output naming: {input_mode}")


def _koma_tag(koma_position: str) -> str:
    pos = str(koma_position).lower()
    if pos in ("center", "real"):
        return pos
    raise ValueError(f"unknown koma position for output naming: {koma_position}")


def _tension_tag(tension_n_m: float) -> str | None:
    tension_kn = tension_n_m / 1000.0
    if abs(tension_kn - 15.0) < 1e-9:
        return None
    return f"t{_compact_number(tension_kn)}k"


def _thickness_tag(thickness_m: float) -> str | None:
    thickness_mm = thickness_m * 1000.0
    if abs(thickness_mm - 0.6) < 1e-9:
        return None
    return f"h{_compact_number(thickness_mm)}mm"


def _sim_time_tag(sim_time_sec: float) -> str | None:
    if abs(sim_time_sec - 0.3) < 1e-9:
        return None
    return f"s{_compact_number(sim_time_sec)}"


def build_output_stem(
    input_mode: str,
    tension_n_m: float,
    thickness_m: float,
    koma_position: str,
    sim_time_sec: float,
) -> str:
    parts = ["sanshin_force", _source_tag(input_mode)]
    for tag in (
        _tension_tag(tension_n_m),
        _thickness_tag(thickness_m),
        _koma_tag(koma_position),
        _sim_time_tag(sim_time_sec),
    ):
        if tag:
            parts.append(tag)
    return "_".join(parts)


OUTPUT_STEM = build_output_stem(
    input_mode=INPUT_MODE,
    tension_n_m=MEMBRANE_TENSION,
    thickness_m=MEMBRANE_THICKNESS_M,
    koma_position=KOMA_POSITION,
    sim_time_sec=SIM_TIME,
)
OUTPUT_NPY = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}.npy"
OUTPUT_JSON_PATH = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_summary.json"
OUTPUT_PNG_PATH = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_diagnostics.png"
OUTPUT_FARFIELD_SURFACE = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_farfield_surface.json"
RUN_NAME = OUTPUT_STEM

SEAM_REL_WARM = 0.02

METRIC_KEYS = (
    "peak_obs_pressure_pa",
    "peak_front_center_w_m",
    "peak_back_center_w_m",
    "peak_front_jump_pa",
    "peak_back_jump_pa",
)

HISTORY_COMPARE_KEYS = (
    "obs_pressure",
    "front_center_w",
    "back_center_w",
    "front_jump",
    "back_jump",
)


def spectrum_peak(signal: np.ndarray, dt: float, min_hz: float = 20.0):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 8:
        return None
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return None
    window = np.hanning(signal.size)
    freq = np.fft.rfftfreq(signal.size, dt)
    spec = np.abs(np.fft.rfft((signal - signal.mean()) * window))
    valid = freq >= min_hz
    if not np.any(valid):
        return None
    idxs = np.flatnonzero(valid)
    idx = int(idxs[np.argmax(spec[valid])])
    return {"frequency_hz": float(freq[idx]), "amplitude": float(spec[idx])}


def peak_value(signal: np.ndarray, t_axis: np.ndarray):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return None
    idx = int(np.argmax(np.abs(signal)))
    return {"time_sec": float(t_axis[idx]), "value": float(signal[idx])}


def window_signal_stats(signal: np.ndarray, t_axis: np.ndarray, dt: float, start_sec: float, end_sec=None):
    signal = np.asarray(signal, dtype=np.float64)
    t_axis = np.asarray(t_axis, dtype=np.float64)
    if end_sec is None:
        mask = t_axis >= start_sec
    else:
        mask = (t_axis >= start_sec) & (t_axis < end_sec)
    if not np.any(mask):
        return {
            "start_sec": float(start_sec),
            "end_sec": None if end_sec is None else float(end_sec),
            "samples": 0,
            "peak_abs": None,
            "rms": None,
            "peak_time": None,
            "spectrum_peak": None,
            "peak_abs_over_global_peak": None,
        }
    seg = signal[mask]
    seg_t = t_axis[mask]
    return {
        "start_sec": float(start_sec),
        "end_sec": None if end_sec is None else float(end_sec),
        "samples": int(seg.size),
        "peak_abs": float(np.max(np.abs(seg))),
        "rms": float(np.sqrt(np.mean(seg * seg))),
        "peak_time": peak_value(seg, seg_t),
        "spectrum_peak": spectrum_peak(seg, dt),
    }


def post_drive_analysis(histories: dict, t_axis: np.ndarray, dt: float) -> dict:
    drive = np.asarray(histories["drive"], dtype=np.float64)
    active = np.flatnonzero(np.abs(drive) > 1e-15)
    if active.size:
        drive_start = float(t_axis[int(active[0])])
        drive_end = float(t_axis[int(active[-1])] + dt)
    else:
        drive_start = None
        drive_end = 0.0

    end_time = float(t_axis[-1] + dt) if t_axis.size else 0.0
    windows = [
        ("post_drive_all", drive_end, None),
        ("post_066_100ms", 0.066, 0.100),
        ("post_100_150ms", 0.100, 0.150),
        ("post_150_200ms", 0.150, 0.200),
    ]
    signals = ["obs_pressure", "front_center_w", "back_center_w", "front_jump", "back_jump"]
    result = {
        "drive_start_sec": drive_start,
        "drive_end_sec": drive_end,
        "sim_end_sec": end_time,
        "signals": {},
    }
    for signal_name in signals:
        signal = np.asarray(histories[signal_name], dtype=np.float64)
        global_peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        per_window = {}
        for window_name, start_sec, end_sec in windows:
            stats = window_signal_stats(signal, t_axis, dt, start_sec, end_sec)
            peak_abs = stats["peak_abs"]
            stats["peak_abs_over_global_peak"] = (
                None if peak_abs is None or global_peak == 0.0 else float(peak_abs / global_peak)
            )
            per_window[window_name] = stats
        result["signals"][signal_name] = {
            "global_peak_abs": global_peak,
            "windows": per_window,
        }
    return result


def make_sponge(shape, width: int, strength: float) -> np.ndarray:
    arr = np.ones(shape, dtype=np.float32)
    if width <= 0:
        return arr
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    dist = None
    for axis, grid in enumerate(grids):
        n = shape[axis]
        axis_dist = np.minimum(grid, (n - 1) - grid)
        dist = axis_dist if dist is None else np.minimum(dist, axis_dist)
    edge = np.clip((width - dist) / width, 0.0, 1.0)
    arr *= np.exp(-strength * edge * edge).astype(np.float32)
    return arr


def centered_patch(total_cells: int, patch_cells: int):
    start = (total_cells - patch_cells) // 2
    end = start + patch_cells
    if start < 2 or end > total_cells - 2:
        raise ValueError("patch is too large for the grid")
    return start, end


def build_geometry(nx, ny, nz, dx, dy, dz, membrane_size_m, body_depth_m, koma_position="center"):
    mem_cells_x = max(5, int(round(membrane_size_m / dx)))
    mem_cells_y = max(5, int(round(membrane_size_m / dy)))
    if mem_cells_x % 2 == 0:
        mem_cells_x += 1
    if mem_cells_y % 2 == 0:
        mem_cells_y += 1

    mx0, mx1 = centered_patch(nx, mem_cells_x)
    my0, my1 = centered_patch(ny, mem_cells_y)

    depth_cells = max(2, int(round(body_depth_m / dz)))

    z_front_face = nz // 2
    z_back_face = z_front_face + depth_cells
    if z_back_face >= nz - 2:
        raise ValueError("body depth is too large for the current nz")

    cavity_mask = np.zeros((nx, ny, nz), dtype=bool)
    cavity_mask[mx0:mx1, my0:my1, z_front_face:z_back_face] = True

    solid_mask = np.zeros((nx, ny, nz), dtype=bool)
    solid_mask[:, :, z_front_face:z_back_face] = True
    solid_mask[cavity_mask] = False

    air_mask = ~solid_mask

    obs_offset_cells = int(round(0.30 / dz))
    obs_z = z_front_face - obs_offset_cells
    if obs_z <= 1:
        raise ValueError("front observation point would leave the domain")

    bridge_half_width = max(0, int(round((BRIDGE_WIDTH_M / 2.0) / dx)))
    bridge_x_local = mem_cells_x // 2
    bridge_x_l = bridge_x_local - bridge_half_width
    bridge_x_r = bridge_x_local + bridge_half_width

    if str(koma_position) == "real":
        bridge_y_local = round((mem_cells_y - 1) * 2.0 / 3.0)
    else:
        bridge_y_local = mem_cells_y // 2

    return {
        "mx0": mx0,
        "mx1": mx1,
        "my0": my0,
        "my1": my1,
        "mem_cells_x": mem_cells_x,
        "mem_cells_y": mem_cells_y,
        "z_front_face": z_front_face,
        "z_back_face": z_back_face,
        "depth_cells": depth_cells,
        "air_mask": air_mask,
        "solid_mask": solid_mask,
        "cavity_mask": cavity_mask,
        "obs_x": nx // 2,
        "obs_y": ny // 2,
        "obs_z": obs_z,
        "bridge_x_l": bridge_x_l,
        "bridge_x_r": bridge_x_r,
        "bridge_y_local": bridge_y_local,
    }


def make_face_masks(air_mask: np.ndarray):
    ux_mask = np.zeros((air_mask.shape[0] + 1, air_mask.shape[1], air_mask.shape[2]), dtype=bool)
    uy_mask = np.zeros((air_mask.shape[0], air_mask.shape[1] + 1, air_mask.shape[2]), dtype=bool)
    uz_mask = np.zeros((air_mask.shape[0], air_mask.shape[1], air_mask.shape[2] + 1), dtype=bool)

    ux_mask[1:-1, :, :] = air_mask[:-1, :, :] & air_mask[1:, :, :]
    uy_mask[:, 1:-1, :] = air_mask[:, :-1, :] & air_mask[:, 1:, :]
    uz_mask[:, :, 1:-1] = air_mask[:, :, :-1] & air_mask[:, :, 1:]
    return ux_mask, uy_mask, uz_mask


def membrane_laplacian_torch(w: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    lap = torch.zeros_like(w)
    lap[1:-1, 1:-1] = (
        (w[2:, 1:-1] - 2.0 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / (dx * dx)
        + (w[1:-1, 2:] - 2.0 * w[1:-1, 1:-1] + w[1:-1, :-2]) / (dy * dy)
    )
    return lap


def membrane_potential_energy_torch(w: torch.Tensor, tension: float, dx: float, dy: float) -> torch.Tensor:
    grad_x = torch.diff(w, dim=0) / dx
    grad_y = torch.diff(w, dim=1) / dy
    return 0.5 * float(tension) * (torch.sum(grad_x * grad_x) + torch.sum(grad_y * grad_y)) * dx * dy


def load_sound_drive(args, nt: int, dt: float):
    path = Path(args.sound_file)
    if not path.exists():
        raise FileNotFoundError(f"sound file not found: {path}")
    src = np.loadtxt(path, dtype=np.float32)
    if src.ndim != 1:
        src = np.ravel(src)
    peak = float(np.max(np.abs(src))) if src.size else 0.0
    if peak > 0.0:
        src = src / peak

    start_idx = 0
    if args.sound_onset_detect:
        search_n = min(src.size, int(round(args.sound_onset_max_sec * args.sound_sample_rate)))
        if search_n > 0:
            mask = np.abs(src[:search_n]) >= args.sound_onset_thresh
            if np.any(mask):
                start_idx = int(np.argmax(mask))
    if start_idx > 0:
        src = src[start_idx:]

    t_sim = np.arange(nt, dtype=np.float64) * dt
    t_src = np.arange(src.size, dtype=np.float64) / float(args.sound_sample_rate)
    drive = np.interp(t_sim, t_src, src, left=0.0, right=0.0).astype(np.float32)
    drive *= float(args.sound_gain)
    sound_drive_mode = str(args.sound_drive_mode).lower()
    apply_steps = None
    fade_steps = None
    if sound_drive_mode == "gated":
        gated = np.zeros_like(drive)
        apply_steps = min(nt, max(1, int(round(float(args.sound_apply_sec) / dt))))
        fade_steps = max(0, int(round(float(args.sound_fade_sec) / dt)))
        envelope = np.ones(apply_steps, dtype=np.float32)
        if fade_steps > 0 and fade_steps * 2 < apply_steps:
            fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_steps, dtype=np.float32)))
            envelope[-fade_steps:] *= fade
        gated[:apply_steps] = drive[:apply_steps] * envelope
        drive = gated
    elif sound_drive_mode != "continuous":
        raise ValueError(f"unknown sound drive mode: {args.sound_drive_mode}")
    return drive, {
        "sound_file": str(path),
        "sound_sample_rate_hz": float(args.sound_sample_rate),
        "sound_gain": float(args.sound_gain),
        "sound_drive_mode": sound_drive_mode,
        "sound_apply_sec": float(args.sound_apply_sec) if sound_drive_mode == "gated" else None,
        "sound_fade_sec": float(args.sound_fade_sec) if sound_drive_mode == "gated" else None,
        "sound_apply_steps": int(apply_steps) if apply_steps is not None else None,
        "sound_fade_steps": int(fade_steps) if fade_steps is not None else None,
        "sound_onset_detect": bool(args.sound_onset_detect),
        "sound_onset_thresh": float(args.sound_onset_thresh),
        "sound_onset_shift_samples": int(start_idx),
        "sound_loaded_samples": int(src.size),
        "active_steps": int(np.count_nonzero(drive)),
    }


def prepare_impulse_drive(args, nt: int, dt: float):
    drive = np.zeros(nt, dtype=np.float32)
    profile = str(args.impulse_profile).lower()

    if profile == "single_step":
        drive[0] = float(args.impulse_displacement)

    elif profile == "ricker":
        t = np.arange(nt, dtype=np.float64) * dt
        tau = t - float(args.impulse_delay_sec)
        a = np.pi * float(args.impulse_center_freq_hz) * tau
        wave = (1.0 - 2.0 * a * a) * np.exp(-a * a)
        if args.impulse_duration_sec is not None and args.impulse_duration_sec > 0.0:
            active = np.abs(tau) <= (0.5 * float(args.impulse_duration_sec))
            wave = np.where(active, wave, 0.0)
        mx = float(np.max(np.abs(wave))) if wave.size else 0.0
        if mx > 0.0:
            drive[:] = (float(args.impulse_displacement) * wave / mx).astype(np.float32)

    elif profile == "hann_sine":
        n = max(3, int(round(float(args.impulse_duration_sec) / dt)))
        start = max(0, int(round(float(args.impulse_delay_sec) / dt)))
        stop = min(nt, start + n)
        count = max(0, stop - start)
        if count > 0:
            tloc = np.arange(count, dtype=np.float64) * dt
            window = np.hanning(count)
            wave = np.sin(2.0 * np.pi * float(args.impulse_center_freq_hz) * tloc) * window
            mx = float(np.max(np.abs(wave))) if wave.size else 0.0
            if mx > 0.0:
                drive[start:stop] = (float(args.impulse_displacement) * wave / mx).astype(np.float32)

    elif profile == "band_limited_noise":
        n = max(8, int(round(float(args.impulse_duration_sec) / dt)))
        start = max(0, int(round(float(args.impulse_delay_sec) / dt)))
        stop = min(nt, start + n)
        count = max(0, stop - start)
        if count > 0:
            rng = np.random.default_rng(int(args.impulse_random_seed))
            freqs = np.fft.rfftfreq(count, d=dt)
            band = (freqs >= float(args.impulse_low_freq_hz)) & (freqs <= float(args.impulse_high_freq_hz))
            spectrum = np.zeros(len(freqs), dtype=np.complex128)
            phases = rng.uniform(0.0, 2.0 * np.pi, int(np.count_nonzero(band)))
            spectrum[band] = np.exp(1j * phases)
            wave = np.fft.irfft(spectrum, n=count)
            wave -= float(np.mean(wave))
            fade_n = int(round(float(args.impulse_fade_sec) / dt))
            fade_n = max(0, min(fade_n, count // 2))
            if fade_n > 1:
                fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_n))
                wave[:fade_n] *= fade
                wave[-fade_n:] *= fade[::-1]
            mx = float(np.max(np.abs(wave))) if wave.size else 0.0
            if mx > 0.0:
                drive[start:stop] = (float(args.impulse_displacement) * wave / mx).astype(np.float32)

    else:
        raise ValueError(f"unknown impulse profile: {args.impulse_profile}")

    meta = {
        "impulse_profile": str(args.impulse_profile),
        "impulse_displacement_m": float(args.impulse_displacement),
        "impulse_center_freq_hz": float(args.impulse_center_freq_hz),
        "impulse_low_freq_hz": float(args.impulse_low_freq_hz),
        "impulse_high_freq_hz": float(args.impulse_high_freq_hz),
        "impulse_delay_sec": float(args.impulse_delay_sec),
        "impulse_duration_sec": float(args.impulse_duration_sec),
        "impulse_fade_sec": float(args.impulse_fade_sec),
        "impulse_random_seed": int(args.impulse_random_seed),
        "active_steps": int(np.count_nonzero(drive)),
    }
    return drive, meta


def save_diagnostic_png(
    path: Path,
    t_axis: np.ndarray,
    histories: dict,
    front_frames: np.ndarray,
    back_frames: np.ndarray,
    pressure_frames: np.ndarray,
    summary: dict,
    koma_position: str,
):
    t_ms = t_axis * 1000.0
    dt = float(summary["params"]["dt"])
    fig, ax = plt.subplots(4, 3, figsize=(17, 13), constrained_layout=True)
    axes = ax.ravel()

    axes[0].plot(t_ms, histories["drive"], color="black")
    axes[0].set_title("drive")
    axes[0].set_ylabel("m")

    axes[1].plot(t_ms, histories["obs_pressure"], color="tab:blue")
    axes[1].set_title("observation pressure")
    axes[1].set_ylabel("Pa")

    axes[2].plot(t_ms, histories["front_center_w"], label="front", color="tab:orange")
    axes[2].plot(t_ms, histories["back_center_w"], label="back", color="tab:green")
    axes[2].set_title("center membrane displacement")
    axes[2].set_ylabel("m")
    axes[2].legend()

    axes[3].plot(t_ms, histories["front_jump"], label="front", color="tab:red")
    axes[3].plot(t_ms, histories["back_jump"], label="back", color="tab:purple")
    axes[3].set_title("mean pressure jump")
    axes[3].set_ylabel("Pa")
    axes[3].legend()

    axes[4].plot(t_ms, histories["front_center_v"], label="front", color="tab:orange")
    axes[4].plot(t_ms, histories["back_center_v"], label="back", color="tab:green")
    axes[4].set_title("center membrane velocity")
    axes[4].set_ylabel("m/s")
    axes[4].legend()

    for key, label in [
        ("front_total_energy", "front"),
        ("back_total_energy", "back"),
        ("acoustic_energy", "acoustic"),
    ]:
        y = np.asarray(histories[key], dtype=np.float64)
        axes[5].plot(t_ms, np.maximum(y, 1e-30), label=label)
    axes[5].set_yscale("log")
    axes[5].set_title("energy")
    axes[5].set_ylabel("J")
    axes[5].legend()

    axes[6].plot(t_ms, histories["coupling_work_total"], color="tab:brown")
    axes[6].set_title("cumulative coupling work")
    axes[6].set_ylabel("J")

    freq = np.fft.rfftfreq(t_axis.size, dt)
    for key, label in [
        ("obs_pressure", "obs pressure"),
        ("front_center_w", "front membrane"),
        ("back_center_w", "back membrane"),
    ]:
        signal = np.asarray(histories[key], dtype=np.float64)
        spec = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(signal.size)))
        peak = max(float(np.max(spec)), 1e-30)
        axes[7].plot(freq, spec / peak, label=label)
    axes[7].set_xlim(0.0, min(5000.0, float(freq[-1])))
    axes[7].set_title("normalized spectra")
    axes[7].set_ylabel("relative")
    axes[7].legend()

    post = summary["analysis"].get("post_drive", {}).get("signals", {})
    labels = ["66-100", "100-150", "150-200"]
    x = np.arange(len(labels))
    width = 0.18
    for i, key in enumerate(["obs_pressure", "front_center_w", "back_center_w", "front_jump"]):
        windows = post.get(key, {}).get("windows", {})
        vals = [
            windows.get("post_066_100ms", {}).get("rms"),
            windows.get("post_100_150ms", {}).get("rms"),
            windows.get("post_150_200ms", {}).get("rms"),
        ]
        base = vals[0] if vals[0] not in (None, 0.0) else None
        ratios = [0.0 if base is None or v is None else float(v / base) for v in vals]
        axes[8].bar(x + (i - 1.5) * width, ratios, width, label=key)
    axes[8].set_title("post-drive RMS ratio")
    axes[8].set_xticks(x, labels)
    axes[8].legend(fontsize=8)

    if pressure_frames.size > 0:
        peak_idx = int(np.argmax(np.max(np.abs(pressure_frames), axis=(1, 2))))
        p_norm = max(float(np.max(np.abs(pressure_frames))), 1e-12)
        axes[9].imshow(
            pressure_frames[peak_idx].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-p_norm,
            vmax=p_norm,
            aspect="auto",
        )
        axes[9].set_title(f"pressure x-z peak frame {peak_idx}")
        axes[9].set_xticks([])
        axes[9].set_yticks([])
    else:
        axes[9].axis("off")

    if front_frames.size > 0:
        f_norm = max(float(np.max(np.abs(front_frames))), 1e-15)
        axes[10].imshow(front_frames[-1].T, origin="lower", cmap="RdBu_r", vmin=-f_norm, vmax=f_norm)
        axes[10].set_title("front membrane last")
        axes[10].set_xticks([])
        axes[10].set_yticks([])
    else:
        axes[10].axis("off")

    if back_frames.size > 0:
        b_norm = max(float(np.max(np.abs(back_frames))), 1e-15)
        axes[11].imshow(back_frames[-1].T, origin="lower", cmap="RdBu_r", vmin=-b_norm, vmax=b_norm)
        axes[11].set_title("back membrane last")
        axes[11].set_xticks([])
        axes[11].set_yticks([])
    else:
        axes[11].axis("off")

    for item in axes[:9]:
        item.grid(True, alpha=0.3)
        if item not in (axes[7], axes[8]):
            item.set_xlabel("ms")

    koma_label = "center (L/2)" if koma_position == "center" else "real (2L/3)"
    fig.suptitle(
        f"koma_force  koma={koma_label}  excitation=force",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ff_index_tensor(values, device):
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _select_2d(tensor, axis0_indices, axis1_indices):
    return tensor.index_select(0, _ff_index_tensor(axis0_indices, tensor.device)).index_select(
        1, _ff_index_tensor(axis1_indices, tensor.device)
    )


def build_farfield_surface_layout(args, geom: dict, nt: int, dt: float) -> dict:
    half_cells = int(round(FARFIELD_SURFACE_HALF_EXTENT_M / args.dx))
    stride = int(FARFIELD_SURFACE_STRIDE)
    interval = int(FARFIELD_SURFACE_SAVE_INTERVAL)
    cx = int(round((args.nx - 1) * 0.5))
    cy = int(geom["my0"] + geom["bridge_y_local"])
    cz = int(round(int(geom["z_front_face"]) + 0.5 * float(args.body_depth) / args.dz))
    ix0, ix1 = cx - half_cells, cx + half_cells
    iy0, iy1 = cy - half_cells, cy + half_cells
    iz0, iz1 = cz - half_cells, cz + half_cells
    pml = int(args.pml_width)
    if ix0 <= pml or iy0 <= pml or iz0 <= pml or ix1 >= args.nx - pml or iy1 >= args.ny - pml or iz1 >= args.nz - pml:
        raise ValueError(
            "farfield surface is too close to PML or outside physical saved region"
            f"x={ix0}:{ix1}, y={iy0}:{iy1}, z={iz0}:{iz1}, pml_width={pml}"
        )
    x_idx = np.arange(ix0, ix1 + 1, stride, dtype=np.int64)
    y_idx = np.arange(iy0, iy1 + 1, stride, dtype=np.int64)
    z_idx = np.arange(iz0, iz1 + 1, stride, dtype=np.int64)
    nframes = ((nt - 1) // interval) + 1

    coords = []
    normals = []
    face_offsets = {}

    def add_face(name, coord_array, normal):
        start = len(coords)
        flat = coord_array.reshape(-1, 3).astype(np.float32)
        coords.extend(flat)
        normals.extend(np.tile(np.asarray(normal, dtype=np.float32), (flat.shape[0], 1)))
        face_offsets[name] = [int(start), int(start + flat.shape[0])]

    xs = (x_idx - cx).astype(np.float32) * args.dx
    ys = (y_idx - cy).astype(np.float32) * args.dy
    zs = (z_idx - cz).astype(np.float32) * args.dz
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    add_face("x-", np.stack([np.full_like(yy, (ix0 - cx) * args.dx), yy, zz], axis=-1), (-1, 0, 0))
    add_face("x+", np.stack([np.full_like(yy, (ix1 - cx) * args.dx), yy, zz], axis=-1), (1, 0, 0))
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    add_face("y-", np.stack([xx, np.full_like(xx, (iy0 - cy) * args.dy), zz], axis=-1), (0, -1, 0))
    add_face("y+", np.stack([xx, np.full_like(xx, (iy1 - cy) * args.dy), zz], axis=-1), (0, 1, 0))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    add_face("z-", np.stack([xx, yy, np.full_like(xx, (iz0 - cz) * args.dz)], axis=-1), (0, 0, -1))
    add_face("z+", np.stack([xx, yy, np.full_like(xx, (iz1 - cz) * args.dz)], axis=-1), (0, 0, 1))

    coords_np = np.asarray(coords, dtype=np.float32)
    normals_np = np.asarray(normals, dtype=np.float32)
    OUTPUT_FARFIELD_SURFACE.parent.mkdir(parents=True, exist_ok=True)
    p_path = OUTPUT_FARFIELD_SURFACE.with_name(OUTPUT_FARFIELD_SURFACE.stem + "_p.npy")
    un_path = OUTPUT_FARFIELD_SURFACE.with_name(OUTPUT_FARFIELD_SURFACE.stem + "_un.npy")
    coords_path = OUTPUT_FARFIELD_SURFACE.with_name(OUTPUT_FARFIELD_SURFACE.stem + "_coords.npy")
    normals_path = OUTPUT_FARFIELD_SURFACE.with_name(OUTPUT_FARFIELD_SURFACE.stem + "_normals.npy")
    p_mem = np.lib.format.open_memmap(p_path, mode="w+", dtype=np.float32, shape=(nframes, coords_np.shape[0]))
    un_mem = np.lib.format.open_memmap(un_path, mode="w+", dtype=np.float32, shape=(nframes, coords_np.shape[0]))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "far-field closed-surface pressure and outward normal velocity output",
        "simulation": Path(__file__).name if "__file__" in globals() else "sanshin_analysis_pu_backmem_koma_force.py",
        "p_path": str(p_path),
        "un_path": str(un_path),
        "coords_path": str(coords_path),
        "normals_path": str(normals_path),
        "half_extent_m": float(FARFIELD_SURFACE_HALF_EXTENT_M),
        "surface_stride_cells": stride,
        "save_interval_steps": interval,
        "surface_spacing_m": [float(args.dx * stride), float(args.dy * stride), float(args.dz * stride)],
        "surface_element_area_m2": float((args.dx * stride) * (args.dy * stride)),
        "dt_solver_sec": float(dt),
        "dt_surface_sec": float(dt * interval),
        "p_time_alignment": "p_mid = 0.5 * (p_before_update + p_after_update), aligned approximately with u_n half step",
        "u_n_sign": "positive outward from the closed box",
        "center_raw_indices": [int(cx), int(cy), int(cz)],
        "surface_raw_indices": {"x": [int(ix0), int(ix1)], "y": [int(iy0), int(iy1)], "z": [int(iz0), int(iz1)]},
        "surface_point_count": int(coords_np.shape[0]),
        "frame_count": int(nframes),
        "face_offsets": face_offsets,
        "components": ["p_mid_pa", "u_normal_m_per_s"],
        "notes": [
            "Closed-surface p/u_n output for far-field directivity analysis.",
            "The full pressure volume NPY is optional and not required for this transform.",
        ],
    }
    np.save(coords_path, coords_np)
    np.save(normals_path, normals_np)
    return {
        "x_idx": x_idx, "y_idx": y_idx, "z_idx": z_idx,
        "ix0": ix0, "ix1": ix1, "iy0": iy0, "iy1": iy1, "iz0": iz0, "iz1": iz1,
        "interval": interval,
        "frame_cursor": 0,
        "p": p_mem,
        "un": un_mem,
        "meta": meta,
        "meta_path": OUTPUT_FARFIELD_SURFACE.with_suffix(".json"),
    }


def gather_farfield_pressure_faces(layout, p):
    x_idx, y_idx, z_idx = layout["x_idx"], layout["y_idx"], layout["z_idx"]
    ix0, ix1, iy0, iy1, iz0, iz1 = layout["ix0"], layout["ix1"], layout["iy0"], layout["iy1"], layout["iz0"], layout["iz1"]
    faces = []
    faces.append(p[ix0].index_select(0, _ff_index_tensor(y_idx, p.device)).index_select(1, _ff_index_tensor(z_idx, p.device)).to("cpu"))
    faces.append(p[ix1].index_select(0, _ff_index_tensor(y_idx, p.device)).index_select(1, _ff_index_tensor(z_idx, p.device)).to("cpu"))
    faces.append(p.index_select(0, _ff_index_tensor(x_idx, p.device))[:, iy0, :].index_select(1, _ff_index_tensor(z_idx, p.device)).to("cpu"))
    faces.append(p.index_select(0, _ff_index_tensor(x_idx, p.device))[:, iy1, :].index_select(1, _ff_index_tensor(z_idx, p.device)).to("cpu"))
    faces.append(_select_2d(p[:, :, iz0], x_idx, y_idx).to("cpu"))
    faces.append(_select_2d(p[:, :, iz1], x_idx, y_idx).to("cpu"))
    return np.concatenate([face.numpy().reshape(-1) for face in faces]).astype(np.float32, copy=False)


def gather_farfield_un_faces(layout, ux, uy, uz):
    x_idx, y_idx, z_idx = layout["x_idx"], layout["y_idx"], layout["z_idx"]
    ix0, ix1, iy0, iy1, iz0, iz1 = layout["ix0"], layout["ix1"], layout["iy0"], layout["iy1"], layout["iz0"], layout["iz1"]
    faces = []
    faces.append(-ux[ix0].index_select(0, _ff_index_tensor(y_idx, ux.device)).index_select(1, _ff_index_tensor(z_idx, ux.device)).to("cpu"))
    faces.append(ux[ix1 + 1].index_select(0, _ff_index_tensor(y_idx, ux.device)).index_select(1, _ff_index_tensor(z_idx, ux.device)).to("cpu"))
    faces.append(-uy.index_select(0, _ff_index_tensor(x_idx, uy.device))[:, iy0, :].index_select(1, _ff_index_tensor(z_idx, uy.device)).to("cpu"))
    faces.append(uy.index_select(0, _ff_index_tensor(x_idx, uy.device))[:, iy1 + 1, :].index_select(1, _ff_index_tensor(z_idx, uy.device)).to("cpu"))
    faces.append(-_select_2d(uz[:, :, iz0], x_idx, y_idx).to("cpu"))
    faces.append(_select_2d(uz[:, :, iz1 + 1], x_idx, y_idx).to("cpu"))
    return np.concatenate([face.numpy().reshape(-1) for face in faces]).astype(np.float32, copy=False)


def write_farfield_surface_frame(layout, step: int, p_before: np.ndarray, p_after: np.ndarray, un_values: np.ndarray):
    cursor = int(layout["frame_cursor"])
    p_mid = 0.5 * (p_before + p_after)
    layout["p"][cursor, :] = p_mid
    layout["un"][cursor, :] = un_values
    layout["frame_cursor"] = cursor + 1


def finalize_farfield_surface(layout):
    if layout is None:
        return None
    layout["p"].flush()
    layout["un"].flush()
    layout["meta"]["frames_written"] = int(layout["frame_cursor"])
    layout["meta_path"].write_text(json.dumps(layout["meta"], indent=2), encoding="utf-8")
    return layout["meta"]


def run_simulation(args):
    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("At least one CUDA device is required for this simulation.")
    device0 = torch.device(args.devices[0])

    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = args.dx, args.dy, args.dz

    dt = args.courant / (C_AIR * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)))
    nt = max(1, int(round(args.sim_time / dt)))

    geom = build_geometry(nx, ny, nz, dx, dy, dz, args.membrane_size, args.body_depth, args.koma_position)
    mx0, mx1 = geom["mx0"], geom["mx1"]
    my0, my1 = geom["my0"], geom["my1"]
    z_front_face, z_back_face = geom["z_front_face"], geom["z_back_face"]

    air_mask_np = geom["air_mask"].astype(np.float32)
    ux_mask_np, uy_mask_np, uz_mask_np = make_face_masks(geom["air_mask"])
    sponge_p_np = make_sponge((nx, ny, nz), args.pml_width, args.pml_strength) * air_mask_np
    sponge_ux_np = make_sponge((nx + 1, ny, nz), args.pml_width, args.pml_strength) * ux_mask_np.astype(np.float32)
    sponge_uy_np = make_sponge((nx, ny + 1, nz), args.pml_width, args.pml_strength) * uy_mask_np.astype(np.float32)
    sponge_uz_np = make_sponge((nx, ny, nz + 1), args.pml_width, args.pml_strength) * uz_mask_np.astype(np.float32)

    air = torch.from_numpy(air_mask_np).to(device=device0)
    ux_mask = torch.from_numpy(ux_mask_np).to(device=device0)
    uy_mask = torch.from_numpy(uy_mask_np).to(device=device0)
    uz_mask = torch.from_numpy(uz_mask_np).to(device=device0)
    sponge_p = torch.from_numpy(sponge_p_np).to(device=device0)
    sponge_ux = torch.from_numpy(sponge_ux_np).to(device=device0)
    sponge_uy = torch.from_numpy(sponge_uy_np).to(device=device0)
    sponge_uz = torch.from_numpy(sponge_uz_np).to(device=device0)

    p = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device0)
    ux = torch.zeros((nx + 1, ny, nz), dtype=torch.float32, device=device0)
    uy = torch.zeros((nx, ny + 1, nz), dtype=torch.float32, device=device0)
    uz = torch.zeros((nx, ny, nz + 1), dtype=torch.float32, device=device0)

    mem_nx, mem_ny = geom["mem_cells_x"], geom["mem_cells_y"]
    sigma_s = args.membrane_density * args.membrane_thickness

    w_front = torch.zeros((mem_nx, mem_ny), dtype=torch.float32, device=device0)
    v_front = torch.zeros_like(w_front)
    w_back = torch.zeros_like(w_front)
    v_back = torch.zeros_like(w_front)

    center = (mem_nx // 2, mem_ny // 2)
    bridge_y_local = geom["bridge_y_local"]
    bridge_x_l = geom["bridge_x_l"]
    bridge_x_r = geom["bridge_x_r"]

    d_area = dx * dy
    d_vol = dx * dy * dz

    accel_per_drive = args.bridge_force_scale / (sigma_s * d_area)

    if args.input_mode == "sound":
        drive_np, drive_meta = load_sound_drive(args, nt, dt)
    else:
        drive_np, drive_meta = prepare_impulse_drive(args, nt, dt)

    frame_steps = sorted(set(np.linspace(0, nt - 1, max(1, args.field_frames), dtype=int).tolist()))
    mem_frame_steps = sorted(set(np.linspace(0, nt - 1, max(1, args.membrane_frames), dtype=int).tolist()))
    calc_area_size = nx - 2 * int(args.pml_width)
    save_grid_step = max(1, int(calc_area_size // int(args.target_resolution)))
    volume_save_interval = max(1, nt // int(args.num_volume_saves))
    sx0 = (nx // 2) - (calc_area_size // 2)
    sx1 = (nx // 2) + (calc_area_size // 2)
    sy0 = (ny // 2) - (calc_area_size // 2)
    sy1 = (ny // 2) + (calc_area_size // 2)
    sz0 = (nz // 2) - (calc_area_size // 2)
    sz1 = (nz // 2) + (calc_area_size // 2)

    pressure_frames = []
    front_frames = []
    back_frames = []
    volume_frames = []

    hist_obs_pressure = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_front_jump = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_back_jump = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_front_center_w = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_back_center_w = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_front_center_v = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_back_center_v = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_coupling_work_total = torch.empty(nt, dtype=torch.float32, device=device0)
    hist_front_total_energy = np.empty(nt, dtype=np.float64)
    hist_back_total_energy = np.empty(nt, dtype=np.float64)
    hist_acoustic_energy = np.empty(nt, dtype=np.float64)

    cumulative_coupling_work = torch.zeros((), dtype=torch.float32, device=device0)
    front_energy_last = 0.0
    back_energy_last = 0.0
    acoustic_energy_last = 0.0
    t0 = time.perf_counter()
    last_progress_time = t0
    last_progress_step = 0
    sleep_accumulated = 0.0
    jst = timezone(timedelta(hours=9), "JST")
    progress_interval = max(1, nt // 10)
    energy_interval = max(1, int(args.energy_interval))
    cool_every_steps = max(0, int(args.cool_every_steps))
    cool_sleep_sec = max(0.0, float(args.cool_sleep_sec))

    koma_label = "center (L/2)" if args.koma_position == "center" else "real (2L/3)"
    print(f"--- Simulation start (Single GPU) ---")
    print(f"Grid: {nx} x {ny} x {nz}, dt={dt:.3e} s, nt={nt}, device={device0}")
    print(f"Koma position: {koma_label}  bridge_y_local={bridge_y_local}  (global y={my0+bridge_y_local})")
    print(f"Bridge force scale: {args.bridge_force_scale:.3e} N/m  accel_per_drive: {accel_per_drive:.3e} m/s^2/m")
    sleep_budget = (nt // cool_every_steps) * cool_sleep_sec if cool_every_steps > 0 else 0.0
    print(
        f"Outputs: npy={args.save_npy}, json={args.save_json}, png={args.save_png}; "
        f"cool_sleep_budget={sleep_budget:.1f}s "
        f"(every {cool_every_steps} steps x {cool_sleep_sec:.3f}s)"
    )
    print(f"Output: {args.out}")
    farfield_surface = None
    if SAVE_FARFIELD_SURFACE:
        farfield_surface = build_farfield_surface_layout(args, geom, nt, dt)
        print(
            "Farfield surface: "
            f"half={FARFIELD_SURFACE_HALF_EXTENT_M}m, stride={FARFIELD_SURFACE_STRIDE}, "
            f"interval={FARFIELD_SURFACE_SAVE_INTERVAL}, "
            f"points={farfield_surface['meta']['surface_point_count']}, "
            f"frames={farfield_surface['meta']['frame_count']}"
        )

    with torch.no_grad():
        for step in range(nt):
            if step % progress_interval == 0 or step == nt - 1:
                now = time.perf_counter()
                elapsed = now - t0
                interval_elapsed = now - last_progress_time
                interval_steps = max(1, step + 1 - last_progress_step)
                interval_rate = interval_steps / max(interval_elapsed, 1e-12)
                done = (step + 1) / nt
                rem = elapsed / done - elapsed if done > 0.0 else float("nan")
                fin = datetime.fromtimestamp(time.time() + max(rem, 0.0), jst).strftime("%H:%M:%S")
                mem0_gib = torch.cuda.max_memory_allocated(device0) / (1024.0 ** 3)
                vram_text = (
                    f"{mem0_gib:.2f} GiB"
                )
                print(
                    f"Step {step + 1}/{nt} "
                    f"({done * 100:5.1f}%) - Elapsed: {elapsed:7.1f}s, "
                    f"Rem: {rem:7.1f}s, End: {fin}, "
                    f"Interval: {interval_rate:6.1f} step/s, "
                    f"Sleep: {sleep_accumulated:6.1f}s, "
                    f"MaxVRAM: {vram_text}",
                    flush=True,
                )
                last_progress_time = now
                last_progress_step = step + 1

            zf = z_front_face
            zb = z_back_face
            p_front_minus = p[mx0:mx1, my0:my1, zf - 1]
            p_front_plus  = p[mx0:mx1, my0:my1, zf]
            delta_p_front = p_front_minus - p_front_plus
            p_back_minus = p[mx0:mx1, my0:my1, zb - 1]
            p_back_plus  = p[mx0:mx1, my0:my1, zb]
            delta_p_back = p_back_minus - p_back_plus

            lap_front = membrane_laplacian_torch(w_front, dx, dy)
            lap_back  = membrane_laplacian_torch(w_back, dx, dy)

            a_front = (args.tension * lap_front - args.membrane_damping * v_front + delta_p_front) / sigma_s
            a_back  = ((args.tension * args.back_tension_ratio) * lap_back
                       - args.back_membrane_damping * v_back + delta_p_back) / sigma_s

            drive_val = float(drive_np[step])
            a_bridge = drive_val * accel_per_drive
            a_front[bridge_x_l, bridge_y_local] = a_front[bridge_x_l, bridge_y_local] + a_bridge
            a_front[bridge_x_r, bridge_y_local] = a_front[bridge_x_r, bridge_y_local] + a_bridge

            v_front += dt * a_front
            v_back  += dt * a_back
            w_front += dt * v_front
            w_back  += dt * v_back

            for w, v in [(w_front, v_front), (w_back, v_back)]:
                w[0, :] = 0.0
                w[-1, :] = 0.0
                w[:, 0] = 0.0
                w[:, -1] = 0.0
                v[0, :] = 0.0
                v[-1, :] = 0.0
                v[:, 0] = 0.0
                v[:, -1] = 0.0
            # 速度場の更新 (データ転送無し、通常のFDTD更新)
            ux[1:nx, :, :] -= (dt / (RHO_AIR * dx)) * (p[1:, :, :] - p[:-1, :, :])
            uy[:, 1:ny, :] -= (dt / (RHO_AIR * dy)) * (p[:, 1:, :] - p[:, :-1, :])
            uz[:, :, 1:nz] -= (dt / (RHO_AIR * dz)) * (p[:, :, 1:] - p[:, :, :-1])

            ux[0, :, :] = 0.0; ux[nx, :, :] = 0.0
            uy[:, 0, :] = 0.0; uy[:, ny, :] = 0.0
            uz[:, :, 0] = 0.0; uz[:, :, nz] = 0.0

            # 膜速度が空気の垂直方向の粒子速度を駆動
            uz[mx0:mx1, my0:my1, zf] = v_front
            uz[mx0:mx1, my0:my1, zb] = v_back

            ux *= ux_mask
            uy *= uy_mask
            uz *= uz_mask
            ux *= sponge_ux
            uy *= sponge_uy
            uz *= sponge_uz

            div = (
                (ux[1:, :, :] - ux[:-1, :, :]) / dx
                + (uy[:, 1:, :] - uy[:, :-1, :]) / dy
                + (uz[:, :, 1:] - uz[:, :, :-1]) / dz
            )
            farfield_due = farfield_surface is not None and (step % int(farfield_surface["interval"]) == 0)
            if farfield_due:
                farfield_p_before = gather_farfield_pressure_faces(farfield_surface, p)

            # 音圧場の更新
            
            p -= (K_AIR * dt) * div
            p *= sponge_p
            p *= air

            if farfield_due:
                farfield_p_after = gather_farfield_pressure_faces(farfield_surface, p)
                farfield_un = gather_farfield_un_faces(farfield_surface, ux, uy, uz)
                write_farfield_surface_frame(farfield_surface, step, farfield_p_before, farfield_p_after, farfield_un)

            if args.save_npy and step % volume_save_interval == 0:
                volume_frames.append(p[sx0:sx1:save_grid_step, sy0:sy1:save_grid_step, sz0:sz1:save_grid_step].contiguous().cpu().numpy())

            energy_due = (step % energy_interval == 0) or (step == nt - 1)
            coupling_power_front = torch.sum(delta_p_front * v_front) * d_area
            coupling_power_back  = torch.sum(delta_p_back  * v_back)  * d_area
            cumulative_coupling_work += (coupling_power_front + coupling_power_back) * dt

            wf_np = None
            wb_np = None
            if energy_due:
                front_kin = 0.5 * sigma_s * torch.sum(v_front * v_front) * d_area
                back_kin  = 0.5 * sigma_s * torch.sum(v_back  * v_back)  * d_area
                front_pot = membrane_potential_energy_torch(w_front, args.tension, dx, dy)
                back_pot = membrane_potential_energy_torch(w_back, args.tension * args.back_tension_ratio, dx, dy)
                front_energy_last = float((front_kin + front_pot).item())
                back_energy_last  = float((back_kin  + back_pot).item())
                acoustic_energy_last = float(
                    torch.sum((p * p) / (2.0 * K_AIR)).item() * d_vol
                    + 0.5 * RHO_AIR * d_vol * (
                        torch.sum(ux * ux).item() + torch.sum(uy * uy).item() + torch.sum(uz * uz).item()
                    )
                )

            hist_obs_pressure[step]         = p[geom["obs_x"], geom["obs_y"], geom["obs_z"]]
            hist_front_jump[step]           = torch.mean(delta_p_front)
            hist_back_jump[step]            = torch.mean(delta_p_back)
            hist_front_center_w[step]       = w_front[center]
            hist_back_center_w[step]        = w_back[center]
            hist_front_center_v[step]       = v_front[center]
            hist_back_center_v[step]        = v_back[center]
            hist_coupling_work_total[step]  = cumulative_coupling_work
            hist_front_total_energy[step]   = front_energy_last
            hist_back_total_energy[step]    = back_energy_last
            hist_acoustic_energy[step]      = acoustic_energy_last

            if step in frame_steps:
                pressure_frames.append(p[:, ny // 2, :].detach().cpu().numpy())
            if step in mem_frame_steps:
                if wf_np is None:
                    wf_np = w_front.detach().cpu().numpy()
                    wb_np = w_back.detach().cpu().numpy()
                front_frames.append(wf_np.copy())
                back_frames.append(wb_np.copy())
            if cool_every_steps > 0 and cool_sleep_sec > 0.0 and (step + 1) % cool_every_steps == 0:
                time.sleep(cool_sleep_sec)
                sleep_accumulated += cool_sleep_sec
    elapsed = time.perf_counter() - t0
    farfield_surface_meta = finalize_farfield_surface(farfield_surface)
    t_axis = np.arange(nt, dtype=np.float64) * dt
    histories = {
        "drive":               np.asarray(drive_np, dtype=np.float64),
        "obs_pressure":        hist_obs_pressure.detach().cpu().numpy().astype(np.float64),
        "front_jump":          hist_front_jump.detach().cpu().numpy().astype(np.float64),
        "back_jump":           hist_back_jump.detach().cpu().numpy().astype(np.float64),
        "front_center_w":      hist_front_center_w.detach().cpu().numpy().astype(np.float64),
        "back_center_w":       hist_back_center_w.detach().cpu().numpy().astype(np.float64),
        "front_center_v":      hist_front_center_v.detach().cpu().numpy().astype(np.float64),
        "back_center_v":       hist_back_center_v.detach().cpu().numpy().astype(np.float64),
        "front_total_energy":  hist_front_total_energy,
        "back_total_energy":   hist_back_total_energy,
        "acoustic_energy":     hist_acoustic_energy,
        "coupling_work_total": hist_coupling_work_total.detach().cpu().numpy().astype(np.float64),
    }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": f"{device0}",
        "elapsed_sec": elapsed,
        "params": {
            "nx": nx, "ny": ny, "nz": nz,
            "dx": dx, "dy": dy, "dz": dz,
            "dt": dt, "nt": nt, "sim_time": nt * dt,
            "courant": args.courant,
            "membrane_size_m": args.membrane_size,
            "body_depth_m": args.body_depth,
            "z_front_face": z_front_face,
            "z_back_face": z_back_face,
            "depth_cells": geom["depth_cells"],
            "tension_n_m": args.tension,
            "back_tension_ratio": args.back_tension_ratio,
            "membrane_thickness_m": args.membrane_thickness,
            "membrane_density_vol": args.membrane_density,
            "membrane_damping": args.membrane_damping,
            "back_membrane_damping": args.back_membrane_damping,
            "input_mode": args.input_mode,
            "bridge_excitation_mode": "force",
            "koma_position": args.koma_position,
            "bridge_y_local": bridge_y_local,
            "bridge_y_global": int(my0 + bridge_y_local),
            "bridge_force_scale": args.bridge_force_scale,
            "bridge_force_amplitude_n": float(args.bridge_force_amplitude_n),
            "koma_force_scale": float(args.koma_force_scale) if args.input_mode == "impulse" else None,
            "sound_bridge_force_n": float(args.sound_bridge_force_n) if args.input_mode == "sound" else None,
            "drive_meta": drive_meta,
            "volume_save_interval": int(volume_save_interval),
            "save_grid_step": int(save_grid_step),
            "volume_save_region": {
                "x": [int(sx0), int(sx1)],
                "y": [int(sy0), int(sy1)],
                "z": [int(sz0), int(sz1)],
            },
            "farfield_surface": farfield_surface_meta,
        },
        "analysis": {
            "peak_obs_pressure_pa":    float(np.max(np.abs(histories["obs_pressure"]))),
            "peak_front_center_w_m":   float(np.max(np.abs(histories["front_center_w"]))),
            "peak_back_center_w_m":    float(np.max(np.abs(histories["back_center_w"]))),
            "peak_front_jump_pa":      float(np.max(np.abs(histories["front_jump"]))),
            "peak_back_jump_pa":       float(np.max(np.abs(histories["back_jump"]))),
            "front_center_peak":       spectrum_peak(histories["front_center_w"], dt),
            "back_center_peak":        spectrum_peak(histories["back_center_w"], dt),
            "obs_pressure_peak":       spectrum_peak(histories["obs_pressure"], dt),
            "front_center_peak_time":  peak_value(histories["front_center_w"], t_axis),
            "back_center_peak_time":   peak_value(histories["back_center_w"], t_axis),
            "post_drive":              post_drive_analysis(histories, t_axis, dt),
        },
    }

    # 元のコードと完全に同じように明示的に指定する書き方
    front_frames_np  = np.asarray(front_frames, dtype=np.float32)
    back_frames_np   = np.asarray(back_frames, dtype=np.float32)
    pressure_frames_np = np.asarray(pressure_frames, dtype=np.float32)
    volume_frames_np = np.asarray(volume_frames, dtype=np.float32)

    return summary, histories, front_frames_np, back_frames_np, volume_frames_np, t_axis


def extract_metrics(summary: dict) -> dict:
    analysis = summary["analysis"]
    return {key: float(analysis[key]) for key in METRIC_KEYS}


def build_config():
    if INPUT_MODE == "impulse":
        bridge_force_scale = BRIDGE_FORCE_AMPLITUDE_N / IMPULSE_DISPLACEMENT_M * KOMA_FORCE_SCALE
    else:
        bridge_force_scale = SOUND_BRIDGE_FORCE_N / SOUND_GAIN

    return SimpleNamespace(
        devices=DEVICES,
        out=DEFAULT_OUTPUT_DIR,
        nx=NX, ny=NY, nz=NZ,
        dx=DX, dy=DY, dz=DZ,
        courant=COURANT,
        sanshin_box_reference_courant=SANSHIN_BOX_REFERENCE_COURANT,
        sim_time=SIM_TIME,
        pml_width=PML_WIDTH,
        pml_strength=PML_STRENGTH,
        membrane_size=MEMBRANE_SIZE_M,
        body_depth=BODY_DEPTH_M,
        tension=MEMBRANE_TENSION,
        back_tension_ratio=BACK_TENSION_RATIO,
        membrane_thickness=MEMBRANE_THICKNESS_M,
        membrane_density=MEMBRANE_DENSITY_VOL,
        membrane_damping=MEMBRANE_DAMPING,
        back_membrane_damping=BACK_MEMBRANE_DAMPING,
        input_mode=INPUT_MODE,
        koma_position=KOMA_POSITION,
        bridge_force_scale=bridge_force_scale,
        bridge_force_amplitude_n=BRIDGE_FORCE_AMPLITUDE_N,
        koma_force_scale=KOMA_FORCE_SCALE,
        sound_bridge_force_n=SOUND_BRIDGE_FORCE_N,
        impulse_profile=IMPULSE_PROFILE,
        impulse_displacement=IMPULSE_DISPLACEMENT_M,
        impulse_center_freq_hz=IMPULSE_CENTER_FREQ_HZ,
        impulse_low_freq_hz=IMPULSE_LOW_FREQ_HZ,
        impulse_high_freq_hz=IMPULSE_HIGH_FREQ_HZ,
        impulse_delay_sec=IMPULSE_DELAY_SEC,
        impulse_duration_sec=IMPULSE_DURATION_SEC,
        impulse_fade_sec=IMPULSE_FADE_SEC,
        impulse_random_seed=IMPULSE_RANDOM_SEED,
        sound_file=SOUND_FILE,
        sound_sample_rate=SOUND_SAMPLE_RATE_HZ,
        sound_gain=SOUND_GAIN,
        sound_drive_mode=SOUND_DRIVE_MODE,
        sound_apply_sec=SOUND_APPLY_SEC,
        sound_fade_sec=SOUND_FADE_SEC,
        sound_onset_detect=SOUND_ONSET_DETECT,
        sound_onset_thresh=SOUND_ONSET_THRESH,
        sound_onset_max_sec=SOUND_ONSET_MAX_SEC,
        field_frames=NUM_FIELD_FRAMES,
        membrane_frames=NUM_MEMBRANE_FRAMES,
        save_npy=SAVE_NPY,
        save_json=SAVE_JSON,
        save_png=SAVE_PNG,
        target_resolution=TARGET_RESOLUTION,
        num_volume_saves=NUM_VOLUME_SAVES,
        save_npz=SAVE_NPZ,
        energy_interval=ENERGY_INTERVAL,
        enable_cooling_sleep=ENABLE_COOLING_SLEEP,
        cool_every_steps=COOL_EVERY_STEPS if ENABLE_COOLING_SLEEP else 0,
        cool_sleep_sec=COOL_SLEEP_SEC if ENABLE_COOLING_SLEEP else 0.0,
    )


def main():
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = build_config()
    print(f"Running on Single GPU{RUN_NAME}: sim_time={args.sim_time}s")
    summary, histories, front_frames, back_frames, pressure_frames, volume_frames, t_axis = run_simulation(args)

    if args.save_npy:
        np.save(OUTPUT_NPY, np.asarray(volume_frames, dtype=np.float32))

    if args.save_png:
        save_diagnostic_png(
            OUTPUT_PNG_PATH, t_axis, histories,
            front_frames, back_frames, pressure_frames,
            summary, args.koma_position,
        )
    
    # passed, split plane are deleted
    validation_json = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": f"koma force validation  excitation=force  koma={args.koma_position}",
        "summary": summary,
        "metrics": extract_metrics(summary),
    }
    if args.save_json:
        with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump(validation_json, f, indent=2)

    # passed, split plane are deleted
    print(json.dumps({
        "koma_position": args.koma_position,
        "front_center_peak_hz": summary["analysis"].get("front_center_peak", {}).get("frequency_hz") if summary["analysis"].get("front_center_peak") else None,
        "back_center_peak_hz": summary["analysis"].get("back_center_peak", {}).get("frequency_hz") if summary["analysis"].get("back_center_peak") else None,
        "summary_json": str(OUTPUT_JSON_PATH) if args.save_json else None,
        "elapsed_sec": summary["elapsed_sec"],
        "metrics": extract_metrics(summary),
    }, indent=2))

if __name__ == "__main__":
    main()