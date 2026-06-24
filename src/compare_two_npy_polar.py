from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_NPY_A = SCRIPT_DIR / "sanshin_force_imp_real.zarr"
DEFAULT_NPY_B = SCRIPT_DIR / "sanshin_force_imp_t10k_real.zarr"
DEFAULT_OUT_DIR = SCRIPT_DIR
DEFAULT_OUT_DIR_PEAK = SCRIPT_DIR
DEFAULT_FREQUENCY_MODE = "mode"
FREQUENCY_LABEL_PREFIX = "M"

LABEL_A = "A: 15kN"
LABEL_B = "B: 10kN"

# 極座標計算の既定値。method: farfield / volume
# farfield は閉曲面p/u_n、volume は保存圧力場を使う。
ANGLE_STEP_DEG = 2.0
DB_FLOOR = -30.0
RGRID_STEP_DB = 5.0
BAND_HZ = 30.0
FARFIELD_BAND_HZ = BAND_HZ
FARFIELD_BAND_SAMPLES = 3
FARFIELD_CHUNK_POINTS = 4096
SURFACE_GRID_STRIDE = 1
SMOOTH_ANGLES = 2
HIGH_FREQ_STABILIZE_MIN_HZ = 2000.0
HIGH_FREQ_BAND_HZ = 90.0
HIGH_FREQ_BAND_SAMPLES = 7
HIGH_FREQ_SMOOTH_ANGLES = 4
RADIUS_M = 0.25
DIRECTIVITY_CENTER = "body"

MODE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
LS_A = "-"
LS_B = "--"

OUT_PNG = "comparison_polar_all_modes_tight_legend.png"
OUT_PNG_PEAK = "comparison_peak_modes_farfield_polar_all_modes.png"
SAVE_DPI = 300
SAVE_PAD_IN = 0.1

FONT_FAMILY = "DejaVu Serif"
FONT_SIZE_BASE = 16
FONT_SIZE_TITLE = 20
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICK = 14

LEGEND_BBOX_Y = 0.19

# 周波数選択: mode / peak。peak は保存圧力場からピークを拾う。
PEAK_SPECTRUM_RADIUS_M = 0.30
ZERO_PAD_FACTOR = 8
SPECTRUM_SMOOTH_HZ = 30.0
PEAK_MIN_HZ = 150.0
PEAK_MAX_HZ = 3000.0
PEAK_MIN_SEPARATION_HZ = 180.0
MATCH_WINDOW_HZ = 180.0
N_PEAK_MODES = 4

C_AIR = 343.0
RHO_AIR = 1.205


def load_summary_sidecar(path: Path) -> tuple[dict, dict, dict, Path | None]:
    candidates = [
        path.with_name(f"{path.stem}_summary.json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        summary = dict(payload.get("summary", payload))
        params = dict(summary.get("params", {}))
        analysis = dict(summary.get("analysis", {}))
        return params, analysis, summary, candidate
    return {}, {}, {}, None


def load_histories_sidecar(path: Path) -> dict:
    candidates = [
        path.with_name(f"{path.stem}_histories.npz"),
        path.with_suffix(".npz"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            npz = np.load(candidate, allow_pickle=False)
            out: dict = {}
            for key in npz.files:
                val = np.asarray(npz[key])
                out[key] = val
                if key.startswith("hist_"):
                    out[key[5:]] = val
            return out
        except Exception as exc:
            print(f"warning: failed to load histories sidecar {candidate}: {exc}")
    return {}


# 変更後
def load_package(path: Path):
    loaded = zarr.open(str(path), mode="r")
    params, analysis, summary, _sidecar = load_summary_sidecar(path)
    histories = load_histories_sidecar(path)
    return loaded, params, analysis, summary, np.asarray([], dtype=np.float64), histories


def saved_dt(params: dict, t: np.ndarray, nt_saved: int) -> float:
    if t.size == nt_saved and t.size >= 2:
        return float(np.median(np.diff(t)))
    dt = float(params.get("dt", 1.0))
    interval = int(params.get("volume_save_interval", params.get("save_interval", 1)))
    return dt * max(1, interval)


def saved_spacing(params: dict, axis: str = "x") -> float:
    base = float(params.get({"x": "dx", "y": "dy", "z": "dz"}[axis], params.get("dx", 1.0)))
    return base * int(params.get("save_grid_step", 1))


def volume_region(params: dict, axis: str):
    region = params.get("volume_save_region", {})
    vals = region.get(axis)
    if vals is None:
        n = int(params.get({"x": "nx", "y": "ny", "z": "nz"}[axis], 0))
        return 0.0, float(n)
    return float(vals[0]), float(vals[1])


def to_saved_index(params: dict, axis: str, original_index: float) -> float:
    start, _ = volume_region(params, axis)
    step = float(params.get("save_grid_step", 1))
    return (float(original_index) - start) / step


def source_center_saved(params: dict, shape_xyz) -> tuple[float, float, float]:
    nx_s, ny_s, nz_s = shape_xyz
    nx = float(params.get("nx", nx_s))
    ny = float(params.get("ny", ny_s))
    x0 = float(params.get("koma_center_x", (nx - 1.0) * 0.5))
    y0 = float(params.get("koma_center_y", (ny - 1.0) * 0.5))
    z0 = float(params.get("z_front_face", params.get("nz", nz_s) * 0.5))
    return (
        float(np.clip(to_saved_index(params, "x", x0), 0, nx_s - 1)),
        float(np.clip(to_saved_index(params, "y", y0), 0, ny_s - 1)),
        float(np.clip(to_saved_index(params, "z", z0), 0, nz_s - 1)),
    )


def body_center_saved(params: dict, shape_xyz) -> tuple[float, float, float]:
    cx, cy, cz_front = source_center_saved(params, shape_xyz)
    dz_s = saved_spacing(params, "z")
    body_depth = float(params.get("body_depth_m", 0.1))
    cz = cz_front + 0.5 * body_depth / dz_s
    return (cx, cy, float(np.clip(cz, 0, shape_xyz[2] - 1)))


def analysis_center_saved(params: dict, shape_xyz, mode: str) -> tuple[float, float, float]:
    mode = str(mode).lower()
    if mode == "source":
        return source_center_saved(params, shape_xyz)
    if mode == "body":
        return body_center_saved(params, shape_xyz)
    raise ValueError(f"unknown center mode: {mode}")


def observation_wave(data: np.ndarray, params: dict, radius_m: float) -> np.ndarray:
    _nt, nx, ny, nz = data.shape
    cx, cy, cz = source_center_saved(params, (nx, ny, nz))
    dz_s = saved_spacing(params, "z")
    iz = int(np.clip(round(cz - radius_m / dz_s), 0, nz - 1))
    return data[:, int(round(cx)), int(round(cy)), iz]


def front_membrane_theory_freq(params: dict, m: int, n: int) -> float:
    tension = float(params.get("tension_n_m", params.get("T_tension", 15000.0)))
    density_vol = float(params.get("membrane_density_vol", 1150.0))
    thickness = float(params.get("membrane_thickness_m", 0.0006))
    sigma = float(params.get("rho_s", density_vol * thickness))
    size = float(params.get("membrane_size_m", params.get("src_width_phys", 0.2)))
    c = math.sqrt(tension / sigma)
    return float(0.5 * c * math.sqrt((m / size) ** 2 + (n / size) ** 2))


def parse_mode_pairs(text: str):
    pairs = []
    normalized = str(text).replace("(", "").replace(")", "").replace("x", ",").replace("X", ",")
    for item in normalized.replace(";", " ").split():
        parts = [p.strip() for p in item.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"expected mode pair like 1,3, got {item!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def plane_data(data: np.ndarray, params: dict, plane: str):
    _nt, nx, ny, nz = data.shape
    cx, cy, cz = source_center_saved(params, (nx, ny, nz))
    ix = int(np.clip(round(cx), 0, nx - 1))
    iy = int(np.clip(round(cy), 0, ny - 1))
    iz = int(np.clip(round(cz), 0, nz - 1))
    if plane == "XY":
        return data[:, :, :, iz], (cx, cy), ("X saved index", "Y saved index")
    if plane == "YZ":
        return data[:, ix, :, :], (cy, cz), ("Y saved index", "Z saved index")
    if plane == "XZ":
        return data[:, :, iy, :], (cx, cz), ("X saved index", "Z saved index")
    raise ValueError(f"unknown plane: {plane}")


def plane_center_for_directivity(params: dict, shape_xyz, plane: str, center_mode: str):
    cx, cy, cz = analysis_center_saved(params, shape_xyz, center_mode)
    if plane == "YZ":
        return (cy, cz)
    if plane == "XZ":
        return (cx, cz)
    raise ValueError(f"unknown directivity plane: {plane}")


def bilinear_series(plane: np.ndarray, x: float, y: float) -> np.ndarray:
    _, nx, ny = plane.shape
    x0 = int(np.clip(math.floor(x), 0, nx - 2))
    y0 = int(np.clip(math.floor(y), 0, ny - 2))
    wx = x - x0
    wy = y - y0
    return (
        (1 - wx) * (1 - wy) * plane[:, x0, y0]
        + wx * (1 - wy) * plane[:, x0 + 1, y0]
        + (1 - wx) * wy * plane[:, x0, y0 + 1]
        + wx * wy * plane[:, x0 + 1, y0 + 1]
    )


def band_energy(wave: np.ndarray, dt: float, freq_hz: float, band_hz: float) -> float:
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - np.mean(wave)
    spec = np.fft.rfft(wave * np.hanning(len(wave)))
    freqs = np.fft.rfftfreq(len(wave), d=dt)
    mask = (freqs >= freq_hz - band_hz * 0.5) & (freqs <= freq_hz + band_hz * 0.5)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(np.abs(spec[mask]) ** 2)))


def smooth_circular(values: np.ndarray, half_width: int) -> np.ndarray:
    if half_width <= 0:
        return values
    half_width = int(half_width)
    padded = np.r_[values[-half_width:], values, values[:half_width]]
    kernel = np.ones(2 * half_width + 1, dtype=np.float64) / float(2 * half_width + 1)
    return np.convolve(padded, kernel, mode="same")[half_width:-half_width]


def directivity_curve(data: np.ndarray, params: dict, dt: float, freq_hz: float, plane_name: str, args):
    plane, _, _ = plane_data(data, params, plane_name)
    center = plane_center_for_directivity(params, data.shape[1:], plane_name, args.directivity_center)
    d1 = saved_spacing(params, "y" if plane_name == "YZ" else "x")
    radius_grid = args.radius_m / d1
    angles = np.deg2rad(np.arange(0.0, 360.0 + args.angle_step_deg, args.angle_step_deg))
    amps = []
    for th in angles:
        p1 = center[0] + radius_grid * math.sin(float(th))
        p2 = center[1] - radius_grid * math.cos(float(th))
        amps.append(band_energy(bilinear_series(plane, p1, p2), dt, freq_hz, args.band_hz))
    amps = np.asarray(amps, dtype=np.float64)
    amps = smooth_circular(amps, args.smooth_angles)
    amps_db = 20.0 * np.log10(amps / (np.max(amps) + 1e-30) + 1e-12)
    amps_db = np.maximum(amps_db, args.db_floor)
    return angles, amps_db - args.db_floor


def _resolve_farfield_sidecar_path(meta_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    by_meta_name = meta_path.parent / candidate.name
    if by_meta_name.exists():
        return by_meta_name.resolve()
    by_bundle_data = meta_path.parent.parent / "data" / candidate.name
    if by_bundle_data.exists():
        return by_bundle_data.resolve()
    if candidate.is_absolute():
        return candidate
    return (meta_path.parent / candidate).resolve()


def reconstruct_farfield_surface_geometry(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    center = payload.get("center_raw_indices")
    raw = payload.get("surface_raw_indices", {})
    if center is None or not all(axis in raw for axis in ("x", "y", "z")):
        raise FileNotFoundError("coords/normals sidecars are missing and metadata is insufficient to rebuild them")
    stride = int(payload.get("surface_stride_cells", 1))
    spacing = payload.get("surface_spacing_m", [1.0, 1.0, 1.0])
    dx = float(spacing[0]) / stride
    dy = float(spacing[1]) / stride
    dz = float(spacing[2]) / stride
    cx, cy, cz = [int(v) for v in center]
    ix0, ix1 = [int(v) for v in raw["x"]]
    iy0, iy1 = [int(v) for v in raw["y"]]
    iz0, iz1 = [int(v) for v in raw["z"]]
    x_idx = np.arange(ix0, ix1 + 1, stride, dtype=np.int64)
    y_idx = np.arange(iy0, iy1 + 1, stride, dtype=np.int64)
    z_idx = np.arange(iz0, iz1 + 1, stride, dtype=np.int64)
    xs = (x_idx - cx).astype(np.float32) * dx
    ys = (y_idx - cy).astype(np.float32) * dy
    zs = (z_idx - cz).astype(np.float32) * dz
    coords = []
    normals = []

    def add_face(coord_array: np.ndarray, normal: tuple[float, float, float]) -> None:
        flat = coord_array.reshape(-1, 3).astype(np.float32)
        coords.append(flat)
        normals.append(np.tile(np.asarray(normal, dtype=np.float32), (flat.shape[0], 1)))

    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    add_face(np.stack([np.full_like(yy, (ix0 - cx) * dx), yy, zz], axis=-1), (-1, 0, 0))
    add_face(np.stack([np.full_like(yy, (ix1 - cx) * dx), yy, zz], axis=-1), (1, 0, 0))
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    add_face(np.stack([xx, np.full_like(xx, (iy0 - cy) * dy), zz], axis=-1), (0, -1, 0))
    add_face(np.stack([xx, np.full_like(xx, (iy1 - cy) * dy), zz], axis=-1), (0, 1, 0))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    add_face(np.stack([xx, yy, np.full_like(xx, (iz0 - cz) * dz)], axis=-1), (0, 0, -1))
    add_face(np.stack([xx, yy, np.full_like(xx, (iz1 - cz) * dz)], axis=-1), (0, 0, 1))
    return np.concatenate(coords, axis=0), np.concatenate(normals, axis=0)


def load_or_rebuild_farfield_geometry(meta_path: Path, payload: dict, coords_path: Path, normals_path: Path):
    if coords_path.exists() and normals_path.exists():
        return np.load(coords_path), np.load(normals_path), False
    coords, normals = reconstruct_farfield_surface_geometry(payload)
    coords_out = meta_path.parent / Path(payload["coords_path"]).name
    normals_out = meta_path.parent / Path(payload["normals_path"]).name
    np.save(coords_out, coords)
    np.save(normals_out, normals)
    return coords, normals, True


def load_farfield_surface_package(meta_path: Path):
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    p_path = _resolve_farfield_sidecar_path(meta_path, payload["p_path"])
    un_path = _resolve_farfield_sidecar_path(meta_path, payload["un_path"])
    coords_path = _resolve_farfield_sidecar_path(meta_path, payload["coords_path"])
    normals_path = _resolve_farfield_sidecar_path(meta_path, payload["normals_path"])
    coords, normals, geometry_rebuilt = load_or_rebuild_farfield_geometry(meta_path, payload, coords_path, normals_path)
    package = {
        "meta_path": meta_path,
        "meta": payload,
        "p": zarr.open(str(p_path), mode="r"),
        "un": zarr.open(str(un_path), mode="r"),
        "coords": coords,
        "normals": normals,
        "geometry_rebuilt": bool(geometry_rebuilt),
        "dt": float(payload["dt_surface_sec"]),
        "dA": float(payload["surface_element_area_m2"]),
        "frames_written": int(payload.get("frames_written", payload.get("frame_count", 0))),
    }
    if package["frames_written"] <= 1:
        raise ValueError(f"farfield surface has too few frames: {package['frames_written']}")
    return package


def surface_complex_spectrum_at_freq(surface: dict, freq_hz: float, chunk_points: int) -> tuple[np.ndarray, np.ndarray]:
    p_spec, un_spec = surface_complex_spectra_at_freqs(surface, [freq_hz], chunk_points)
    return p_spec[0], un_spec[0]


def surface_complex_spectra_at_freqs(surface: dict, freqs_hz, chunk_points: int) -> tuple[np.ndarray, np.ndarray]:
    p_mem = surface["p"]
    un_mem = surface["un"]
    nframes = min(int(surface["frames_written"]), int(p_mem.shape[0]), int(un_mem.shape[0]))
    dt = float(surface["dt"])
    window = np.hanning(nframes).astype(np.float64)
    t_axis = np.arange(nframes, dtype=np.float64) * dt
    freqs = np.asarray(list(freqs_hz), dtype=np.float64)
    if freqs.size == 0:
        raise ValueError("empty farfield frequency list")
    kernel = window[:, None] * np.exp(-2j * np.pi * t_axis[:, None] * freqs[None, :])
    npoints = int(p_mem.shape[1])
    p_spec = np.empty((freqs.size, npoints), dtype=np.complex128)
    un_spec = np.empty((freqs.size, npoints), dtype=np.complex128)
    for start in range(0, npoints, int(chunk_points)):
        end = min(npoints, start + int(chunk_points))
        p_block = np.asarray(p_mem[:nframes, start:end], dtype=np.float64)
        un_block = np.asarray(un_mem[:nframes, start:end], dtype=np.float64)
        p_block -= p_block.mean(axis=0, keepdims=True)
        un_block -= un_block.mean(axis=0, keepdims=True)
        p_spec[:, start:end] = kernel.T @ p_block
        un_spec[:, start:end] = kernel.T @ un_block
    return p_spec, un_spec


def farfield_direction_vectors(plane_name: str, angles: np.ndarray) -> np.ndarray:
    if plane_name == "YZ":
        return np.stack(
            [
                np.zeros_like(angles, dtype=np.float64),
                np.sin(angles, dtype=np.float64),
                -np.cos(angles, dtype=np.float64),
            ],
            axis=1,
        )
    if plane_name == "XZ":
        return np.stack(
            [
                np.sin(angles, dtype=np.float64),
                np.zeros_like(angles, dtype=np.float64),
                -np.cos(angles, dtype=np.float64),
            ],
            axis=1,
        )
    raise ValueError(f"unknown farfield plane: {plane_name}")


def farfield_raw_amps_from_surface_spectra(
    surface: dict,
    freq_hz: float,
    p_spec: np.ndarray,
    un_spec: np.ndarray,
    plane_name: str,
    args,
):
    angles = np.deg2rad(np.arange(0.0, 360.0 + args.angle_step_deg, args.angle_step_deg))
    dirs = farfield_direction_vectors(plane_name, angles)
    coords = np.asarray(surface["coords"], dtype=np.float64)
    normals = np.asarray(surface["normals"], dtype=np.float64)
    omega = 2.0 * math.pi * float(freq_hz)
    k = 2.0 * math.pi * float(freq_hz) / C_AIR
    dA = float(surface["dA"])
    accum = np.zeros(len(angles), dtype=np.complex128)
    chunk_points = int(args.farfield_chunk_points)
    for start in range(0, coords.shape[0], chunk_points):
        end = min(coords.shape[0], start + chunk_points)
        coords_chunk = coords[start:end]
        normals_chunk = normals[start:end]
        p_chunk = p_spec[start:end]
        un_chunk = un_spec[start:end]
        phase = np.exp(-1j * k * (coords_chunk @ dirs.T))
        n_dot = normals_chunk @ dirs.T
        dpdn = -1j * omega * RHO_AIR * un_chunk
        source = dpdn[:, None] + (1j * k * n_dot * p_chunk[:, None])
        accum += np.sum(source * phase, axis=0) * dA
    return angles, np.abs(accum)


def farfield_curve_from_surface_spectra(
    surface: dict,
    freq_hz: float,
    p_spec: np.ndarray,
    un_spec: np.ndarray,
    plane_name: str,
    args,
):
    angles, amps = farfield_raw_amps_from_surface_spectra(surface, freq_hz, p_spec, un_spec, plane_name, args)
    amps = smooth_circular(amps, args.smooth_angles)
    amps_db = 20.0 * np.log10(amps / (np.max(amps) + 1e-30) + 1e-12)
    amps_db = np.maximum(amps_db, args.db_floor)
    return angles, amps_db - args.db_floor


def farfield_band_freqs(freq_hz: float, args) -> np.ndarray:
    samples = int(getattr(args, "farfield_band_samples", FARFIELD_BAND_SAMPLES))
    band_hz = float(getattr(args, "farfield_band_hz", FARFIELD_BAND_HZ))
    if samples <= 1 or band_hz <= 0.0:
        return np.asarray([float(freq_hz)], dtype=np.float64)
    return np.linspace(float(freq_hz) - 0.5 * band_hz, float(freq_hz) + 0.5 * band_hz, samples)


def farfield_curve_from_surface_band(surface: dict, freq_hz: float, plane_name: str, args, spectra_cache: dict):
    freqs = farfield_band_freqs(freq_hz, args)
    if len(freqs) == 1:
        key = float(freqs[0])
        if key not in spectra_cache:
            spectra_cache[key] = surface_complex_spectrum_at_freq(surface, key, args.farfield_chunk_points)
        p_spec, un_spec = spectra_cache[key]
        return farfield_curve_from_surface_spectra(surface, key, p_spec, un_spec, plane_name, args)

    angles = np.deg2rad(np.arange(0.0, 360.0 + args.angle_step_deg, args.angle_step_deg))
    power = np.zeros(len(angles), dtype=np.float64)
    for freq in freqs:
        key = float(freq)
        if key not in spectra_cache:
            spectra_cache[key] = surface_complex_spectrum_at_freq(surface, key, args.farfield_chunk_points)
        p_spec, un_spec = spectra_cache[key]
        _, amps = farfield_raw_amps_from_surface_spectra(surface, key, p_spec, un_spec, plane_name, args)
        power += amps * amps
    amps = np.sqrt(power / float(len(freqs)))
    amps = smooth_circular(amps, args.smooth_angles)
    amps_db = 20.0 * np.log10(amps / (np.max(amps) + 1e-30) + 1e-12)
    amps_db = np.maximum(amps_db, args.db_floor)
    return angles, amps_db - args.db_floor


pu = SimpleNamespace(
    load_package=load_package,
    saved_dt=saved_dt,
    observation_wave=observation_wave,
    front_membrane_theory_freq=front_membrane_theory_freq,
    parse_mode_pairs=parse_mode_pairs,
    load_farfield_surface_package=load_farfield_surface_package,
    surface_complex_spectra_at_freqs=surface_complex_spectra_at_freqs,
    farfield_curve_from_surface_band=farfield_curve_from_surface_band,
    directivity_curve=directivity_curve,
)


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
        }
    )
    return plt


def mode_label(k: int) -> str:
    return f"{FREQUENCY_LABEL_PREFIX}{k}"


def parse_freqs(text: str | None) -> list[float] | None:
    if text is None or not str(text).strip():
        return None
    freqs = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            freqs.append(float(item))
    return freqs


def load_case(zarr_path: Path):
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    data, params, _analysis, _summary, t, _histories = pu.load_package(zarr_path)


def load_params(zarr_path: Path) -> dict:
    _data, params, _analysis, _summary, _t, _histories = pu.load_package(zarr_path)
    return params


def spectrum_mag(wave: np.ndarray, dt: float, zp_factor: int) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - np.mean(wave)
    n = len(wave)
    nfft = max(n, int(n * max(1, zp_factor)))
    spec = np.fft.rfft(wave * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    return freqs, np.abs(spec)


def db20(x: np.ndarray, ref: float) -> np.ndarray:
    eps = 1e-30
    return 20.0 * np.log10((x + eps) / (ref + eps))


def moving_average_hz(y: np.ndarray, freqs: np.ndarray, smooth_hz: float) -> np.ndarray:
    if smooth_hz <= 0.0 or len(freqs) < 2:
        return y.copy()
    df = float(freqs[1] - freqs[0])
    bins = max(1, int(round(float(smooth_hz) / df)))
    if bins <= 1:
        return y.copy()
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(y, kernel, mode="same")


def load_volume_spectrum(npy_path: Path, radius_m: float, zp_factor: int, smooth_hz: float):
    data, params, dt = load_case(npy_path)
    wave = pu.observation_wave(data, params, radius_m)
    freqs, mag = spectrum_mag(wave, dt, zp_factor)
    ref = float(np.max(mag)) if mag.size else 1.0
    db = db20(mag, ref)
    db_sm = moving_average_hz(db, freqs, smooth_hz)
    return params, freqs, mag, db_sm


def local_maxima_indices(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=np.int64)
    return np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def pick_top_peaks(
    freqs: np.ndarray,
    db_sm: np.ndarray,
    n_modes: int,
    f_min: float,
    f_max: float,
    min_separation_hz: float,
) -> list[float]:
    mask = (freqs >= f_min) & (freqs <= f_max)
    candidates = local_maxima_indices(db_sm)
    candidates = candidates[mask[candidates]]
    if candidates.size == 0:
        raise RuntimeError("no spectrum peaks found in the requested frequency range")

    order = candidates[np.argsort(db_sm[candidates])[::-1]]
    selected: list[int] = []
    for idx in order:
        freq = float(freqs[idx])
        if all(abs(freq - float(freqs[prev])) >= min_separation_hz for prev in selected):
            selected.append(int(idx))
        if len(selected) >= n_modes:
            break
    if len(selected) < n_modes:
        raise RuntimeError(f"only found {len(selected)} separated peaks; requested {n_modes}")
    return sorted(float(freqs[idx]) for idx in selected)


def pick_local_peak(freqs: np.ndarray, db_sm: np.ndarray, target_hz: float, window_hz: float) -> float:
    mask = (freqs >= target_hz - window_hz) & (freqs <= target_hz + window_hz)
    candidates = local_maxima_indices(db_sm)
    candidates = candidates[mask[candidates]]
    if candidates.size == 0:
        return float(target_hz)
    best = int(candidates[np.argmax(db_sm[candidates])])
    return float(freqs[best])


def tension_scale(params_a: dict, params_b: dict, fallback: float | None) -> float:
    if fallback is not None:
        return float(fallback)
    ta = float(params_a.get("tension_n_m", params_a.get("T_tension", 15000.0)))
    tb = float(params_b.get("tension_n_m", params_b.get("T_tension", 10000.0)))
    if ta <= 0.0 or tb <= 0.0:
        return 1.0
    return math.sqrt(tb / ta)


def peak_freqs_for_cases(args: argparse.Namespace) -> tuple[dict, dict, list[float], list[float]]:
    print("loading spectra from volumes for peak-derived frequency selection...")
    params_a, freqs_a_spec, _mag_a, db_a_sm = load_volume_spectrum(
        args.npy_a, args.spectrum_radius_m, args.zp_factor, args.spectrum_smooth_hz
    )
    params_b, freqs_b_spec, _mag_b, db_b_sm = load_volume_spectrum(
        args.npy_b, args.spectrum_radius_m, args.zp_factor, args.spectrum_smooth_hz
    )
    freqs_a = pick_top_peaks(
        freqs_a_spec,
        db_a_sm,
        args.n_peak_modes,
        args.peak_min_hz,
        args.peak_max_hz,
        args.peak_min_separation_hz,
    )
    scale = tension_scale(params_a, params_b, args.b_frequency_scale)
    freqs_b = [pick_local_peak(freqs_b_spec, db_b_sm, f_a * scale, args.match_window_hz) for f_a in freqs_a]
    print(f"frequency scale for B matching: {scale:.6f}")
    return params_a, params_b, freqs_a, freqs_b


def default_farfield_path(npy_path: Path) -> Path:
    return npy_path.with_name(f"{npy_path.stem}_farfield_surface.json")


def decimate_surface_flat(surface: dict, stride: int) -> dict:
    stride = max(1, int(stride))
    if stride <= 1:
        return surface
    out = dict(surface)
    out["p"] = surface["p"][:, ::stride]
    out["un"] = surface["un"][:, ::stride]
    out["coords"] = np.asarray(surface["coords"])[::stride]
    out["normals"] = np.asarray(surface["normals"])[::stride]
    out["dA"] = float(surface["dA"]) * float(stride)
    out["surface_point_stride"] = stride
    return out


def decimate_surface_by_face_grid(surface: dict, grid_stride: int) -> dict:
    """面ごとの2D格子として間引く。選択肢: face-grid / flat"""
    grid_stride = max(1, int(grid_stride))
    if grid_stride <= 1:
        return surface
    face_offsets = surface.get("meta", {}).get("face_offsets")
    if not isinstance(face_offsets, dict):
        return decimate_surface_flat(surface, grid_stride * grid_stride)

    keep_parts = []
    for _face, offsets in face_offsets.items():
        start, end = int(offsets[0]), int(offsets[1])
        count = end - start
        side = int(round(np.sqrt(count)))
        if side * side != count:
            idx = np.arange(start, end, grid_stride * grid_stride, dtype=np.int64)
        else:
            rows = np.arange(0, side, grid_stride, dtype=np.int64)
            cols = np.arange(0, side, grid_stride, dtype=np.int64)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            idx = start + (rr.ravel() * side + cc.ravel())
        keep_parts.append(idx)
    keep = np.concatenate(keep_parts)

    out = dict(surface)
    out["p"] = surface["p"][:, keep]
    out["un"] = surface["un"][:, keep]
    out["coords"] = np.asarray(surface["coords"])[keep]
    out["normals"] = np.asarray(surface["normals"])[keep]
    out["dA"] = float(surface["dA"]) * float(grid_stride * grid_stride)
    out["surface_grid_stride"] = grid_stride
    return out


def theory_freqs(params: dict, pairs: list[tuple[int, int]]) -> list[float]:
    return [pu.front_membrane_theory_freq(params, m, n) for m, n in pairs]


def farfield_band_freqs_for_modes(freqs: list[float], args: argparse.Namespace) -> list[float]:
    out = []
    for freq in freqs:
        cargs = curve_args(args, freq)
        band_hz = float(cargs.farfield_band_hz)
        samples = int(cargs.farfield_band_samples)
        if samples <= 1 or band_hz <= 0.0:
            out.append(float(freq))
        else:
            out.extend(np.linspace(float(freq) - 0.5 * band_hz, float(freq) + 0.5 * band_hz, int(samples)).tolist())
    # 順序を保ってFFT重複を避ける。
    deduped = []
    seen = set()
    for freq in out:
        key = round(float(freq), 9)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(freq))
    return deduped


def seed_farfield_spectra_cache(surface: dict, freqs: list[float], chunk_points: int, cache: dict) -> None:
    missing = [float(freq) for freq in freqs if float(freq) not in cache]
    if not missing:
        return
    p_specs, un_specs = pu.surface_complex_spectra_at_freqs(surface, missing, chunk_points)
    for i, freq in enumerate(missing):
        cache[float(freq)] = (p_specs[i], un_specs[i])


def setup_polar_axis_db(ax, title: str) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    rmax = -DB_FLOOR
    ax.set_rlim(0, rmax)

    ticks = np.arange(0.0, rmax + 1e-9, RGRID_STEP_DB)
    tick_labels = [f"{DB_FLOOR + t:.0f}" for t in ticks]

    ax.set_rgrids(ticks, labels=tick_labels, angle=22.5, fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=25)


def curve_args(args: argparse.Namespace, freq_hz: float | None = None) -> SimpleNamespace:
    """極座標計算用の引数をまとめる。選択肢: 通常帯域 / 高周波安定化"""
    band_hz = float(args.band_hz)
    band_samples = int(args.farfield_band_samples)
    smooth_angles = int(args.smooth_angles)
    if (
        freq_hz is not None
        and bool(args.stabilize_high_freq)
        and float(freq_hz) >= float(args.high_freq_stabilize_min_hz)
    ):
        band_hz = float(args.high_freq_band_hz)
        band_samples = int(args.high_freq_band_samples)
        smooth_angles = int(args.high_freq_smooth_angles)

    return SimpleNamespace(
        radius_m=float(args.radius_m),
        angle_step_deg=float(args.angle_step_deg),
        db_floor=float(args.db_floor),
        band_hz=band_hz,
        farfield_band_hz=band_hz,
        farfield_band_samples=band_samples,
        farfield_chunk_points=int(args.farfield_chunk_points),
        smooth_angles=smooth_angles,
        directivity_center=str(args.directivity_center),
    )


def polar_curve(
    source,
    params,
    dt,
    freq_hz: float,
    plane_type: str,
    cargs: SimpleNamespace,
    method: str,
    spectra_cache: dict,
):
    if method == "farfield":
        return pu.farfield_curve_from_surface_band(source, freq_hz, plane_type, cargs, spectra_cache)
    return pu.directivity_curve(source, params, dt, freq_hz, plane_type, cargs)


def cached_polar_curve(
    cache: dict,
    key,
    source,
    params,
    dt,
    freq_hz: float,
    plane_type: str,
    cargs: SimpleNamespace,
    method: str,
    spectra_cache: dict,
):
    if key not in cache:
        cache[key] = polar_curve(source, params, dt, freq_hz, plane_type, cargs, method, spectra_cache)
    return cache[key]


def cache_key(case_label: str, plane_type: str, k: int, cargs: SimpleNamespace):
    return (
        case_label,
        plane_type,
        k,
        float(cargs.farfield_band_hz),
        int(cargs.farfield_band_samples),
        int(cargs.smooth_angles),
    )


def plot_mode_pair(
    ax,
    source_a,
    params_a,
    dt_a,
    source_b,
    params_b,
    dt_b,
    freq_a: float,
    freq_b: float,
    mode_index: int,
    plane_type: str,
    label_a: str,
    label_b: str,
    args: argparse.Namespace,
    curve_cache: dict,
    method: str,
    spectra_cache_a: dict,
    spectra_cache_b: dict,
    show_legend: bool = True,
) -> None:
    color = MODE_COLORS[mode_index % len(MODE_COLORS)]
    cargs_a = curve_args(args, freq_a)
    cargs_b = curve_args(args, freq_b)
    ang_a, r_a = cached_polar_curve(
        curve_cache,
        cache_key("A", plane_type, mode_index, cargs_a),
        source_a,
        params_a,
        dt_a,
        freq_a,
        plane_type,
        cargs_a,
        method,
        spectra_cache_a,
    )
    ang_b, r_b = cached_polar_curve(
        curve_cache,
        cache_key("B", plane_type, mode_index, cargs_b),
        source_b,
        params_b,
        dt_b,
        freq_b,
        plane_type,
        cargs_b,
        method,
        spectra_cache_b,
    )
    ax.plot(ang_a, r_a, color=color, linestyle=LS_A, linewidth=3.0, alpha=0.9, label=f"{label_a} ({freq_a:.1f}Hz)")
    ax.plot(ang_b, r_b, color=color, linestyle=LS_B, linewidth=2.5, alpha=0.85, label=f"{label_b} ({freq_b:.1f}Hz)")
    if show_legend:
        ax.legend(loc="lower left", fontsize=max(10, FONT_SIZE_TICK - 2), frameon=False)


def save_all_modes(
    source_a,
    params_a,
    dt_a,
    source_b,
    params_b,
    dt_b,
    freqs_a: list[float],
    freqs_b: list[float],
    label_a: str,
    label_b: str,
    out_dir: Path,
    out_name: str,
    args: argparse.Namespace,
    dpi: int,
    curve_cache: dict,
    method: str,
    spectra_cache_a: dict,
    spectra_cache_b: dict,
) -> Path:
    plt = require_matplotlib()
    n_modes = min(len(freqs_a), len(freqs_b))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={"projection": "polar"})
    planes = [
        {"name": "(a) YZ Plane (Vertical)", "type": "YZ"},
        {"name": "(b) XZ Plane (Horizontal)", "type": "XZ"},
    ]

    for i, plane in enumerate(planes):
        ax = axes[i]
        setup_polar_axis_db(ax, plane["name"])
        for k in range(n_modes):
            plot_mode_pair(
                ax,
                source_a,
                params_a,
                dt_a,
                source_b,
                params_b,
                dt_b,
                freqs_a[k],
                freqs_b[k],
                k,
                plane["type"],
                f"{mode_label(k + 1)} {label_a}" if i == 1 else "_nolegend_",
                f"{mode_label(k + 1)} {label_b}" if i == 1 else "_nolegend_",
                args,
                curve_cache,
                method,
                spectra_cache_a,
                spectra_cache_b,
                show_legend=False,
            )

    handles, legend_labels = axes[1].get_legend_handles_labels()
    labels = [text for text in legend_labels if not text.startswith("_")]
    handles = [handle for handle, text in zip(handles, legend_labels) if not text.startswith("_")]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.2)
    path = out_dir / out_name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
    plt.close(fig)
    return path


def save_individual_modes(
    source_a,
    params_a,
    dt_a,
    source_b,
    params_b,
    dt_b,
    freqs_a: list[float],
    freqs_b: list[float],
    label_a: str,
    label_b: str,
    out_dir: Path,
    args: argparse.Namespace,
    dpi: int,
    curve_cache: dict,
    method: str,
    spectra_cache_a: dict,
    spectra_cache_b: dict,
) -> list[Path]:
    plt = require_matplotlib()
    n_modes = min(len(freqs_a), len(freqs_b))
    planes = [
        {"name": "(a) YZ Plane (Vertical)", "type": "YZ"},
        {"name": "(b) XZ Plane (Horizontal)", "type": "XZ"},
    ]
    outputs = []

    for k_plot in range(n_modes):
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={"projection": "polar"})
        for i, plane in enumerate(planes):
            ax = axes[i]
            setup_polar_axis_db(ax, f"{plane['name']} {mode_label(k_plot + 1)}")
            plot_mode_pair(
                ax,
                source_a,
                params_a,
                dt_a,
                source_b,
                params_b,
                dt_b,
                freqs_a[k_plot],
                freqs_b[k_plot],
                k_plot,
                plane["type"],
                label_a,
                label_b,
                args,
                curve_cache,
                method,
                spectra_cache_a,
                spectra_cache_b,
                show_legend=True,
            )
        plt.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.08, wspace=0.2)
        path = out_dir / f"comparison_polar_M{k_plot + 1}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
        plt.close(fig)
        outputs.append(path)
    return outputs


def save_mode_grid_by_plane(
    source_a,
    params_a,
    dt_a,
    source_b,
    params_b,
    dt_b,
    freqs_a: list[float],
    freqs_b: list[float],
    label_a: str,
    label_b: str,
    out_dir: Path,
    args: argparse.Namespace,
    dpi: int,
    curve_cache: dict,
    method: str,
    spectra_cache_a: dict,
    spectra_cache_b: dict,
) -> list[Path]:
    plt = require_matplotlib()
    n_modes = min(4, len(freqs_a), len(freqs_b))
    layout = [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]
    planes = [
        ("YZ", "YZ Plane (Vertical)"),
        ("XZ", "XZ Plane (Horizontal)"),
    ]
    outputs = []

    for plane_type, plane_title in planes:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={"projection": "polar"})
        for row, col, k_plot in layout:
            ax = axes[row, col]
            if k_plot >= n_modes:
                ax.set_visible(False)
                continue
            title = f"{mode_label(k_plot + 1)} {plane_title}"
            setup_polar_axis_db(ax, title)
            plot_mode_pair(
                ax,
                source_a,
                params_a,
                dt_a,
                source_b,
                params_b,
                dt_b,
                freqs_a[k_plot],
                freqs_b[k_plot],
                k_plot,
                plane_type,
                label_a,
                label_b,
                args,
                curve_cache,
                method,
                spectra_cache_a,
                spectra_cache_b,
                show_legend=True,
            )

        plt.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04, wspace=0.18, hspace=0.18)
        path = out_dir / f"comparison_polar_modes_grid_{plane_type}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
        plt.close(fig)
        outputs.append(path)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare polar directivity from two final_bundle NPY files. "
            "Far-field mode uses the in-file closed-surface p/u_n method. No GIF output."
        )
    )
    p.add_argument("--npy-a", type=Path, default=DEFAULT_NPY_A)
    p.add_argument("--npy-b", type=Path, default=DEFAULT_NPY_B)
    # 指向性計算方式: farfield / volume
    p.add_argument(
        "--method",
        choices=["farfield", "volume"],
        default="farfield",
        help="farfield uses the saved p/u_n closed surface; volume samples pressure directly from the NPY field.",
    )
    p.add_argument("--farfield-a", type=Path, default=None, help="Override A far-field surface JSON path.")
    p.add_argument("--farfield-b", type=Path, default=None, help="Override B far-field surface JSON path.")
    p.add_argument("--label-a", default=LABEL_A)
    p.add_argument("--label-b", default=LABEL_B)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-name", default=None)
    # 周波数選択: mode / peak
    p.add_argument(
        "--frequency-mode",
        choices=["mode", "peak"],
        default=DEFAULT_FREQUENCY_MODE,
        help=(
            "mode plots theoretical membrane modes for paper figures; "
            "peak derives supplemental diagnostic frequencies from volume spectra."
        ),
    )
    p.add_argument("--mode-pairs", default="1,1;1,3;1,5;1,7")
    p.add_argument("--freqs-a", default=None, help="Comma-separated override frequencies for A.")
    p.add_argument("--freqs-b", default=None, help="Comma-separated override frequencies for B.")
    p.add_argument("--n-peak-modes", type=int, default=N_PEAK_MODES)
    p.add_argument("--peak-min-hz", type=float, default=PEAK_MIN_HZ)
    p.add_argument("--peak-max-hz", type=float, default=PEAK_MAX_HZ)
    p.add_argument("--peak-min-separation-hz", type=float, default=PEAK_MIN_SEPARATION_HZ)
    p.add_argument("--match-window-hz", type=float, default=MATCH_WINDOW_HZ)
    p.add_argument("--b-frequency-scale", type=float, default=None)
    p.add_argument(
        "--spectrum-radius-m",
        type=float,
        default=PEAK_SPECTRUM_RADIUS_M,
        help="Observation radius used only when --frequency-mode peak derives frequencies from volume spectra.",
    )
    p.add_argument("--spectrum-smooth-hz", type=float, default=SPECTRUM_SMOOTH_HZ)
    p.add_argument("--zp-factor", type=int, default=ZERO_PAD_FACTOR)
    p.add_argument(
        "--radius-m",
        type=float,
        default=RADIUS_M,
        help="Observation radius for --method volume. Far-field plots use the closed-surface geometry instead.",
    )
    p.add_argument("--angle-step-deg", type=float, default=ANGLE_STEP_DEG, help="Angular resolution in degrees.")
    p.add_argument("--db-floor", type=float, default=DB_FLOOR)
    p.add_argument("--band-hz", type=float, default=BAND_HZ, help="Narrowband RMS span for polar curves.")
    p.add_argument(
        "--farfield-band-samples",
        type=int,
        default=FARFIELD_BAND_SAMPLES,
        help="Number of frequencies sampled across --band-hz. Default 3 matches the closed-surface analyzer.",
    )
    p.add_argument("--farfield-chunk-points", type=int, default=FARFIELD_CHUNK_POINTS)
    p.add_argument(
        "--surface-grid-stride",
        type=int,
        default=SURFACE_GRID_STRIDE,
        help=(
            "Use every Nth row/column on each far-field surface face. "
            "Default 1 keeps the full surface; "
            "larger values are only for quick previews."
        ),
    )
    p.add_argument(
        "--smooth-angles",
        type=int,
        default=SMOOTH_ANGLES,
        help="Circular smoothing half-width in angle samples. Default 2 matches the closed-surface analyzer.",
    )
    # 高周波安定化: --stabilize-high-freq / --no-stabilize-high-freq
    p.add_argument(
        "--stabilize-high-freq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use wider band averaging and stronger angular smoothing for high-frequency modes.",
    )
    p.add_argument("--high-freq-stabilize-min-hz", type=float, default=HIGH_FREQ_STABILIZE_MIN_HZ)
    p.add_argument("--high-freq-band-hz", type=float, default=HIGH_FREQ_BAND_HZ)
    p.add_argument("--high-freq-band-samples", type=int, default=HIGH_FREQ_BAND_SAMPLES)
    p.add_argument("--high-freq-smooth-angles", type=int, default=HIGH_FREQ_SMOOTH_ANGLES)
    # 中心位置: body / source
    p.add_argument("--directivity-center", choices=["body", "source"], default=DIRECTIVITY_CENTER)
    p.add_argument("--dpi", type=int, default=SAVE_DPI)
    # 個別図保存: --individual / --no-individual
    p.add_argument(
        "--individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save individual M1-M4 figures. Each figure contains YZ and XZ planes.",
    )
    # グリッド図保存: --mode-grids / --no-mode-grids
    p.add_argument(
        "--mode-grids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 2x2 mode grids as separate YZ and XZ figures.",
    )
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR_PEAK if args.frequency_mode == "peak" else DEFAULT_OUT_DIR
    if args.out_name is None:
        args.out_name = OUT_PNG_PEAK if args.frequency_mode == "peak" else OUT_PNG
    return args


def main() -> None:
    global FREQUENCY_LABEL_PREFIX
    args = parse_args()
    FREQUENCY_LABEL_PREFIX = "P" if args.frequency_mode == "peak" else "M"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"frequency mode: {args.frequency_mode}")
    print(f"polar method: {args.method}")

    peak_params_a = None
    peak_params_b = None
    peak_freqs_a = None
    peak_freqs_b = None
    if args.frequency_mode == "peak":
        peak_params_a, peak_params_b, peak_freqs_a, peak_freqs_b = peak_freqs_for_cases(args)

    if args.method == "farfield":
        farfield_a = args.farfield_a or default_farfield_path(args.npy_a)
        farfield_b = args.farfield_b or default_farfield_path(args.npy_b)
        print(f"loading A far-field surface: {farfield_a}")
        source_a = decimate_surface_by_face_grid(pu.load_farfield_surface_package(farfield_a), args.surface_grid_stride)
        params_a = peak_params_a if peak_params_a is not None else load_params(args.npy_a)
        dt_a = None
        print(f"  surface points used: {source_a['coords'].shape[0]} (grid stride={args.surface_grid_stride})")
        print(f"loading B far-field surface: {farfield_b}")
        source_b = decimate_surface_by_face_grid(pu.load_farfield_surface_package(farfield_b), args.surface_grid_stride)
        params_b = peak_params_b if peak_params_b is not None else load_params(args.npy_b)
        dt_b = None
        print(f"  surface points used: {source_b['coords'].shape[0]} (grid stride={args.surface_grid_stride})")
    else:
        print(f"loading A volume: {args.npy_a}")
        source_a, params_a, dt_a = load_case(args.npy_a)
        print(f"  shape={source_a.shape}, saved_dt={dt_a:.6e}s")
        print(f"loading B volume: {args.npy_b}")
        source_b, params_b, dt_b = load_case(args.npy_b)
        print(f"  shape={source_b.shape}, saved_dt={dt_b:.6e}s")

    if args.frequency_mode == "peak":
        freqs_a = parse_freqs(args.freqs_a) or list(peak_freqs_a or [])
        freqs_b = parse_freqs(args.freqs_b) or list(peak_freqs_b or [])
    else:
        pairs = pu.parse_mode_pairs(args.mode_pairs)
        freqs_a = parse_freqs(args.freqs_a) or theory_freqs(params_a, pairs)
        freqs_b = parse_freqs(args.freqs_b) or theory_freqs(params_b, pairs)
    n_modes = min(len(freqs_a), len(freqs_b))
    freqs_a = freqs_a[:n_modes]
    freqs_b = freqs_b[:n_modes]

    selection_label = "peak frequencies" if args.frequency_mode == "peak" else "theoretical modes"
    print(f"\nFinal {selection_label} to plot: {n_modes}")
    for k in range(n_modes):
        print(f"  {mode_label(k + 1)}: {args.label_a}={freqs_a[k]:.1f}Hz, {args.label_b}={freqs_b[k]:.1f}Hz")

    curve_cache: dict = {}
    spectra_cache_a: dict = {}
    spectra_cache_b: dict = {}
    if args.method == "farfield":
        freqs_a_for_fft = farfield_band_freqs_for_modes(freqs_a, args)
        freqs_b_for_fft = farfield_band_freqs_for_modes(freqs_b, args)
        if args.stabilize_high_freq:
            print(
                "high-frequency stabilization: "
                f"freq>={args.high_freq_stabilize_min_hz:.0f}Hz, "
                f"band={args.high_freq_band_hz:.0f}Hz/{args.high_freq_band_samples} samples, "
                f"smooth={args.high_freq_smooth_angles}"
            )
        print("precomputing A far-field spectra...")
        seed_farfield_spectra_cache(source_a, freqs_a_for_fft, args.farfield_chunk_points, spectra_cache_a)
        print("precomputing B far-field spectra...")
        seed_farfield_spectra_cache(source_b, freqs_b_for_fft, args.farfield_chunk_points, spectra_cache_b)
    out_all = save_all_modes(
        source_a,
        params_a,
        dt_a,
        source_b,
        params_b,
        dt_b,
        freqs_a,
        freqs_b,
        args.label_a,
        args.label_b,
        args.out_dir,
        args.out_name,
        args,
        args.dpi,
        curve_cache,
        args.method,
        spectra_cache_a,
        spectra_cache_b,
    )
    print(f"Saved: {out_all}")

    if args.individual:
        for path in save_individual_modes(
            source_a,
            params_a,
            dt_a,
            source_b,
            params_b,
            dt_b,
            freqs_a,
            freqs_b,
            args.label_a,
            args.label_b,
            args.out_dir,
            args,
            args.dpi,
            curve_cache,
            args.method,
            spectra_cache_a,
            spectra_cache_b,
        ):
            print(f"Saved: {path}")

    if args.mode_grids:
        for path in save_mode_grid_by_plane(
            source_a,
            params_a,
            dt_a,
            source_b,
            params_b,
            dt_b,
            freqs_a,
            freqs_b,
            args.label_a,
            args.label_b,
            args.out_dir,
            args,
            args.dpi,
            curve_cache,
            args.method,
            spectra_cache_a,
            spectra_cache_b,
        ):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
