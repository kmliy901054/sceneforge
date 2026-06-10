"""sceneforge/augment/lerobot_io.py — minimal NATIVE LeRobot v2.x dataset I/O
plus the episode-dataset augmenter built on the restyle pipeline.

WHY NATIVE (design decision): SceneForge only needs to (a) enumerate episodes,
(b) decode/encode the per-episode camera mp4s, and (c) copy parquet/meta files
verbatim. The ``lerobot`` package is a heavy, fast-churning dependency (pulls
torch-vision/video stacks, dataset APIs have broken across minor versions, and
the on-disk format itself just jumped v2.1 → v3.0). The v2.x ON-DISK layout,
however, is tiny and stable — three jsonl/json meta files, parquet episodes,
one mp4 per (episode, camera). So this module implements a minimal reader/
writer against the LAYOUT (verified against the official HF examples, e.g.
``lerobot/pusht`` revision ``v2.1``) using pyarrow + ffmpeg subprocesses, and
``scripts/augment_dataset.py`` stays runnable in this repo's pinned env.

Supported layout (codebase_version ``v2.0`` / ``v2.1``)::

    meta/info.json              # totals, fps, path templates, features
    meta/episodes.jsonl         # {"episode_index", "tasks", "length"}
    meta/tasks.jsonl            # {"task_index", "task"}
    meta/episodes_stats.jsonl   # v2.1 per-episode stats   (optional)
    meta/stats.json             # v2.0 aggregate stats     (optional)
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

``augment_dataset`` writes ONE output dataset per style, named
``<source_name>_<style_slug>``: parquet/meta copied verbatim (actions/states/
timestamps bitwise preserved), only the selected cameras' mp4 frames replaced
by the restyled composites (near pixels bitwise identical per frame, the
restyle invariant). Output videos are re-encoded with libx264/yuv420p and the
feature's ``video.codec`` meta field is updated accordingly (AV1 encoders are
not assumed present). Stats files are copied untouched — schema-valid; pixel
stats of restyled cameras are then approximate, recorded in provenance.

Datasets that store camera frames as parquet-embedded image arrays
(``dtype == "image"``) are explicitly out of scope and raise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
from PIL import Image

from sceneforge import gpu
from sceneforge.config import AppConfig, get_config
from sceneforge.augment import restyle
from sceneforge.augment.restyle import (
    DEFAULT_DILATE_PX,
    DEFAULT_KEEP_BELOW_FRAC,
    DEFAULT_KEEP_PERCENTILE,
    DEFAULT_SMOOTH_WINDOW,
    build_keep_masks,
    composite_exact,
    control_from_nearness,
    normalize_nearness,
)
from sceneforge.diffusion.pipeline import ForgePipeline, aspect_size
from sceneforge.spec import NEGATIVE_PROMPT

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ("v2.0", "v2.1")
#: codec written by :func:`write_video` (libx264 → "h264" in stream metadata).
OUTPUT_CODEC = "h264"


class LeRobotError(RuntimeError):
    """Unreadable / unsupported / structurally invalid LeRobot dataset."""


# ============================================================== ffmpeg helpers
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise LeRobotError("ffmpeg not found on PATH (required for video I/O)")
    return exe


def _ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise LeRobotError("ffprobe not found on PATH (required for video I/O)")
    return exe


def probe_video(path: Union[str, Path]) -> dict:
    """ffprobe the first video stream → {"width", "height", "fps", "n_frames"}.

    ``n_frames`` counts demuxed packets (codec-agnostic — reliable for AV1/h264
    in mp4, unlike the optional ``nb_frames`` tag).
    """
    cmd = [_ffprobe(), "-v", "error", "-select_streams", "v:0",
           "-count_packets",
           "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
           "-of", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise LeRobotError(f"ffprobe failed for {path}: {proc.stderr.decode(errors='replace')}")
    streams = json.loads(proc.stdout.decode()).get("streams") or []
    if not streams:
        raise LeRobotError(f"{path}: no video stream")
    s = streams[0]
    num, _, den = str(s.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": fps or 10.0, "n_frames": int(s.get("nb_read_packets", 0))}


def read_video(path: Union[str, Path]) -> tuple[list[np.ndarray], float]:
    """Decode a video → (list of RGB uint8 (H, W, 3) frames, fps).

    Decodes via an ffmpeg rawvideo pipe — codec-robust (LeRobot publishes AV1
    mp4s, which the system OpenCV may not decode).
    """
    meta = probe_video(path)
    w, h = meta["width"], meta["height"]
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        raise LeRobotError(f"ffmpeg decode failed for {path}: "
                           f"{proc.stderr.decode(errors='replace')}")
    frame_bytes = w * h * 3
    if not proc.stdout or len(proc.stdout) % frame_bytes:
        raise LeRobotError(f"{path}: decoded {len(proc.stdout)} bytes, not a "
                           f"multiple of {frame_bytes} ({w}x{h} rgb24)")
    arr = np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)
    return [arr[i].copy() for i in range(arr.shape[0])], meta["fps"]


def write_video(frames: Sequence[np.ndarray], path: Union[str, Path],
                fps: float, crf: int = 18) -> str:
    """Encode RGB uint8 frames → mp4 (libx264, yuv420p) via an ffmpeg rawvideo
    pipe; returns the written codec name (:data:`OUTPUT_CODEC`).

    libx264 + ``-crf 18`` is visually lossless-ish and universally decodable;
    AV1 encoders are NOT assumed present, so callers must update the feature's
    ``video.codec`` meta field (``augment_dataset`` does).
    """
    if not frames:
        raise ValueError("write_video: no frames")
    h, w = frames[0].shape[:2]
    if w % 2 or h % 2:
        raise ValueError(f"write_video: yuv420p needs even dimensions, got {w}x{h}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", f"{fps:g}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           str(path)]
    payload = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes()
                       for f in frames)
    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=1800)
    if proc.returncode != 0 or not path.is_file():
        raise LeRobotError(f"ffmpeg encode failed for {path}: "
                           f"{proc.stderr.decode(errors='replace')}")
    return OUTPUT_CODEC


# ================================================================== the reader
class LeRobotDataset:
    """Minimal native reader for a LeRobot v2.x dataset directory.

    Parses ``meta/*`` eagerly (cheap), resolves parquet/video paths from the
    info.json templates, and keeps the RAW jsonl line per episode so meta files
    can be re-emitted bitwise for selected episodes.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise LeRobotError(f"{self.root}: not a LeRobot dataset "
                               f"(missing meta/info.json)")
        self.info: dict = json.loads(info_path.read_text(encoding="utf-8"))
        version = str(self.info.get("codebase_version", "?"))
        if not any(version.startswith(v) for v in SUPPORTED_VERSIONS):
            raise LeRobotError(
                f"{self.root}: codebase_version {version!r} unsupported — this "
                f"reader implements the v2.x layout ({'/'.join(SUPPORTED_VERSIONS)}); "
                f"v3.0 datasets use file-based chunking and are out of scope")
        self.episodes: list[dict] = []      # parsed episodes.jsonl records
        self._episode_lines: dict[int, str] = {}   # episode_index → raw line
        for line in self._read_jsonl_lines("episodes.jsonl"):
            rec = json.loads(line)
            self.episodes.append(rec)
            self._episode_lines[int(rec["episode_index"])] = line
        self.tasks: list[dict] = [json.loads(l)
                                  for l in self._read_jsonl_lines("tasks.jsonl")]

    def _read_jsonl_lines(self, name: str) -> list[str]:
        path = self.root / "meta" / name
        if not path.is_file():
            raise LeRobotError(f"{self.root}: missing meta/{name}")
        return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ------------------------------------------------------------- features
    @property
    def features(self) -> dict:
        return dict(self.info.get("features") or {})

    @property
    def video_keys(self) -> list[str]:
        return [k for k, f in self.features.items()
                if str(f.get("dtype")) == "video"]

    @property
    def image_keys(self) -> list[str]:
        """Parquet-embedded image-array features (out of scope for augment)."""
        return [k for k, f in self.features.items()
                if str(f.get("dtype")) == "image"]

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 10))

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size", 1000))

    def episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    # ---------------------------------------------------------------- paths
    def parquet_path(self, episode_index: int) -> Path:
        return self.root / self.info["data_path"].format(
            episode_chunk=self.episode_chunk(episode_index),
            episode_index=episode_index)

    def video_path(self, episode_index: int, video_key: str) -> Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=self.episode_chunk(episode_index),
            video_key=video_key, episode_index=episode_index)

    def episode_raw_line(self, episode_index: int) -> str:
        return self._episode_lines[episode_index]

    def read_episode_frames(self, episode_index: int, video_key: str
                            ) -> tuple[list[np.ndarray], float]:
        return read_video(self.video_path(episode_index, video_key))

    def resolve_camera(self, name: str) -> str:
        """Camera CLI name → video key: exact match, else unique suffix match
        (``top`` → ``observation.images.top``). Clear error for image-array
        features and unknown/ambiguous names."""
        vkeys = self.video_keys
        if name in vkeys:
            return name
        img_hit = (name in self.image_keys
                   or any(k.endswith(f".{name}") for k in self.image_keys))
        if img_hit:
            raise LeRobotError(
                f"camera {name!r} has dtype 'image' (frames embedded in parquet) "
                f"— non-video image-array datasets are out of scope for "
                f"augment_dataset; only dtype 'video' cameras are supported")
        matches = [k for k in vkeys if k.endswith(f".{name}")]
        if len(matches) == 1:
            return matches[0]
        raise LeRobotError(
            f"camera {name!r} not found{' (ambiguous)' if len(matches) > 1 else ''}; "
            f"video keys: {vkeys or 'NONE — this dataset has no video features'}")


# ================================================================== validator
def validate_dataset(root: Union[str, Path], *, check_frame_counts: bool = True
                     ) -> dict:
    """Structural validity check (our own validator, no lerobot dependency).

    Verifies: meta files parse; info totals match the episodes.jsonl inventory;
    every episode's parquet exists with row count == episode length and uniform
    ``episode_index`` column; every (episode, video_key) mp4 exists and (when
    ``check_frame_counts``) its frame count == episode length. Raises
    :class:`LeRobotError` on the first violation; returns an inventory dict.
    """
    import pyarrow.parquet as pq

    ds = LeRobotDataset(root)
    info = ds.info
    eps = ds.episodes
    if len(eps) != int(info["total_episodes"]):
        raise LeRobotError(f"{root}: total_episodes={info['total_episodes']} but "
                           f"episodes.jsonl has {len(eps)} records")
    total_len = sum(int(e["length"]) for e in eps)
    if total_len != int(info["total_frames"]):
        raise LeRobotError(f"{root}: total_frames={info['total_frames']} but "
                           f"episode lengths sum to {total_len}")
    if len(ds.tasks) != int(info["total_tasks"]):
        raise LeRobotError(f"{root}: total_tasks={info['total_tasks']} but "
                           f"tasks.jsonl has {len(ds.tasks)} records")
    vkeys = ds.video_keys
    expected_videos = len(eps) * len(vkeys)
    if int(info.get("total_videos", expected_videos)) != expected_videos:
        raise LeRobotError(f"{root}: total_videos={info.get('total_videos')} != "
                           f"episodes×video_keys={expected_videos}")

    n_video_frames = 0
    for e in eps:
        idx, length = int(e["episode_index"]), int(e["length"])
        pq_path = ds.parquet_path(idx)
        if not pq_path.is_file():
            raise LeRobotError(f"{root}: missing {pq_path.relative_to(ds.root)}")
        pf = pq.ParquetFile(pq_path)
        if pf.metadata.num_rows != length:
            raise LeRobotError(f"{root}: {pq_path.name} has {pf.metadata.num_rows} "
                               f"rows, episodes.jsonl says length={length}")
        ep_col = pf.read(columns=["episode_index"]).column("episode_index")
        uniq = set(ep_col.to_pylist())
        if uniq != {idx}:
            raise LeRobotError(f"{root}: {pq_path.name} episode_index column "
                               f"{sorted(uniq)} != {idx}")
        for key in vkeys:
            vp = ds.video_path(idx, key)
            if not vp.is_file():
                raise LeRobotError(f"{root}: missing {vp.relative_to(ds.root)}")
            if check_frame_counts:
                n = probe_video(vp)["n_frames"]
                if n != length:
                    raise LeRobotError(f"{root}: {vp.relative_to(ds.root)} has "
                                       f"{n} frames, expected {length}")
                n_video_frames += n
    return {"ok": True, "root": str(ds.root),
            "codebase_version": info.get("codebase_version"),
            "episodes": len(eps), "frames": total_len,
            "video_keys": vkeys, "videos": expected_videos,
            "video_frames_counted": n_video_frames if check_frame_counts else None}


# ================================================================ small utils
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_episode_range(spec: Optional[str]) -> Optional[tuple[int, int]]:
    """CLI ``--episodes "a:b"`` → (a, b) half-open; None passes through."""
    if spec is None or spec == "":
        return None
    a, sep, b = spec.partition(":")
    if not sep:
        raise ValueError(f"--episodes expects 'start:stop', got {spec!r}")
    return int(a or 0), int(b)


def _select_episodes(ds: LeRobotDataset,
                     episodes: Optional[tuple[int, int]]) -> list[dict]:
    if episodes is None:
        return list(ds.episodes)
    a, b = episodes
    sel = [e for e in ds.episodes if a <= int(e["episode_index"]) < b]
    if not sel:
        raise LeRobotError(f"--episodes {a}:{b} selects no episodes "
                           f"(dataset has {len(ds.episodes)})")
    if a != 0:
        logger.warning(
            "episode selection %d:%d is not a prefix — output keeps the SOURCE "
            "episode indices/filenames (parquet is copied verbatim); strict "
            "lerobot loaders expect 0..N-1, so prefer 0:N slices", a, b)
    return sel


def _resolve_styles(style_prompts, n_styles: int, use_llm_styles: bool,
                    task: str, seed: int, cfg: AppConfig
                    ) -> tuple[restyle.StyleList, str]:
    """Same precedence as restyle_frames: explicit > LLM > DEFAULT_STYLES."""
    if style_prompts is not None:
        return restyle._normalize_styles(style_prompts, n_styles), "explicit"
    if use_llm_styles:
        return restyle._llm_styles(task, n_styles, seed, cfg)
    return restyle._default_styles(n_styles), "default"


# ============================================================== the augmenter
def augment_dataset(
    dataset_dir: Union[str, Path],
    out_dir: Union[str, Path],
    cameras: Sequence[str],
    n_styles: int = 2,
    style_prompts: Union[Sequence[Any], Mapping[str, str], None] = None,
    *,
    use_llm_styles: bool = False,
    episodes: Optional[tuple[int, int]] = None,
    keep_percentile: float = DEFAULT_KEEP_PERCENTILE,
    keep_below_frac: Optional[float] = DEFAULT_KEEP_BELOW_FRAC,
    dilate_px: int = DEFAULT_DILATE_PX,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    seed: int = 42,
    task: str = "a robot arm manipulating objects on a cluttered table",
    crf: int = 18,
    write_audit: bool = True,
    cfg: Optional[AppConfig] = None,
    pipeline: Any = None,
    depth_fn: Optional[Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Restyle the chosen camera streams of a LeRobot v2.x dataset into
    ``n_styles`` appearance worlds — one VALID output dataset per style.

    Output layout: ``out_dir/<source_name>_<style_slug>/`` per style, each a
    self-contained LeRobot dataset: parquet + tasks.jsonl + stats copied
    BYTEWISE (actions/states/timestamps untouched), episodes.jsonl filtered to
    the selection (raw source lines), info.json totals/splits adjusted, the
    selected cameras' mp4s replaced by restyled composites (near pixels
    bitwise identical per frame — asserted), and untouched cameras' mp4s
    copied bytewise. A ``provenance.json`` is written into each output root.

    Pipeline per (episode, camera): decode video → depth (once, shared by all
    styles) → per-episode temporally-smoothed keep masks → per style: SDXL
    depth-ControlNet generation with a FIXED seed+prompt (``seed + style_k``)
    → exact composite → libx264 re-encode at the source feature fps.

    ``pipeline``/``depth_fn`` are injectable as in ``restyle_frames`` (tests
    run CPU-only with fakes); a None pipeline builds one ForgePipeline inside
    ``gpu.phase("diffusion")`` for the WHOLE run (one load, many episodes).
    """
    t_start = time.monotonic()
    cfg = cfg if cfg is not None else get_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        logger.info("%s", msg)
        if progress_cb is not None:
            progress_cb(msg)

    ds = LeRobotDataset(dataset_dir)
    if not ds.video_keys:
        raise LeRobotError(
            f"{ds.root}: no dtype='video' features"
            + (f" (image-array features {ds.image_keys} are parquet-embedded "
               f"and out of scope)" if ds.image_keys else ""))
    if not cameras:
        raise LeRobotError("augment_dataset: no cameras given")
    cam_keys = [ds.resolve_camera(c) for c in cameras]
    untouched_keys = [k for k in ds.video_keys if k not in cam_keys]
    selected = _select_episodes(ds, episodes)
    sel_idx = [int(e["episode_index"]) for e in selected]

    styles, style_source = _resolve_styles(style_prompts, n_styles,
                                           use_llm_styles, task, seed, cfg)
    style_seeds = {slug: seed + k for k, (slug, _) in enumerate(styles)}
    _progress(f"dataset {ds.root.name}: {len(selected)} episode(s) × "
              f"{len(cam_keys)} camera(s) × {len(styles)} style(s) "
              f"[{style_source}]")

    # ---- source fingerprint (meta + the parquets we copy) -------------------
    fingerprint = {
        "path": str(ds.root),
        "codebase_version": ds.info.get("codebase_version"),
        "total_episodes": ds.info.get("total_episodes"),
        "total_frames": ds.info.get("total_frames"),
        "sha256": {
            "meta/info.json": _sha256(ds.root / "meta" / "info.json"),
            "meta/episodes.jsonl": _sha256(ds.root / "meta" / "episodes.jsonl"),
            "meta/tasks.jsonl": _sha256(ds.root / "meta" / "tasks.jsonl"),
        },
        "parquet_sha256": {},
    }

    # ---- per-style output scaffolding: meta + verbatim copies ---------------
    out_roots: dict[str, Path] = {}
    new_total_frames = sum(int(e["length"]) for e in selected)
    new_total_chunks = max(ds.episode_chunk(i) for i in sel_idx) + 1
    for slug, _prompt in styles:
        root = out_dir / f"{ds.root.name}_{slug}"
        if root.exists():
            raise LeRobotError(f"output dataset {root} already exists — refusing "
                               f"to overwrite")
        (root / "meta").mkdir(parents=True)
        out_roots[slug] = root

        info = json.loads(json.dumps(ds.info))  # deep copy
        info["total_episodes"] = len(selected)
        info["total_frames"] = new_total_frames
        info["total_videos"] = len(selected) * len(ds.video_keys)
        info["total_chunks"] = new_total_chunks
        info["splits"] = {"train": f"0:{len(selected)}"}
        for key in cam_keys:  # we re-encode with libx264
            feat_info = info["features"][key].setdefault("info", {})
            if "video.codec" in feat_info:
                feat_info["video.codec"] = OUTPUT_CODEC
        (root / "meta" / "info.json").write_text(
            json.dumps(info, indent=4), encoding="utf-8")

        (root / "meta" / "episodes.jsonl").write_text(
            "".join(ds.episode_raw_line(i) + "\n" for i in sel_idx),
            encoding="utf-8")
        shutil.copy2(ds.root / "meta" / "tasks.jsonl", root / "meta" / "tasks.jsonl")
        # stats: copied untouched (schema-valid; restyled-camera pixel stats
        # become approximate — recorded in provenance)
        src_stats_json = ds.root / "meta" / "stats.json"
        if src_stats_json.is_file():
            shutil.copy2(src_stats_json, root / "meta" / "stats.json")
        src_ep_stats = ds.root / "meta" / "episodes_stats.jsonl"
        if src_ep_stats.is_file():
            wanted = set(sel_idx)
            lines = [l for l in src_ep_stats.read_text(encoding="utf-8").splitlines()
                     if l.strip() and int(json.loads(l)["episode_index"]) in wanted]
            (root / "meta" / "episodes_stats.jsonl").write_text(
                "".join(l + "\n" for l in lines), encoding="utf-8")

        for i in sel_idx:
            src_pq = ds.parquet_path(i)
            if not src_pq.is_file():
                raise LeRobotError(f"{ds.root}: missing {src_pq}")
            rel = src_pq.relative_to(ds.root)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pq, root / rel)          # VERBATIM — bitwise
            fingerprint["parquet_sha256"].setdefault(str(rel), _sha256(src_pq))
            for key in untouched_keys:                # untouched cameras: bytewise
                src_v = ds.video_path(i, key)
                relv = src_v.relative_to(ds.root)
                (root / relv).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_v, root / relv)

    audit_dir = out_dir / f"{ds.root.name}_augment_audit"

    # ---- pass 1: decode + depth + masks per (episode, camera) ---------------
    # Depth runs BEFORE the diffusion phase (restyle.py VRAM discipline); the
    # cached nearness/keep arrays are small relative to the decoded frames.
    per_ep: dict[tuple[int, str], dict] = {}
    depth = depth_fn or restyle.estimate_depth
    t_depth = time.monotonic()
    for i in sel_idx:
        for key in cam_keys:
            _progress(f"depth: episode {i} · {key}")
            frames, _fps = ds.read_episode_frames(i, key)
            exp_len = next(int(e["length"]) for e in selected
                           if int(e["episode_index"]) == i)
            if len(frames) != exp_len:
                raise LeRobotError(f"{ds.root}: episode {i} {key} decoded "
                                   f"{len(frames)} frames, meta says {exp_len}")
            nearness = [normalize_nearness(d) for d in depth(frames)]
            keeps = build_keep_masks(nearness, keep_percentile, keep_below_frac,
                                     dilate_px, smooth_window)
            per_ep[(i, key)] = {"frames": frames, "nearness": nearness,
                                "keeps": keeps}
    depth_s = time.monotonic() - t_depth

    # ---- pass 2: one diffusion burst for every (style, episode, camera) -----
    restyle_fracs: list[float] = []
    gen_seconds: list[float] = []

    def _video_fps(key: str) -> float:
        feat = ds.features.get(key) or {}
        return float((feat.get("info") or {}).get("video.fps") or ds.fps)

    def _burst(pipe: Any) -> None:
        for slug, prompt in styles:
            gen_seed = style_seeds[slug]
            for i in sel_idx:
                for key in cam_keys:
                    rec = per_ep[(i, key)]
                    frames, nearness, keeps = (rec["frames"], rec["nearness"],
                                               rec["keeps"])
                    h, w = frames[0].shape[:2]
                    gen_w, gen_h = aspect_size(w, h, cfg.gen.resolution)
                    comps: list[np.ndarray] = []
                    for frame, near, keep in zip(frames, nearness, keeps):
                        control = control_from_nearness(near, (gen_w, gen_h))
                        t1 = time.monotonic()
                        img = pipe.generate(control, prompt, NEGATIVE_PROMPT,
                                            gen_seed,
                                            cond_scale=cfg.gen.cond_scale,
                                            steps=cfg.gen.steps,
                                            size=(gen_w, gen_h))
                        gen_seconds.append(time.monotonic() - t1)
                        gen_np = np.asarray(
                            img.resize((w, h), Image.BICUBIC).convert("RGB"))
                        comp = composite_exact(frame, gen_np, keep)
                        assert np.array_equal(comp[keep], frame[keep]), \
                            "near-pixel composite must be bitwise exact"
                        comps.append(comp.astype(np.uint8))
                        restyle_fracs.append(round(float(1.0 - keep.mean()), 4))
                    out_mp4 = (out_roots[slug] /
                               ds.info["video_path"].format(
                                   episode_chunk=ds.episode_chunk(i),
                                   video_key=key, episode_index=i))
                    write_video(comps, out_mp4, _video_fps(key), crf=crf)
                    if write_audit:
                        restyle._audit_sheet(
                            frames, nearness, keeps, {slug: comps},
                            audit_dir / f"episode_{i:06d}_{key}_{slug}.jpg")
                    _progress(f"restyled: style {slug} · episode {i} · {key} "
                              f"({len(comps)} frames)")

    owns_pipeline = pipeline is None
    t_diff = time.monotonic()
    if owns_pipeline:
        with gpu.phase("diffusion", cfg=cfg):  # Ollama eviction barrier (§10.3)
            pipeline = ForgePipeline(resolution=cfg.gen.resolution)
            pipeline.load(int(cfg.gen.level[1]))
            try:
                _burst(pipeline)
            finally:
                pipeline.unload()
    else:
        _burst(pipeline)  # caller manages VRAM/load state
    diffusion_s = time.monotonic() - t_diff

    # ---- provenance per output dataset + run summary -------------------------
    common = {
        "tool": "sceneforge.augment.lerobot_io.augment_dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": fingerprint,
        "episodes": sel_idx,
        "cameras_restyled": cam_keys,
        "cameras_untouched": untouched_keys,
        "params": {
            "n_styles": len(styles), "keep_percentile": keep_percentile,
            "keep_below_frac": keep_below_frac, "dilate_px": dilate_px,
            "smooth_window": smooth_window, "seed": seed, "crf": crf,
        },
        "depth_model": (restyle.DEPTH_MODEL_ID if depth_fn is None
                        else "injected depth_fn"),
        "style_source": style_source,
        "styles": [{"name": slug, "prompt": prompt, "seed": style_seeds[slug]}
                   for slug, prompt in styles],
        "generation": {
            "pipeline": "ForgePipeline" if owns_pipeline else type(pipeline).__name__,
            "level": cfg.gen.level, "steps": cfg.gen.steps,
            "cond_scale": cfg.gen.cond_scale,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "video_codec_written": OUTPUT_CODEC,
        "stats_inherited_from_source": True,
        "parquet_copied_verbatim": True,
        "near_pixels_bitwise_identical": True,
        "restyle_frac_mean": (round(float(np.mean(restyle_fracs)), 4)
                              if restyle_fracs else None),
    }
    for slug, prompt in styles:
        prov = dict(common)
        prov["style"] = {"name": slug, "prompt": prompt,
                         "seed": style_seeds[slug]}
        prov["output_dataset"] = str(out_roots[slug])
        (out_roots[slug] / "provenance.json").write_text(
            json.dumps(prov, indent=2), encoding="utf-8")

    summary = dict(common)
    summary["outputs"] = {slug: str(root) for slug, root in out_roots.items()}
    summary["audit_dir"] = str(audit_dir) if write_audit else None
    summary["n_frames_restyled"] = new_total_frames * len(cam_keys) * len(styles)
    summary["timings_s"] = {
        "depth": round(depth_s, 3),
        "diffusion": round(diffusion_s, 3),
        "per_image": (round(float(np.mean(gen_seconds)), 3)
                      if gen_seconds else None),
        "total": round(time.monotonic() - t_start, 3),
    }
    _progress(f"done: {len(styles)} output dataset(s) under {out_dir} "
              f"in {summary['timings_s']['total']:.1f}s")
    return summary
