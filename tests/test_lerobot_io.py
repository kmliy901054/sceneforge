"""tests/test_lerobot_io.py — LeRobot v2.x dataset augmentation, CPU-only.

A synthetic mini LeRobot v2.1 dataset (2 episodes × 8 frames, 64×48, REAL
parquet via pyarrow + REAL mp4 via ffmpeg) is built from scratch, then
augmented with an injected fake restyler (no GPU, no downloads). Covers:
read→write roundtrip preserves actions/timestamps BITWISE (parquet bytes),
structural validity of the augmented outputs (our own validator: file
inventory + parquet row counts + video frame counts), style-suffix naming,
untouched-camera bytewise copies, episode slicing, the image-array (non-video)
out-of-scope error, and provenance. Plus the UI smoke test: build_app()
constructs with tabs (Forge + Video Augment) without touching the GPU.
"""
import sceneforge.compat  # noqa: F401  — import FIRST (repo rule, §0)

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pyarrow")
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg/ffprobe required for LeRobot video I/O tests",
                allow_module_level=True)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sceneforge.augment import lerobot_io as lio  # noqa: E402
from sceneforge.config import AppConfig  # noqa: E402

H, W = 48, 64
N_EPISODES, EP_LEN = 2, 8
FPS = 10.0
TOP = "observation.images.top"
WRIST = "observation.images.wrist"
TASK = "push the block to the target"


# ------------------------------------------------------------------- builders
def _episode_frames(ep: int) -> list[np.ndarray]:
    """Deterministic frames with strong near/far structure: dark 'far' upper
    band, bright 'near' lower band, a moving white square as the 'robot'."""
    frames = []
    for t in range(EP_LEN):
        f = np.zeros((H, W, 3), np.uint8)
        f[: H // 2] = (30 + 10 * ep, 40, 60)          # far background
        f[H // 2:] = (170, 160, 150)                  # near workspace
        x = 4 + 6 * t
        f[H - 14: H - 4, x: x + 10] = 255             # moving effector
        frames.append(f)
    return frames


def _video_feature() -> dict:
    return {
        "dtype": "video",
        "shape": [H, W, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": FPS, "video.codec": "h264",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def build_mini_dataset(root: Path, video_keys=(TOP, WRIST)) -> Path:
    """A real on-disk LeRobot v2.1 dataset: meta jsonl + parquet + mp4."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {key: _video_feature() for key in video_keys}
    features.update({
        "observation.state": {"dtype": "float32", "shape": [2],
                              "names": ["motor_0", "motor_1"]},
        "action": {"dtype": "float32", "shape": [2],
                   "names": ["motor_0", "motor_1"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    })
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_bot",
        "total_episodes": N_EPISODES,
        "total_frames": N_EPISODES * EP_LEN,
        "total_tasks": 1,
        "total_videos": N_EPISODES * len(video_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(FPS),
        "splits": {"train": f"0:{N_EPISODES}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": i, "tasks": [TASK], "length": EP_LEN}) + "\n"
        for i in range(N_EPISODES)))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": TASK}) + "\n")
    (meta / "episodes_stats.jsonl").write_text("".join(
        json.dumps({"episode_index": i, "stats": {"action": {
            "mean": [0.0, 0.0], "std": [1.0, 1.0],
            "min": [-1.0, -1.0], "max": [1.0, 1.0], "count": [EP_LEN]}}}) + "\n"
        for i in range(N_EPISODES)))

    rng = np.random.default_rng(11)
    idx0 = 0
    for i in range(N_EPISODES):
        table = pa.table({
            "observation.state": pa.array(
                rng.normal(size=(EP_LEN, 2)).astype(np.float32).tolist(),
                type=pa.list_(pa.float32(), 2)),
            "action": pa.array(
                rng.normal(size=(EP_LEN, 2)).astype(np.float32).tolist(),
                type=pa.list_(pa.float32(), 2)),
            "timestamp": pa.array((np.arange(EP_LEN) / FPS).astype(np.float32)),
            "frame_index": pa.array(np.arange(EP_LEN, dtype=np.int64)),
            "episode_index": pa.array(np.full(EP_LEN, i, dtype=np.int64)),
            "index": pa.array(np.arange(idx0, idx0 + EP_LEN, dtype=np.int64)),
            "task_index": pa.array(np.zeros(EP_LEN, dtype=np.int64)),
            "next.done": pa.array([t == EP_LEN - 1 for t in range(EP_LEN)]),
        })
        idx0 += EP_LEN
        pq_path = root / f"data/chunk-000/episode_{i:06d}.parquet"
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, pq_path)
        for key in video_keys:
            vp = root / f"videos/chunk-000/{key}/episode_{i:06d}.mp4"
            lio.write_video(_episode_frames(i), vp, FPS)
    return root


# ----------------------------------------------------------------------- fakes
class FakePipeline:
    """ForgePipeline stand-in: solid magenta at the requested size; records
    (prompt, seed) per call so fixed-seed-per-style is checkable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, control, prompt, negative="", seed=0,
                 cond_scale=None, steps=None, size=None):
        self.calls.append((prompt, int(seed)))
        w, h = size if size is not None else (64, 64)
        assert control.size == (w, h)
        return Image.new("RGB", (w, h), (255, 0, 255))


def fake_depth(frames):
    """Vertical gradient (Depth-Anything polarity: larger = nearer)."""
    h, w = frames[0].shape[:2]
    col = np.linspace(0.0, 10.0, h, dtype=np.float32)[:, None]
    return [np.repeat(col, w, axis=1) for _ in frames]


STYLES = [{"name": "Neon Bar!", "prompt": "a neon bar, photo"},
          ("lab", "a bright laboratory, photo")]
SLUGS = ["neon_bar", "lab"]


# -------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def src_ds(tmp_path_factory) -> Path:
    return build_mini_dataset(tmp_path_factory.mktemp("lerobot") / "mini")


@pytest.fixture(scope="module")
def augmented(src_ds, tmp_path_factory):
    """One shared fake-restyler augment run over the full mini dataset."""
    out = tmp_path_factory.mktemp("aug")
    pipe = FakePipeline()
    summary = lio.augment_dataset(
        src_ds, out, cameras=["top"], n_styles=2, style_prompts=STYLES,
        seed=7, cfg=AppConfig(), pipeline=pipe, depth_fn=fake_depth,
        smooth_window=3)
    return src_ds, out, pipe, summary


# ----------------------------------------------------------- reader/validator
class TestReaderAndValidator:
    def test_reads_layout(self, src_ds):
        ds = lio.LeRobotDataset(src_ds)
        assert ds.video_keys == [TOP, WRIST]
        assert [e["episode_index"] for e in ds.episodes] == [0, 1]
        assert ds.tasks[0]["task"] == TASK
        assert ds.parquet_path(1).is_file() and ds.video_path(1, TOP).is_file()
        assert ds.resolve_camera("top") == TOP        # suffix resolution
        assert ds.resolve_camera(WRIST) == WRIST      # exact key

    def test_source_dataset_validates(self, src_ds):
        report = lio.validate_dataset(src_ds)
        assert report["ok"] and report["episodes"] == N_EPISODES
        assert report["frames"] == N_EPISODES * EP_LEN
        assert report["video_frames_counted"] == N_EPISODES * EP_LEN * 2

    def test_validator_catches_missing_video(self, tmp_path, src_ds):
        broken = tmp_path / "broken"
        shutil.copytree(src_ds, broken)
        next(iter((broken / "videos/chunk-000" / TOP).glob("*.mp4"))).unlink()
        with pytest.raises(lio.LeRobotError, match="missing"):
            lio.validate_dataset(broken)

    def test_video_roundtrip_count_and_shape(self, tmp_path):
        frames = _episode_frames(0)
        mp4 = tmp_path / "rt.mp4"
        assert lio.write_video(frames, mp4, FPS) == "h264"
        back, fps = lio.read_video(mp4)
        assert len(back) == EP_LEN and back[0].shape == (H, W, 3)
        assert fps == pytest.approx(FPS)
        assert lio.probe_video(mp4)["n_frames"] == EP_LEN

    def test_unsupported_version_rejected(self, tmp_path, src_ds):
        v3 = tmp_path / "v3"
        shutil.copytree(src_ds, v3)
        info = json.loads((v3 / "meta/info.json").read_text())
        info["codebase_version"] = "v3.0"
        (v3 / "meta/info.json").write_text(json.dumps(info))
        with pytest.raises(lio.LeRobotError, match="v3.0"):
            lio.LeRobotDataset(v3)


# ------------------------------------------------- augmented dataset contract
class TestAugmentedOutputs:
    def test_style_suffix_naming(self, augmented):
        src, out, _pipe, summary = augmented
        assert sorted(summary["outputs"]) == sorted(SLUGS)
        for slug in SLUGS:
            assert (out / f"{src.name}_{slug}").is_dir()
            assert Path(summary["outputs"][slug]).name == f"{src.name}_{slug}"

    def test_parquet_bitwise_preserved(self, augmented):
        """Roundtrip invariant: actions/states/timestamps untouched — the
        parquet FILES are byte-identical to the source."""
        src, out, _pipe, _summary = augmented
        for slug in SLUGS:
            for i in range(N_EPISODES):
                rel = f"data/chunk-000/episode_{i:06d}.parquet"
                assert (out / f"{src.name}_{slug}" / rel).read_bytes() == \
                    (src / rel).read_bytes()

    def test_meta_files_and_totals(self, augmented):
        src, out, _pipe, _summary = augmented
        for slug in SLUGS:
            root = out / f"{src.name}_{slug}"
            assert (root / "meta/tasks.jsonl").read_bytes() == \
                (src / "meta/tasks.jsonl").read_bytes()
            assert (root / "meta/episodes.jsonl").read_bytes() == \
                (src / "meta/episodes.jsonl").read_bytes()  # full selection
            assert (root / "meta/episodes_stats.jsonl").read_bytes() == \
                (src / "meta/episodes_stats.jsonl").read_bytes()
            info = json.loads((root / "meta/info.json").read_text())
            assert info["total_episodes"] == N_EPISODES
            assert info["total_frames"] == N_EPISODES * EP_LEN
            assert info["total_videos"] == N_EPISODES * 2
            assert info["features"][TOP]["info"]["video.codec"] == "h264"

    def test_outputs_structurally_valid(self, augmented):
        """Our validator: inventory + parquet rows + video frame counts."""
        src, out, _pipe, _summary = augmented
        for slug in SLUGS:
            report = lio.validate_dataset(out / f"{src.name}_{slug}")
            assert report["ok"]
            assert report["video_frames_counted"] == N_EPISODES * EP_LEN * 2

    def test_untouched_camera_bytewise_restyled_camera_not(self, augmented):
        src, out, _pipe, _summary = augmented
        for slug in SLUGS:
            root = out / f"{src.name}_{slug}"
            for i in range(N_EPISODES):
                wrist = f"videos/chunk-000/{WRIST}/episode_{i:06d}.mp4"
                top = f"videos/chunk-000/{TOP}/episode_{i:06d}.mp4"
                assert (root / wrist).read_bytes() == (src / wrist).read_bytes()
                assert (root / top).read_bytes() != (src / top).read_bytes()

    def test_restyled_frames_near_kept_far_magenta(self, augmented):
        """Decoded output: far region ≈ magenta (fake gen), near region ≈
        source (codec noise only — the bitwise invariant holds pre-encode and
        is asserted inside augment_dataset on every composite)."""
        src, out, _pipe, _summary = augmented
        src_frames, _ = lio.read_video(
            src / f"videos/chunk-000/{TOP}/episode_000000.mp4")
        aug_frames, _ = lio.read_video(
            out / f"{src.name}_{SLUGS[0]}" / f"videos/chunk-000/{TOP}/episode_000000.mp4")
        assert len(aug_frames) == len(src_frames) == EP_LEN
        for t in (0, EP_LEN // 2, EP_LEN - 1):
            near_src = src_frames[t][H - 8:].astype(np.int16)
            near_aug = aug_frames[t][H - 8:].astype(np.int16)
            assert np.abs(near_aug - near_src).mean() < 3.0   # codec noise only
            far = aug_frames[t][:4].astype(np.int16)          # restyled region
            assert np.abs(far - np.array([255, 0, 255])).mean() < 30.0

    def test_fixed_seed_per_style_and_depth_shared(self, augmented):
        _src, _out, pipe, summary = augmented
        # 2 styles × 2 episodes × 8 frames, depth shared across styles
        assert len(pipe.calls) == 2 * N_EPISODES * EP_LEN
        by_prompt: dict[str, set] = {}
        for prompt, s in pipe.calls:
            by_prompt.setdefault(prompt, set()).add(s)
        assert all(len(s) == 1 for s in by_prompt.values())
        assert {s["seed"] for s in summary["styles"]} == {7, 8}

    def test_provenance(self, augmented):
        src, out, _pipe, summary = augmented
        for slug in SLUGS:
            prov = json.loads(
                (out / f"{src.name}_{slug}" / "provenance.json").read_text())
            assert prov["style"]["name"] == slug
            assert prov["parquet_copied_verbatim"] is True
            assert prov["near_pixels_bitwise_identical"] is True
            fp = prov["source_dataset"]
            assert fp["codebase_version"] == "v2.1"
            assert len(fp["sha256"]) == 3 and len(fp["parquet_sha256"]) == 2
            assert prov["params"]["seed"] == 7
            assert prov["style_source"] == "explicit"
        assert summary["restyle_frac_mean"] is not None


# --------------------------------------------------- slicing + scope errors
class TestSelectionAndErrors:
    def test_episode_prefix_slice(self, src_ds, tmp_path):
        out = tmp_path / "sliced"
        summary = lio.augment_dataset(
            src_ds, out, cameras=[TOP], n_styles=1,
            style_prompts=["a forest, photo"], episodes=(0, 1),
            cfg=AppConfig(), pipeline=FakePipeline(), depth_fn=fake_depth,
            smooth_window=3, write_audit=False)
        root = Path(summary["outputs"]["style_1"])
        report = lio.validate_dataset(root)
        assert report["episodes"] == 1 and report["frames"] == EP_LEN
        info = json.loads((root / "meta/info.json").read_text())
        assert info["splits"] == {"train": "0:1"}
        assert len((root / "meta/episodes_stats.jsonl")
                   .read_text().splitlines()) == 1
        assert not (root / "data/chunk-000/episode_000001.parquet").exists()

    def test_image_array_dataset_out_of_scope(self, tmp_path):
        root = build_mini_dataset(tmp_path / "img", video_keys=(TOP,))
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"]["observation.images.cam"] = {
            "dtype": "image", "shape": [H, W, 3], "names": None}
        info_path.write_text(json.dumps(info))
        ds = lio.LeRobotDataset(root)
        with pytest.raises(lio.LeRobotError, match="out of scope"):
            ds.resolve_camera("observation.images.cam")
        with pytest.raises(lio.LeRobotError, match="out of scope"):
            lio.augment_dataset(root, tmp_path / "o", cameras=["cam"],
                                n_styles=1, style_prompts=["x"],
                                cfg=AppConfig(), pipeline=FakePipeline(),
                                depth_fn=fake_depth)

    def test_unknown_camera_errors(self, src_ds, tmp_path):
        with pytest.raises(lio.LeRobotError, match="not found"):
            lio.augment_dataset(src_ds, tmp_path / "o", cameras=["nope"],
                                n_styles=1, style_prompts=["x"],
                                cfg=AppConfig(), pipeline=FakePipeline(),
                                depth_fn=fake_depth)

    def test_refuses_to_overwrite(self, src_ds, tmp_path, augmented):
        _src, out, _pipe, _summary = augmented
        with pytest.raises(lio.LeRobotError, match="already exists"):
            lio.augment_dataset(src_ds, out, cameras=[TOP], n_styles=2,
                                style_prompts=STYLES, cfg=AppConfig(),
                                pipeline=FakePipeline(), depth_fn=fake_depth)

    def test_parse_episode_range(self):
        assert lio.parse_episode_range(None) is None
        assert lio.parse_episode_range("0:10") == (0, 10)
        assert lio.parse_episode_range(":3") == (0, 3)
        with pytest.raises(ValueError):
            lio.parse_episode_range("5")


# ------------------------------------------------------------------- UI smoke
class TestUISmoke:
    def test_build_app_constructs_with_tabs(self):
        import gradio as gr

        from sceneforge.ui.blocks import build_app

        demo = build_app()
        tabs = [b for b in demo.blocks.values() if isinstance(b, gr.Tab)]
        labels = {t.label for t in tabs}
        assert {"Forge", "Video Augment"} <= labels
        # forge tab wiring intact: the FORGE button + its handler still exist
        fns = {getattr(f.fn, "__name__", "") for f in demo.fns.values()}
        assert {"on_forge", "on_reforge", "on_export", "on_toggle",
                "on_timer", "on_video_augment"} <= fns
