"""tests/test_masks.py — labels + gpu acceptance (ARCHITECTURE.md §8, §3.1, §10.3).

Covers: seg→RLE→annToMask round-trip and python-int casts (§8.1); depth16 mm
on-disk round-trip (§3.1); §8.3 COCO acceptance over a synthetic 2-layout ×
2-style export (COCO() loads, annToMask round-trips, globally-unique ann ids,
json.dumps succeeds); overlay modes (§8.2); ollama_unload 400-fallback with
mocked HTTP — no live Ollama dependency (§10.3).
"""
import sceneforge.compat  # noqa: F401  — import FIRST (repo rule, §0)

import dataclasses
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests as real_requests
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

import sceneforge.gpu as gpu
from sceneforge.config import AppConfig, OllamaConfig, VramConfig
from sceneforge.labels import masks as masks_mod
from sceneforge.labels.coco import CATEGORIES, CocoWriter, export_zip
from sceneforge.labels.overlay import OVERLAY_MODES, draw_overlay
from sceneforge.render import camera as camera_math
from sceneforge.spec import CameraSpec

H = W = 96


# --------------------------------------------------------------------- fixtures
def _objects_meta():
    """Duck-typed ObjectSpec metadata against the documented §3.2 field names."""
    return [
        SimpleNamespace(instance_id=1, asset_id="mug", category="mug", is_target=True),
        SimpleNamespace(instance_id=2, asset_id="bowl", category="bowl", is_target=False),
        SimpleNamespace(instance_id=3, asset_id="can", category="can", is_target=False),
    ]


def _seg_ids():
    seg = np.zeros((H, W), dtype=np.int32)
    seg[10:40, 12:44] = 1  # 30x32 = 960 px
    seg[50:80, 50:86] = 2  # 30x36 = 1080 px
    seg[5:9, 5:9] = 3      # 16 px → filtered by MIN_AREA_PX=200 (§8.1)
    return seg


# ----------------------------------------------------------------- §8.1 masks
class TestExtractInstances:
    def test_rle_round_trip(self):
        seg = _seg_ids()
        labels = masks_mod.extract_instances(seg, _objects_meta())
        assert [l.instance_id for l in labels] == [1, 2]
        for label in labels:
            decoded = mask_util.decode(dict(label.rle)).astype(bool)
            assert decoded.shape == (H, W)
            assert np.array_equal(decoded, seg == label.instance_id)
            assert label.rle["size"] == [H, W]

    def test_bbox_and_area(self):
        labels = masks_mod.extract_instances(_seg_ids(), _objects_meta())
        by_id = {l.instance_id: l for l in labels}
        assert by_id[1].bbox_xywh == (12, 10, 32, 30)
        assert by_id[1].area_px == 960
        assert by_id[2].bbox_xywh == (50, 50, 36, 30)
        assert by_id[2].area_px == 1080
        assert by_id[1].is_target is True and by_id[2].is_target is False

    def test_area_filter_drops_small_instances(self):
        labels = masks_mod.extract_instances(_seg_ids(), _objects_meta())
        assert all(l.instance_id != 3 for l in labels)
        assert all(l.area_px >= masks_mod.MIN_AREA_PX for l in labels)

    def test_python_int_casts_and_json_safe(self):
        """§8.1 review fix: numpy int64 is not JSON-serializable."""
        for label in masks_mod.extract_instances(_seg_ids(), _objects_meta()):
            assert type(label.instance_id) is int
            assert type(label.area_px) is int
            assert all(type(v) is int for v in label.bbox_xywh)
            assert isinstance(label.rle["counts"], str)  # ascii, not bytes
            json.dumps(dataclasses.asdict(label))

    def test_accepts_spec_like_object(self):
        """§5.4 call site passes a SceneSpec; .objects must be used."""
        spec_like = SimpleNamespace(objects=_objects_meta())
        a = masks_mod.extract_instances(_seg_ids(), spec_like)
        b = masks_mod.extract_instances(_seg_ids(), _objects_meta())
        assert [l.instance_id for l in a] == [l.instance_id for l in b]


# ---------------------------------------------------------------- §3.1 depth16
class TestDepth16:
    def test_round_trip_mm(self, tmp_path):
        rng = np.random.default_rng(7)
        depth = rng.uniform(0.3, 3.0, size=(H, W)).astype(np.float32)
        depth[:8, :8] = 0.0  # no-hit region
        path = masks_mod.write_depth16(tmp_path / "depth16.png", depth)
        back = masks_mod.read_depth16(path)
        assert back.dtype == np.float32 and back.shape == depth.shape
        assert np.all(back[:8, :8] == 0.0)            # 0 = no-hit preserved
        assert float(np.abs(back - depth).max()) <= 0.0005 + 1e-6  # ±0.5 mm

    def test_clip_to_uint16_range(self, tmp_path):
        depth = np.full((4, 4), 70.0, dtype=np.float32)  # > 65.535 m
        back = masks_mod.read_depth16(masks_mod.write_depth16(tmp_path / "d.png", depth))
        assert np.all(back == pytest.approx(65.535))


# ---------------------------------------------------------------- §8.2 overlay
class TestOverlay:
    def test_modes(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        labels = masks_mod.extract_instances(_seg_ids(), _objects_meta())
        before = img.copy()
        for mode in OVERLAY_MODES:
            out = draw_overlay(img, labels, mode)
            assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert np.array_equal(img, before)  # input never mutated
        assert np.array_equal(draw_overlay(img, labels, "off"), img)
        assert not np.array_equal(draw_overlay(img, labels, "both"), img)
        with pytest.raises(ValueError):
            draw_overlay(img, labels, "wireframe")


# ------------------------------------------------------------------ §8.3 COCO
def _camera_view(layout_idx: int) -> dict:
    """Per-layout orbit view params (server-pinned yfov/look_at, §3.1)."""
    return {"azimuth_deg": 35.0 + 20.0 * layout_idx, "elevation_deg": 30.0,
            "distance_m": 1.1, "yfov_deg": 50.0, "look_at": [0.0, 0.0, 0.05]}


def _make_layout(layout_idx: int, rng: np.random.Generator, tmp_path: Path = None):
    """LayoutRecord shim (§4.6 field names) with tiny random masks + RGB-D
    sources (feature C): layout 0 carries explicit render K/pose + depth_m;
    layout 1 carries only the spec camera (recompute path) + an on-disk
    depth16_path (copy path) when ``tmp_path`` is given."""
    seg = np.zeros((H, W), dtype=np.int32)
    objs = []
    cats = ["mug", "bowl", "can", "box"]
    for iid in (1, 2, 3):
        side = int(rng.integers(16, 30))
        x0 = int(rng.integers(0, W - side))
        y0 = int(rng.integers(0, H - side))
        seg[y0:y0 + side, x0:x0 + side] = iid  # ≥ 256 px ≥ MIN_AREA_PX
        objs.append(SimpleNamespace(instance_id=iid, asset_id=cats[iid],
                                    category=cats[iid], is_target=(iid == 1)))
    instances = masks_mod.extract_instances(seg, objs)
    assert len(instances) >= 2  # later squares may overwrite earlier ids
    render = SimpleNamespace(width=W, height=H, instances=instances)
    spec = {"task": "pick the red mug", "seed": 42, "schema_version": "1.1",
            "camera": _camera_view(layout_idx)}
    layout = SimpleNamespace(layout_idx=layout_idx, spec=spec, render=render,
                             control_path=None, glb_path=None)

    depth = rng.uniform(0.5, 3.0, size=(H, W)).astype(np.float32)
    depth[:4, :4] = 0.0  # no-hit corner survives the round-trip
    if tmp_path is None or layout_idx % 2 == 0:  # ndarray source (encoded by writer)
        cam = CameraSpec.model_validate(_camera_view(layout_idx))
        render.depth_m = depth
        render.K = camera_math.intrinsics(W, H, cam.yfov_deg)
        render.camera_pose = camera_math.pose_from_orbit(cam)
    else:  # on-disk source (copied by writer); K/pose recomputed from spec camera
        layout.depth16_path = str(masks_mod.write_depth16(
            tmp_path / f"depth16_l{layout_idx}.png", depth))
    layout.gt_depth = depth  # test-side reference, ignored by the writer
    return layout


def _make_run(tmp_path: Path, rng: np.random.Generator):
    """Synthetic 2-layout × 2-style run (§8.3 acceptance fixture)."""
    layouts = [_make_layout(0, rng, tmp_path), _make_layout(1, rng, tmp_path)]
    images = []
    for layout in layouts:
        for style in ("rustic_kitchen", "clean_lab"):
            p = tmp_path / f"img_l{layout.layout_idx}_{style}.png"
            cv2.imwrite(str(p), rng.integers(0, 255, size=(H, W, 3)).astype(np.uint8))
            images.append(SimpleNamespace(path=str(p), layout_idx=layout.layout_idx,
                                          style_name=style, seed=42, gen_seconds=0.1))
    return layouts, images


class TestCocoExport:
    def test_acceptance(self, tmp_path):
        """§8.3: COCO() loads; annToMask round-trips; unique ann ids; json.dumps OK."""
        rng = np.random.default_rng(3)
        layouts, images = _make_run(tmp_path, rng)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        zip_path = CocoWriter().export(run_dir, layouts, images,
                                       fidelity_summary={"match_rate": 1.0})

        ann_file = run_dir / "coco" / "annotations.json"
        doc = json.loads(ann_file.read_text())
        json.dumps(doc)  # whole document JSON-serializable (§8.3)

        # every images entry has explicit width/height (§8.3)
        assert len(doc["images"]) == 4
        assert all(im["width"] == W and im["height"] == H for im in doc["images"])

        # annotation ids: ONE global running counter, unique across ALL images
        ann_ids = [a["id"] for a in doc["annotations"]]
        assert len(set(ann_ids)) == len(ann_ids)
        n_expected = sum(len(l.render.instances) for l in layouts) * 2  # 2 styles each
        assert len(ann_ids) == n_expected

        # attributes + provenance block
        a0 = doc["annotations"][0]["attributes"]
        assert set(a0) == {"is_target", "layout_idx", "style", "instance_id"}
        assert doc["sceneforge"]["task"] == "pick the red mug"
        assert len(doc["sceneforge"]["specs"]) == 2
        assert doc["sceneforge"]["fidelity_summary"] == {"match_rate": 1.0}

        # fixed 15-category list, 1-based library order (§8.3)
        cats = {c["name"]: c["id"] for c in doc["categories"]}
        assert all(cats[name] == i + 1 for i, name in enumerate(CATEGORIES))

        # pycocotools acceptance: loads + annToMask round-trips every annotation
        coco = COCO(str(ann_file))
        anns = coco.loadAnns(coco.getAnnIds())
        assert len(anns) == n_expected
        for ann in anns:
            assert int(coco.annToMask(ann).sum()) == ann["area"]

        # zip: annotations.json + images/ at root, returned for gr.DownloadButton
        assert zip_path == run_dir / "dataset.zip" and zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "annotations.json" in names
            assert sum(n.startswith("images/") and n.endswith(".png") for n in names) == 4

    def test_keep_filter_and_export_zip(self, tmp_path):
        rng = np.random.default_rng(4)
        layouts, images = _make_run(tmp_path, rng)
        run_dir = tmp_path / "run2"
        run_dir.mkdir()
        keep = [images[0].path, images[3].path]  # quarantine 2 of 4 (§4.6 kept list)
        zip_path = export_zip(run_dir, layouts, images, keep)
        doc = json.loads((run_dir / "coco" / "annotations.json").read_text())
        assert len(doc["images"]) == 2
        image_ids = {im["id"] for im in doc["images"]}
        assert all(a["image_id"] in image_ids for a in doc["annotations"])
        assert zip_path.exists()


# --------------------------------------------- feature C: RGB-D export contract
class TestRgbdExport:
    """cameras.json + depth/ round-trip over the same §8.3 synthetic fixture."""

    def _export(self, tmp_path):
        rng = np.random.default_rng(11)
        layouts, images = _make_run(tmp_path, rng)
        run_dir = tmp_path / "run_rgbd"
        run_dir.mkdir()
        zip_path = CocoWriter().export(run_dir, layouts, images)
        return layouts, run_dir, zip_path

    def test_zip_contains_depth_and_cameras(self, tmp_path):
        layouts, run_dir, zip_path = self._export(tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "cameras.json" in names
        # ONE depth file per layout (shared by both styles), §3.1 naming
        assert "depth/layout_0_depth16.png" in names
        assert "depth/layout_1_depth16.png" in names
        assert sum(n.startswith("depth/") and n.endswith(".png")
                   for n in names) == len(layouts)

    def test_every_image_has_camera_record(self, tmp_path):
        layouts, run_dir, _ = self._export(tmp_path)
        doc = json.loads((run_dir / "coco" / "annotations.json").read_text())
        cams = json.loads((run_dir / "coco" / "cameras.json").read_text())
        json.dumps(cams)  # JSON-able end to end

        assert "depth_encoding" in cams and cams["depth_encoding"]["no_hit_value"] == 0
        by_id = {rec["image_id"]: rec for rec in cams["images"]}
        assert len(by_id) == len(doc["images"]) == 4  # EVERY image has a record

        for im in doc["images"]:
            inline = im["sceneforge_camera"]          # inline copy on the entry
            rec = by_id[im["id"]]
            for record in (inline, rec):
                K = np.asarray(record["K"], dtype=np.float64)
                pose = np.asarray(record["pose"], dtype=np.float64)
                assert K.shape == (3, 3) and pose.shape == (4, 4)
                assert record["depth_file"].startswith("depth/")
                assert set(record["view"]) == {"azimuth_deg", "elevation_deg",
                                               "distance_m", "yfov_deg", "look_at"}
            assert inline["K"] == rec["K"] and inline["pose"] == rec["pose"]
            # K/pose match the NORMATIVE §3.1 math for the recorded view — true
            # for BOTH sources (explicit render K/pose and recompute-from-spec)
            cam = CameraSpec.model_validate(rec["view"])
            assert np.allclose(rec["K"], camera_math.intrinsics(W, H, cam.yfov_deg))
            assert np.allclose(rec["pose"], camera_math.pose_from_orbit(cam))

    def test_depth_round_trips_from_zip(self, tmp_path):
        layouts, run_dir, zip_path = self._export(tmp_path)
        out = tmp_path / "unzipped"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        for layout in layouts:  # covers ndarray-encoded AND copied sources
            back = masks_mod.read_depth16(
                out / "depth" / f"layout_{layout.layout_idx}_depth16.png")
            assert back.shape == (H, W)
            assert np.all(back[:4, :4] == 0.0)  # no-hit preserved as exactly 0
            assert float(np.abs(back - layout.gt_depth).max()) <= 0.0005 + 1e-6

    def test_layouts_without_camera_or_depth_export_as_before(self, tmp_path):
        """Legacy duck-typed layouts (no camera/depth sources) stay untouched."""
        rng = np.random.default_rng(12)
        layouts, images = _make_run(tmp_path, rng)
        for layout in layouts:  # strip every RGB-D source
            layout.spec.pop("camera", None)
            for attr in ("depth_m", "K", "camera_pose"):
                if hasattr(layout.render, attr):
                    delattr(layout.render, attr)
            if hasattr(layout, "depth16_path"):
                del layout.depth16_path
        run_dir = tmp_path / "run_legacy"
        run_dir.mkdir()
        zip_path = CocoWriter().export(run_dir, layouts, images)
        doc = json.loads((run_dir / "coco" / "annotations.json").read_text())
        assert all("sceneforge_camera" not in im for im in doc["images"])
        assert not (run_dir / "coco" / "cameras.json").exists()
        with zipfile.ZipFile(zip_path) as zf:
            assert not any(n.startswith("depth/") for n in zf.namelist())

    def test_view_prefix_naming(self, tmp_path):
        """Viewsweep layouts (``name='view_<k>'``) drive image+depth naming."""
        rng = np.random.default_rng(13)
        layouts, images = _make_run(tmp_path, rng)
        for layout in layouts:
            layout.name = f"view_{layout.layout_idx}"
        run_dir = tmp_path / "run_views"
        run_dir.mkdir()
        zip_path = CocoWriter().export(run_dir, layouts, images)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "depth/view_0_depth16.png" in names and "depth/view_1_depth16.png" in names
        doc = json.loads((run_dir / "coco" / "annotations.json").read_text())
        assert all(im["file_name"].startswith("view_") for im in doc["images"])


# ------------------------------------------------- §10.3 ollama_unload (mocked)
class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _fake_requests(ps_sequence, post_log, raise_on_get=False):
    """requests stand-in: GET /api/ps walks ps_sequence (last repeats); POST
    /api/generate answers 400 for embedding models (§10.3 verified behavior)."""
    state = {"i": 0}

    def get(url, timeout=None):
        assert url.endswith("/api/ps")
        if raise_on_get:
            raise real_requests.exceptions.ConnectionError("server stopped")
        models = ps_sequence[min(state["i"], len(ps_sequence) - 1)]
        state["i"] += 1
        return _Resp(200, {"models": [{"name": n} for n in models]})

    def post(url, json=None, timeout=None):
        endpoint = url.rsplit("/api/", 1)[1]
        post_log.append((endpoint, json))
        if endpoint == "generate" and "embed" in json["model"]:
            return _Resp(400, {"error": f"{json['model']} does not support generate"})
        return _Resp(200, {})

    return SimpleNamespace(get=get, post=post,
                           RequestException=real_requests.RequestException)


class TestOllamaUnload:
    HOST = "http://mock:11434"

    def test_400_embed_fallback_then_ps_empties(self, monkeypatch):
        post_log = []
        fake = _fake_requests([["gemma4:e4b", "embeddinggemma"], []], post_log)
        monkeypatch.setattr(gpu, "requests", fake)
        assert gpu.ollama_unload(self.HOST, timeout_s=2, poll_s=0.01) is True
        assert ("generate", {"model": "gemma4:e4b", "keep_alive": 0}) in post_log
        assert ("generate", {"model": "embeddinggemma", "keep_alive": 0}) in post_log
        # HTTP 400 → /api/embed fallback with empty input + keep_alive 0 (§10.3)
        assert ("embed", {"model": "embeddinggemma", "input": "", "keep_alive": 0}) in post_log
        assert len(post_log) == 3  # no embed fallback for the generate-capable model

    def test_only_filter_leaves_gemma_resident(self, monkeypatch):
        """Coresident diffusion barrier: unload embed models only (§10.3)."""
        post_log = []
        fake = _fake_requests([["gemma4:e4b", "embeddinggemma"], ["gemma4:e4b"]], post_log)
        monkeypatch.setattr(gpu, "requests", fake)
        ok = gpu.ollama_unload(self.HOST, timeout_s=2, poll_s=0.01,
                               only=lambda n: "embed" in n.lower())
        assert ok is True
        assert all(body["model"] == "embeddinggemma" for _, body in post_log)

    def test_dead_server_counts_as_empty(self, monkeypatch):
        """Demo must complete with Ollama stopped (§1) — no exception, True."""
        post_log = []
        fake = _fake_requests([], post_log, raise_on_get=True)
        monkeypatch.setattr(gpu, "requests", fake)
        assert gpu.ollama_unload(self.HOST, timeout_s=1, poll_s=0.01) is True
        assert post_log == []

    def test_timeout_returns_false(self, monkeypatch):
        fake = _fake_requests([["stuck_model"]], [])
        monkeypatch.setattr(gpu, "requests", fake)
        assert gpu.ollama_unload(self.HOST, timeout_s=0.1, poll_s=0.02) is False


# ----------------------------------------------------------- §10.3 phase() mgr
def _cfg(mode="sequential"):
    return AppConfig(vram=VramConfig(mode=mode),
                     ollama=OllamaConfig(host="http://mock:11434"))


class TestPhase:
    @pytest.fixture(autouse=True)
    def _mock_gpu(self, monkeypatch):
        self.unload_calls = []
        monkeypatch.setattr(gpu, "ollama_unload",
                            lambda host=None, **kw: self.unload_calls.append((host, kw)) or True)
        monkeypatch.setattr(gpu, "snapshot", lambda host=None: {"free_gb": 0.0})

    def test_diffusion_sequential_unloads_all(self):
        log = []
        with gpu.phase("diffusion", cfg=_cfg(), vram_log=log) as record:
            assert record["enter"] == {"free_gb": 0.0}
        (host, kw), = self.unload_calls
        assert host == "http://mock:11434" and kw.get("only") is None
        assert log == [record] and "exit" in record  # enter/exit snapshots logged

    def test_diffusion_coresident_unloads_only_embed_models(self):
        with gpu.phase("diffusion", cfg=_cfg("coresident")):
            pass
        (_, kw), = self.unload_calls
        only = kw["only"]
        assert only("embeddinggemma") is True and only("gemma4:e4b") is False

    def test_eval_unloads_in_both_modes(self):
        for mode in ("sequential", "coresident"):
            with gpu.phase("eval", cfg=_cfg(mode)):
                pass
        assert len(self.unload_calls) == 2
        assert all(kw.get("only") is None for _, kw in self.unload_calls)

    def test_spec_no_unload(self):
        with gpu.phase("spec", cfg=_cfg()):  # gc + empty_cache only (§10.3)
            pass
        assert self.unload_calls == []

    def test_spec_quality_pipe_swap(self):
        """§10.4: pipe → cpu on entry; unload-wait then pipe → cuda on exit."""
        moves = []
        pipe = SimpleNamespace(to=lambda device: moves.append(device))
        with gpu.phase("spec_quality", cfg=_cfg(), pipe=pipe):
            assert moves == ["cpu"]
            assert self.unload_calls == []
        assert moves == ["cpu", "cuda"]
        assert len(self.unload_calls) == 1  # waits for qwen (keep_alive=0) to clear

    def test_unknown_phase_raises(self):
        with pytest.raises(ValueError):
            with gpu.phase("render", cfg=_cfg()):
                pass


# -------------------------------------------------------------- §10.2 helpers
def test_vram_ladder_constants():
    assert gpu.VRAM_LEVELS == ("V0", "V1", "V2", "V3", "V4")
    assert gpu.next_vram_level("V1") == "V2"
    assert gpu.next_vram_level("V4") == "V4"  # terminal
    assert gpu.guard_gb(_cfg()) == 3.0       # ONE constant, from config (§10.2)
