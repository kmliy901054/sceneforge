"""sceneforge/labels/coco.py — CocoWriter + export_zip (ARCHITECTURE.md §8.3)
plus the RGB-D export extension (v2 feature C).

Review fixes baked in:
- Annotation ids: ONE global running counter across ALL images (per-layout reuse
  breaks COCO indexing).
- Every ``images`` entry carries explicit ``width``/``height``.
- All ids/bbox/area are python ints; ``json.dumps`` of the whole document is
  asserted in tests/test_masks.py.
- Extra top-level ``"sceneforge"`` provenance block (loaders ignore unknown keys).

RGB-D extension (rendered ground-truth depth is SceneForge's differentiator —
RGB-estimated depth is not real 3D grounding):
- ``coco/depth/<layout|view>_depth16.png`` — ONE depth map per layout/view
  (shared by every style variant of that geometry), encoded per the §3.1
  on-disk contract: uint16 millimeters, 0 = no-hit.
- ``coco/cameras.json`` — per-image camera record: pinhole ``K`` (3×3),
  world-from-camera ``pose`` (4×4, §3.1 orbit convention), orbit ``view``
  params, and the relative path of the image's depth file, plus the depth
  encoding contract block.
- every ``images`` entry additionally carries the same camera record inline
  under ``"sceneforge_camera"`` (COCO loaders ignore unknown keys).

Camera/depth sources are duck-typed per layout and all OPTIONAL (legacy callers
keep working unchanged):
- depth: ``layout.depth16_path`` (existing §3.1 png, copied) else
  ``layout.render.depth_m`` (ndarray, encoded here);
- K/pose: ``layout.render.K`` / ``layout.render.camera_pose`` (ndarrays) else
  recomputed from the spec's orbit camera via ``render.camera`` (§3.1);
- file prefix: ``layout.name`` (e.g. ``"view_3"``) else ``"layout_<idx>"``.

``layouts``/``images`` are duck-typed against the §4.5/§4.6 dataclasses
(``LayoutRecord(layout_idx, spec, render, control_path, glb_path)`` and
``GeneratedImage(path, layout_idx, style_name, seed, gen_seconds)``) — they are
owned by the orchestrator module and accessed here by attribute name only.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

#: §3.1 depth16.png on-disk contract, embedded verbatim in cameras.json so the
#: zip is self-describing.
DEPTH_ENCODING: dict[str, Any] = {
    "format": "uint16 PNG, single channel",
    "units": "millimeters",
    "depth_m": "png_value / 1000.0",
    "no_hit_value": 0,
    "max_depth_m": 65.535,
    "contract": "ARCHITECTURE.md §3.1: depth_mm = round(depth_m*1000).clip(0, 65535); 0 = no-hit",
}

#: §8.3/§5.1 — fixed 15-asset list in builder-table order; category_id = index + 1
#: (1-based library order, stable). The kitchen set #1–10 is the protected core.
CATEGORIES: tuple[str, ...] = (
    "mug", "bowl", "plate", "cup", "bottle", "can", "box", "book", "pan", "ball",
    "pot", "screwdriver", "hammer", "cutting_board", "laptop",
)


def _spec_dump(spec: Any) -> Any:
    """SceneSpec (pydantic) or plain dict → JSON-able dict."""
    if hasattr(spec, "model_dump"):
        return spec.model_dump(mode="json")
    return spec


def _spec_task(spec: Any) -> Optional[str]:
    task = getattr(spec, "task", None)
    if task is None and isinstance(spec, dict):
        task = spec.get("task")
    return task


def _ascii_rle(rle: dict) -> dict:
    counts = rle["counts"]
    return {
        "size": [int(s) for s in rle["size"]],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else str(counts),
    }


# ------------------------------------------------ RGB-D extension (feature C)
def _layout_prefix(layout: Any) -> str:
    """Depth/file-name prefix: ``layout.name`` (viewsweep uses ``view_<k>``)
    else the classic ``layout_<idx>``."""
    name = getattr(layout, "name", None)
    return str(name) if name else f"layout_{int(layout.layout_idx)}"


def _spec_camera_dict(spec: Any) -> Optional[dict]:
    """The spec's camera as a plain dict, or None (specs are SceneSpec or dict)."""
    cam = getattr(spec, "camera", None)
    if cam is None and isinstance(spec, dict):
        cam = spec.get("camera")
    if cam is None:
        return None
    if hasattr(cam, "model_dump"):
        return cam.model_dump(mode="json")
    return dict(cam) if isinstance(cam, dict) else None


def _matrix(value: Any, shape: tuple[int, int]) -> Optional[list[list[float]]]:
    """ndarray/nested-list → JSON-able list-of-lists iff it has ``shape``."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        return None
    return [[float(v) for v in row] for row in arr]


def _layout_camera(layout: Any) -> Optional[dict]:
    """Per-layout camera record ``{"K", "pose", "view"}`` or None (§3.1).

    Prefers the render's measured ``K``/``camera_pose``; falls back to
    recomputing both from the spec's orbit camera (deterministic — the §3.1
    equations in ``render.camera`` are normative). ``view`` carries the orbit
    params so consumers can re-derive or perturb the viewpoint.
    """
    render = getattr(layout, "render", None)
    K = _matrix(getattr(render, "K", None), (3, 3))
    pose = _matrix(getattr(render, "camera_pose", None), (4, 4))
    view = _spec_camera_dict(getattr(layout, "spec", None))

    if (K is None or pose is None) and view is not None:
        try:  # recompute from the orbit view params (no GL involved)
            from sceneforge.render import camera as camera_math
            from sceneforge.spec import CameraSpec

            cam = CameraSpec.model_validate(view)
            if pose is None:
                pose = _matrix(camera_math.pose_from_orbit(cam), (4, 4))
            if K is None and render is not None:
                K = _matrix(
                    camera_math.intrinsics(int(render.width), int(render.height),
                                           cam.yfov_deg),
                    (3, 3),
                )
        except Exception as exc:  # malformed spec camera — camera info is optional
            logger.warning("cameras.json: cannot derive camera for %s: %s",
                           _layout_prefix(layout), exc)

    if K is None or pose is None:
        return None
    record: dict[str, Any] = {"K": K, "pose": pose}
    if view is not None:
        record["view"] = view
    return record


def _write_layout_depth(layout: Any, depth_dir: Path) -> Optional[str]:
    """Write/copy this layout's §3.1 depth16 png into ``coco/depth/``.

    Returns the zip-relative path (``depth/<prefix>_depth16.png``) or None when
    the layout carries no depth source. Sources, in order: an existing
    ``layout.depth16_path`` file (copied byte-for-byte), else
    ``layout.render.depth_m`` encoded via ``masks.write_depth16``.
    """
    prefix = _layout_prefix(layout)
    rel = f"depth/{prefix}_depth16.png"
    src = getattr(layout, "depth16_path", None)
    if src and Path(src).is_file():
        depth_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, depth_dir / f"{prefix}_depth16.png")
        return rel
    depth_m = getattr(getattr(layout, "render", None), "depth_m", None)
    if depth_m is not None:
        from sceneforge.labels import masks as masks_mod

        depth_dir.mkdir(parents=True, exist_ok=True)
        masks_mod.write_depth16(depth_dir / f"{prefix}_depth16.png", np.asarray(depth_m))
        return rel
    return None


class CocoWriter:
    """COCO dataset writer with a global annotation-id counter (§8.3)."""

    def __init__(self, categories: Sequence[str] = CATEGORIES) -> None:
        self._cat_ids: dict[str, int] = {c: i + 1 for i, c in enumerate(categories)}

    def category_id(self, name: str) -> int:
        """1-based id in fixed library order; unknown names (optional GLB drop-ins,
        §12-A — additive only) are appended after the fixed list in first-seen order."""
        if name not in self._cat_ids:
            self._cat_ids[name] = len(self._cat_ids) + 1
            logger.warning("CocoWriter: registering non-library category %r as id %d",
                           name, self._cat_ids[name])
        return self._cat_ids[name]

    def export(
        self,
        run_dir: Union[str, Path],
        layouts: Sequence[Any],
        images: Sequence[Any],
        keep: Optional[Iterable[Union[str, Path]]] = None,
        *,
        fidelity_summary: Optional[dict] = None,
    ) -> Path:
        """Build ``<run_dir>/coco/`` and zip it to ``<run_dir>/dataset.zip`` (§8.3/§4.7).

        Copies kept images → ``coco/images/``, writes ``coco/annotations.json``
        (one global annotation-id counter; explicit width/height per image entry;
        per-annotation ``attributes`` {is_target, layout_idx, style, instance_id};
        top-level ``"sceneforge"`` {task, specs, fidelity_summary}), then
        ``shutil.make_archive`` → ``dataset.zip``. Returns the zip path for
        ``gr.DownloadButton``.

        RGB-D extension (feature C): layouts that carry a depth source get ONE
        ``depth/<layout|view>_depth16.png`` each (§3.1 encoding); layouts that
        carry (or allow recomputing) camera info get a per-image
        ``"sceneforge_camera"`` record inline AND in ``coco/cameras.json``
        (written whenever any image has camera or depth info). Both are
        optional per layout — legacy duck-typed layouts export exactly as before.

        Args:
            keep: optional iterable of image paths to include (the fidelity
                ``kept`` list, §4.6); None keeps everything.
        """
        run_dir = Path(run_dir)
        coco_dir = run_dir / "coco"
        images_dir = coco_dir / "images"
        if coco_dir.exists():
            shutil.rmtree(coco_dir)  # re-export must not accumulate stale images
        images_dir.mkdir(parents=True)
        depth_dir = coco_dir / "depth"

        layout_by_idx: dict[int, Any] = {int(l.layout_idx): l for l in layouts}
        keep_set = None if keep is None else {str(Path(p)) for p in keep}

        # One camera record + one depth png per layout (shared by all styles).
        # Depth files are written lazily so fully-quarantined layouts leave
        # no orphan depth/ entries in the zip.
        camera_by_idx: dict[int, Optional[dict]] = {
            idx: _layout_camera(l) for idx, l in layout_by_idx.items()
        }
        depth_rel_by_idx: dict[int, Optional[str]] = {}
        camera_images: list[dict] = []

        doc: dict[str, Any] = {
            "info": {
                "description": "SceneForge synthetic dataset",
                "version": "1.1",
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        next_image_id = 1
        next_ann_id = 1  # GLOBAL across all images (§8.3)
        used_names: set[str] = set()
        for gen in images:
            src = Path(gen.path)
            if keep_set is not None and str(src) not in keep_set:
                continue
            layout_idx = int(gen.layout_idx)
            if layout_idx not in layout_by_idx:
                raise ValueError(f"image {src} references unknown layout_idx {layout_idx}")
            layout = layout_by_idx[layout_idx]
            render = layout.render
            style = str(gen.style_name)

            prefix = _layout_prefix(layout)
            file_name = f"{prefix}_{style}.png"
            n = 2
            while file_name in used_names:  # e.g. re-forged same layout+style
                file_name = f"{prefix}_{style}_{n}.png"
                n += 1
            used_names.add(file_name)
            shutil.copy2(src, images_dir / file_name)

            image_id = next_image_id
            next_image_id += 1
            entry: dict[str, Any] = {
                "id": int(image_id),
                "file_name": file_name,
                "width": int(render.width),    # explicit on EVERY entry (§8.3)
                "height": int(render.height),
            }
            camera = camera_by_idx.get(layout_idx)
            if layout_idx not in depth_rel_by_idx:
                depth_rel_by_idx[layout_idx] = _write_layout_depth(layout, depth_dir)
            depth_rel = depth_rel_by_idx[layout_idx]
            if camera is not None or depth_rel is not None:
                cam_record: dict[str, Any] = dict(camera or {})
                if depth_rel is not None:
                    cam_record["depth_file"] = depth_rel
                entry["sceneforge_camera"] = cam_record  # loaders ignore unknown keys
                camera_images.append({
                    "image_id": int(image_id),
                    "file_name": file_name,
                    **cam_record,
                })
            doc["images"].append(entry)
            for inst in render.instances:  # layout labels shared by all styles (§4.5)
                x, y, w, h = inst.bbox_xywh
                doc["annotations"].append({
                    "id": int(next_ann_id),
                    "image_id": int(image_id),
                    "category_id": int(self.category_id(str(inst.category))),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(inst.area_px),
                    "segmentation": _ascii_rle(inst.rle),
                    "iscrowd": 0,
                    "attributes": {
                        "is_target": bool(inst.is_target),
                        "layout_idx": int(layout_idx),
                        "style": style,
                        "instance_id": int(inst.instance_id),
                    },
                })
                next_ann_id += 1

        doc["categories"] = [
            {"id": int(cid), "name": name, "supercategory": "object"}
            for name, cid in sorted(self._cat_ids.items(), key=lambda kv: kv[1])
        ]
        ordered = sorted(layout_by_idx.items())
        doc["sceneforge"] = {  # loaders ignore unknown top-level keys (§8.3)
            "task": next((_spec_task(l.spec) for _, l in ordered if _spec_task(l.spec)), None),
            "specs": [_spec_dump(l.spec) for _, l in ordered],
            "fidelity_summary": fidelity_summary or {},
        }

        if camera_images:  # feature C: self-describing RGB-D sidecar
            cameras_doc = {
                "depth_encoding": DEPTH_ENCODING,
                "pose_convention": (
                    "world-from-camera, ARCHITECTURE.md §3.1: world +Z up, meters; "
                    "camera looks down its -Z; pose columns [x_axis, y_axis, z_axis, eye]; "
                    "u = cx + fx*x_c/(-z_c), v = cy - fy*y_c/(-z_c)"
                ),
                "images": camera_images,
            }
            (coco_dir / "cameras.json").write_text(json.dumps(cameras_doc, indent=2),
                                                   encoding="utf-8")

        (coco_dir / "annotations.json").write_text(json.dumps(doc), encoding="utf-8")
        zip_path = Path(shutil.make_archive(str(run_dir / "dataset"), "zip",
                                            root_dir=str(coco_dir)))
        logger.info("COCO export: %d images, %d annotations, %d camera records → %s",
                    len(doc["images"]), len(doc["annotations"]), len(camera_images),
                    zip_path)
        return zip_path


def export_zip(
    run_dir: Union[str, Path],
    layouts: Sequence[Any],
    images: Sequence[Any],
    keep: Optional[Iterable[Union[str, Path]]] = None,
    *,
    fidelity_summary: Optional[dict] = None,
) -> Path:
    """Convenience wrapper: ``CocoWriter().export(...)`` (§2 module tree)."""
    return CocoWriter().export(run_dir, layouts, images, keep,
                               fidelity_summary=fidelity_summary)
