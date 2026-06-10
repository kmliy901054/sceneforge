"""sceneforge/labels/coco.py — CocoWriter + export_zip (ARCHITECTURE.md §8.3).

Review fixes baked in:
- Annotation ids: ONE global running counter across ALL images (per-layout reuse
  breaks COCO indexing).
- Every ``images`` entry carries explicit ``width``/``height``.
- All ids/bbox/area are python ints; ``json.dumps`` of the whole document is
  asserted in tests/test_masks.py.
- Extra top-level ``"sceneforge"`` provenance block (loaders ignore unknown keys).

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

logger = logging.getLogger(__name__)

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

        layout_by_idx: dict[int, Any] = {int(l.layout_idx): l for l in layouts}
        keep_set = None if keep is None else {str(Path(p)) for p in keep}

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

            file_name = f"layout_{layout_idx}_{style}.png"
            n = 2
            while file_name in used_names:  # e.g. re-forged same layout+style
                file_name = f"layout_{layout_idx}_{style}_{n}.png"
                n += 1
            used_names.add(file_name)
            shutil.copy2(src, images_dir / file_name)

            image_id = next_image_id
            next_image_id += 1
            doc["images"].append({
                "id": int(image_id),
                "file_name": file_name,
                "width": int(render.width),    # explicit on EVERY entry (§8.3)
                "height": int(render.height),
            })
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

        (coco_dir / "annotations.json").write_text(json.dumps(doc), encoding="utf-8")
        zip_path = Path(shutil.make_archive(str(run_dir / "dataset"), "zip",
                                            root_dir=str(coco_dir)))
        logger.info("COCO export: %d images, %d annotations → %s",
                    len(doc["images"]), len(doc["annotations"]), zip_path)
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
