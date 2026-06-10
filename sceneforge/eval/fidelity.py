"""OWLv2 fidelity eval (ARCHITECTURE.md §9): recall + hallucination terms.

google/owlv2-base-patch16-ensemble, fp16, loaded only inside gpu.phase("eval")
at runtime (M1's harness drives it directly). Needs transformers<5 (P0 §0).

SQUARE-IMAGE NOTE (§9): OWLv2's processor pads inputs to square — our images
are square 768², so padding is identity and
``post_process_object_detection(threshold=0.15, target_sizes=[(H, W)])``
returns usable pixel boxes directly. This SILENTLY BREAKS for non-square
generations; do not feed non-square images.

Metrics:
  - Recall term: per GT instance, iou_i = max IoU vs same-category detections
    (0 if none). Per-image ``fidelity = mean_i(iou_i)``. Batch: ``match_rate``
    = fraction of GT with iou >= 0.5; ``mean_matched_iou`` = mean iou over
    matched GT only.
  - Hallucination term: per image, detections of in-layout categories with
    score >= 0.3 whose IoU with EVERY GT box of that category is < 0.3.
    ``fidelity_adj = fidelity - 0.5 * hallucination_count / max(1, n_gt)``;
    quarantine on ``fidelity_adj < keep_threshold`` (default 0.45).
  - Gate population: all gate/threshold statistics are computed over GT
    instances with ``area_px >= 1000`` (OWLv2 recall collapses on tiny/thin
    instances — a detector failure, not a label failure); smaller instances
    are scored but excluded from gate stats, flagged ``gate_eligible=False``.

Honest-metric caveat: OWLv2 box jitter caps matched IoU at ~0.75–0.85 even on
perfect labels — the eval is a LOWER BOUND on label validity.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "google/owlv2-base-patch16-ensemble"
DET_THRESHOLD = 0.15        # recall-side detection threshold (§9)
HALLUC_SCORE = 0.3          # hallucination-side score floor (§9)
HALLUC_IOU = 0.3            # below this IoU vs every same-cat GT => hallucination
MATCH_IOU = 0.5             # GT counts as matched at IoU >= 0.5 (gate)
GATE_AREA_PX = 1000         # gate-eligible GT population (§9 review fix)
KEEP_THRESHOLD = 0.45       # cfg.eval.keep_threshold default


@dataclass
class GTInstance:
    """Minimal ground-truth box for scoring (decoupled from render.types)."""
    category: str
    bbox_xywh: tuple[int, int, int, int]   # COCO pixel order
    area_px: int
    is_target: bool = False


@dataclass
class ImageScore:
    path: str
    fidelity: float                 # mean iou_i over gate-eligible GT
    fidelity_adj: float             # fidelity - 0.5 * halluc / max(1, n_gt)
    hallucination_count: int
    per_instance: list[dict] = field(default_factory=list)


@dataclass
class FidelityReport:
    per_image: list[ImageScore]
    match_rate: float               # gate-eligible GT with iou >= 0.5
    mean_matched_iou: float         # mean iou over matched gate-eligible GT
    hallucination_rate: float       # total hallucinations / total GT (informational)
    kept: list[str]
    quarantined: list[str]
    n_gate_eligible: int = 0
    n_gt_total: int = 0


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two COCO [x, y, w, h] boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


class Owlv2Scorer:
    """Zero-shot open-vocab spot-check of the renderer's labels (§9)."""

    def __init__(
        self,
        device: str = "cuda",
        keep_threshold: float = KEEP_THRESHOLD,
        det_threshold: float = DET_THRESHOLD,
        gate_area_px: int = GATE_AREA_PX,
    ) -> None:
        self.device = device
        self.keep_threshold = keep_threshold
        self.det_threshold = det_threshold
        self.gate_area_px = gate_area_px
        self.processor: Any = None
        self.model: Any = None

    # ----------------------------------------------------------- lifecycle
    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        logger.info("loading %s (fp16)", MODEL_ID)
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to(self.device).eval()

    def unload(self) -> None:
        self.processor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ detection
    @torch.no_grad()
    def detect(self, image: Image.Image, categories: list[str]) -> list[dict]:
        """Run OWLv2 with one '"a photo of a {category}"' query per category.

        Returns [{category, score, bbox_xywh}] at score >= det_threshold.
        Requires a SQUARE image (see module docstring).
        """
        self.load()
        w, h = image.size
        if w != h:
            raise ValueError(
                f"Owlv2Scorer requires square images (got {w}x{h}); the padded-"
                "square shortcut in post_process breaks otherwise (§9)")
        queries = [f"a photo of a {c}" for c in categories]
        inputs = self.processor(text=queries, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.det_threshold,
            target_sizes=torch.tensor([(h, w)]))[0]
        dets = []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box]
            dets.append({
                "category": categories[int(label)],
                "score": float(score),
                "bbox_xywh": (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)),
            })
        return dets

    # -------------------------------------------------------------- scoring
    def score_image(self, image: Image.Image, gt: Sequence[GTInstance],
                    path: str = "") -> ImageScore:
        """Score one image against its layout's GT instances (§9)."""
        categories = sorted({g.category for g in gt})
        dets = self.detect(image, categories)

        per_instance: list[dict] = []
        eligible_ious: list[float] = []
        for g in gt:
            same_cat = [d for d in dets if d["category"] == g.category]
            iou = max((iou_xywh(g.bbox_xywh, d["bbox_xywh"]) for d in same_cat),
                      default=0.0)
            eligible = g.area_px >= self.gate_area_px
            per_instance.append({
                "category": g.category,
                "bbox_xywh": [int(v) for v in g.bbox_xywh],
                "area_px": int(g.area_px),
                "iou": iou,
                "matched": iou >= MATCH_IOU,
                "gate_eligible": eligible,
                "is_target": g.is_target,
            })
            if eligible:
                eligible_ious.append(iou)

        # fidelity over gate-eligible GT; fall back to all GT if none eligible
        pool = eligible_ious or [pi["iou"] for pi in per_instance] or [0.0]
        fidelity = float(np.mean(pool))

        # hallucination: confident same-category detections matching NO GT box
        halluc = 0
        for d in dets:
            if d["score"] < HALLUC_SCORE:
                continue
            gt_boxes = [g.bbox_xywh for g in gt if g.category == d["category"]]
            if gt_boxes and all(
                    iou_xywh(d["bbox_xywh"], b) < HALLUC_IOU for b in gt_boxes):
                halluc += 1

        fidelity_adj = fidelity - 0.5 * halluc / max(1, len(gt))
        return ImageScore(path=path, fidelity=fidelity, fidelity_adj=fidelity_adj,
                          hallucination_count=halluc, per_instance=per_instance)

    def score_batch(
        self,
        images: Sequence[Image.Image | str],
        layouts: Sequence[Sequence[GTInstance]],
    ) -> FidelityReport:
        """Score aligned (image, GT-list) pairs; batch stats per §9.

        ``layouts[i]`` holds the GT instances for ``images[i]`` (label transfer
        is identity — every style variant of a layout reuses its instances).
        """
        if len(images) != len(layouts):
            raise ValueError("images and layouts must be aligned 1:1")
        self.load()
        per_image: list[ImageScore] = []
        for img, gt in zip(images, layouts):
            path = img if isinstance(img, str) else getattr(img, "filename", "")
            pil = Image.open(img).convert("RGB") if isinstance(img, str) else img
            per_image.append(self.score_image(pil, list(gt), path=str(path)))

        elig = [pi for s in per_image for pi in s.per_instance if pi["gate_eligible"]]
        matched = [pi["iou"] for pi in elig if pi["matched"]]
        n_gt_total = sum(len(s.per_instance) for s in per_image)
        total_halluc = sum(s.hallucination_count for s in per_image)
        kept = [s.path for s in per_image if s.fidelity_adj >= self.keep_threshold]
        quar = [s.path for s in per_image if s.fidelity_adj < self.keep_threshold]
        return FidelityReport(
            per_image=per_image,
            match_rate=float(len(matched) / len(elig)) if elig else 0.0,
            mean_matched_iou=float(np.mean(matched)) if matched else 0.0,
            hallucination_rate=float(total_halluc / max(1, n_gt_total)),
            kept=kept,
            quarantined=quar,
            n_gate_eligible=len(elig),
            n_gt_total=n_gt_total,
        )
