"""tests/test_render.py — EGL render backend tests (ARCHITECTURE.md §3.1/§5.4/§5.5).

Covers: box+cylinder scene (depth nonzero under seg; seg ids round-trip exactly
{1, 2}); byte-identical seg across two renders; persistent OffscreenRenderer
reuse/recreation; orbit camera math vs the §3.1 formulas; normative projection
sanity at 768²; numpy contingency stub.

InstanceLabels are built INLINE here (mask → cv2.boundingRect → pycocotools
RLE) — deliberately NOT via sceneforge.labels, which is owned by a concurrent
workstream (§8.1 owns the production path).
"""
import sceneforge.compat  # noqa: F401  — must precede any pyrender import

import json
import math

import cv2
import numpy as np
import pytest
import trimesh
from pycocotools import mask as cocomask

from sceneforge.render import ComposedScene, InstanceLabel, get_renderer
from sceneforge.render import camera as camera_math
from sceneforge.render import numpy_backend
from sceneforge.spec import CameraSpec

BOX_ID, CYL_ID = 1, 2


# ---------------------------------------------------------------- fixtures --
def make_box_cylinder_scene() -> ComposedScene:
    """Box (id 1) + cylinder (id 2) resting on the z=0 tabletop (§3.1), with
    table slab and far floor as static (depth-only) geometry."""
    table = trimesh.creation.box(extents=(1.2, 0.8, 0.04))
    table.apply_translation([0.0, 0.0, -0.02])           # TOP surface at z = 0
    floor = trimesh.creation.box(extents=(2.5, 2.5, 0.02))
    floor.apply_translation([0.0, 0.0, -0.75 - 0.01])    # top at z = -0.75

    box = trimesh.creation.box(extents=(0.18, 0.12, 0.08))
    box.apply_translation([0.0, 0.0, 0.04])              # min z = 0
    t_box = np.eye(4)
    t_box[:3, 3] = [-0.15, 0.0, 0.0]

    cyl = trimesh.creation.cylinder(radius=0.045, height=0.12)
    cyl.apply_translation([0.0, 0.0, 0.06])              # min z = 0
    t_cyl = np.eye(4)
    t_cyl[:3, 3] = [0.15, 0.05, 0.0]

    return ComposedScene(
        instances=[(BOX_ID, box, t_box), (CYL_ID, cyl, t_cyl)],
        static=[("_floor", floor, np.eye(4)), ("_table", table, np.eye(4))],
        glb_path="",
    )


@pytest.fixture(scope="module")
def backend():
    return get_renderer()


@pytest.fixture(scope="module")
def composed() -> ComposedScene:
    return make_box_cylinder_scene()


@pytest.fixture(scope="module")
def result_256(backend, composed):
    return backend.render_scene(composed, CameraSpec(), 256, 256)


# ----------------------------------------------------------- 256 px checks --
def test_get_renderer_is_cached(backend):
    assert get_renderer() is backend


def test_seg_ids_round_trip_exactly(result_256):
    """Red-channel SEG ids survive the GL round-trip exactly: {1, 2} (§12-B)."""
    ids = set(np.unique(result_256.seg_ids)) - {0}
    assert ids == {BOX_ID, CYL_ID}
    for iid in (BOX_ID, CYL_ID):
        assert int((result_256.seg_ids == iid).sum()) >= 50  # clearly visible


def test_depth_nonzero_where_seg_nonzero(result_256):
    hit = result_256.seg_ids > 0
    assert hit.any()
    depths = result_256.depth_m[hit]
    assert np.all(depths > 0.0)
    # plausible metric range: objects sit ~1.1 m from the orbit camera
    assert depths.min() > 0.3 and depths.max() < 2.5


def test_result_contract_shapes(result_256):
    r = result_256
    assert (r.width, r.height) == (256, 256)
    assert r.color.shape == (256, 256, 3) and r.color.dtype == np.uint8
    assert r.depth_m.shape == (256, 256) and r.depth_m.dtype == np.float32
    assert r.seg_ids.shape == (256, 256) and r.seg_ids.dtype == np.int32
    assert r.camera_pose.shape == (4, 4) and r.camera_pose.dtype == np.float32
    assert r.K.shape == (3, 3) and r.K.dtype == np.float32
    assert r.instances == []  # no spec passed → caller extracts labels


def test_seg_byte_identical_across_two_renders(backend, composed):
    a = backend.render_scene(composed, CameraSpec(), 256, 256)
    b = backend.render_scene(composed, CameraSpec(), 256, 256)
    assert a.seg_ids.tobytes() == b.seg_ids.tobytes()


def test_offscreen_renderer_persistence(backend, composed):
    """§5.4: one persistent OffscreenRenderer, recreated only on size change."""
    backend.render_scene(composed, CameraSpec(), 256, 256)
    first = backend._renderer
    backend.render_scene(composed, CameraSpec(), 256, 256)
    assert backend._renderer is first              # reused at same size
    backend.render_scene(composed, CameraSpec(), 320, 320)
    assert backend._renderer is not first          # recreated on size change
    assert backend._size == (320, 320)


# --------------------------------------------------------- camera math §3.1 --
def test_orbit_eye_position_formula():
    cam = CameraSpec(azimuth_deg=35.0, elevation_deg=30.0, distance_m=1.1)
    pose = camera_math.pose_from_orbit(cam)
    az, el, d = math.radians(35.0), math.radians(30.0), 1.1
    expected_eye = np.array(cam.look_at) + d * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    np.testing.assert_allclose(pose[:3, 3], expected_eye, atol=1e-9)


def test_orbit_pose_axes():
    cam = CameraSpec(azimuth_deg=-120.0, elevation_deg=55.0, distance_m=1.8)
    pose = camera_math.pose_from_orbit(cam)
    rot = pose[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)  # orthonormal
    assert np.linalg.det(rot) == pytest.approx(1.0)                  # right-handed
    # camera -Z points from eye toward look_at; z_axis = normalize(eye - look_at)
    to_eye = pose[:3, 3] - np.array(cam.look_at)
    np.testing.assert_allclose(rot[:, 2], to_eye / np.linalg.norm(to_eye), atol=1e-12)
    assert rot[2, 1] > 0.0  # up axis points up-ish (world +Z component)


def test_intrinsics_formula():
    K = camera_math.intrinsics(256, 256, 50.0)
    f = 256 / (2.0 * math.tan(math.radians(50.0) / 2.0))
    assert K[0, 0] == pytest.approx(f) and K[1, 1] == pytest.approx(f)
    assert K[0, 2] == pytest.approx(128.0) and K[1, 2] == pytest.approx(128.0)
    assert K[2, 2] == 1.0 and K[0, 1] == 0.0


# --------------------------------------- 768 px case + inline InstanceLabels --
def bbox_and_rle(seg_ids: np.ndarray, iid: int) -> tuple[tuple[int, int, int, int], dict, np.ndarray]:
    """Inline label build: mask → cv2.boundingRect → pycocotools RLE (§8.1 shape,
    built here WITHOUT importing sceneforge.labels)."""
    mask = (seg_ids == iid).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)
    rle = cocomask.encode(np.asfortranarray(mask))
    rle = {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}
    return (int(x), int(y), int(w), int(h)), rle, mask.astype(bool)


def test_768_render_with_inline_instance_labels(backend, composed):
    cam = CameraSpec()
    r = backend.render_scene(composed, cam, 768, 768)

    assert set(np.unique(r.seg_ids)) - {0} == {BOX_ID, CYL_ID}
    assert np.all(r.depth_m[r.seg_ids > 0] > 0.0)

    labels = []
    for iid, asset_id in ((BOX_ID, "box"), (CYL_ID, "can")):
        bbox, rle, mask = bbox_and_rle(r.seg_ids, iid)
        label = InstanceLabel(
            instance_id=iid,
            asset_id=asset_id,
            category=asset_id,
            is_target=(iid == BOX_ID),
            bbox_xywh=bbox,
            area_px=int(mask.sum()),
            rle=rle,
        )
        labels.append(label)
        # RLE decodes back to the exact mask
        decoded = cocomask.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
        assert np.array_equal(decoded.astype(bool), mask)
        assert label.area_px >= 200 and label.area_px == int(cocomask.area(
            {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        ))
        # json-safe (python ints / ascii str only)
        json.dumps({"bbox": label.bbox_xywh, "area": label.area_px, "rle": label.rle})

    # NORMATIVE §3.1 projection sanity (the cross-backend mirror-bug check):
    # each instance's world centroid projects inside its own bbox.
    pose = camera_math.pose_from_orbit(cam)
    K = camera_math.intrinsics(768, 768, cam.yfov_deg)
    world_from_cam = pose
    cam_from_world = np.linalg.inv(world_from_cam)
    for (iid, mesh, t), label in zip(composed.instances, labels):
        centroid_w = (t @ np.append(mesh.bounding_box.centroid, 1.0))[:3]
        x_c, y_c, z_c = (cam_from_world @ np.append(centroid_w, 1.0))[:3]
        assert z_c < 0.0  # in front of the camera
        u = K[0, 2] + K[0, 0] * x_c / (-z_c)
        v = K[1, 2] - K[1, 1] * y_c / (-z_c)
        bx, by, bw, bh = label.bbox_xywh
        assert bx <= u <= bx + bw, f"instance {iid}: u={u} outside bbox {label.bbox_xywh}"
        assert by <= v <= by + bh, f"instance {iid}: v={v} outside bbox {label.bbox_xywh}"


# ------------------------------------------------------- contingency stub --
def test_numpy_backend_is_a_loud_stub(composed):
    with pytest.raises(NotImplementedError) as exc_info:
        numpy_backend.render_scene(composed, CameraSpec(), 64, 64)
    msg = str(exc_info.value)
    assert "IoU" in msg and "2 mm" in msg          # activation checklist present
    assert "u = cx + fx" in msg                    # normative equations carried
    with pytest.raises(NotImplementedError):
        numpy_backend.NumpyBackend().render_scene(composed, CameraSpec(), 64, 64)
