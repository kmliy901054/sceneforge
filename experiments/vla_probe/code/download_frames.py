"""Download distinct BridgeData V2 raw frames + temporal neighbors for the
appearance-invariance probe. Provenance recorded in images/provenance.json.

Source: https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/
Layout: <scene>/<task>/<id>/<date>/raw/traj_group0/trajN/{images0/im_K.jpg, lang.txt}
"""
import json
import os
import re
import sys
import urllib.request

BASE = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2"
OUT = "/home/pairlab/DGAN/sceneforge/experiments/vla_probe/images/originals"
os.makedirs(OUT, exist_ok=True)

HREF = re.compile(r'href="([^"?/][^"?]*)"')


def listing(url):
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  listing FAIL {url}: {e}")
        return []
    return HREF.findall(html)


def fetch(url, path):
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  fetch FAIL {url}: {e}")
        return False


def fetch_text(url):
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


# (scene, task, trajs_wanted) — diverse scenes/backgrounds, tabletop manipulation
COMBOS = [
    ("datacol2_toykitchen7", "drawer_pnp", 2),
    ("datacol2_toykitchen7", "stack_blocks", 2),
    ("datacol2_folding_table", None, 2),
    ("datacol2_tabletop_dark_wood", None, 2),
    ("datacol2_toykitchen2", None, 2),
    ("datacol1_toykitchen6", None, 2),
    ("deepthought_robot_desk", None, 2),
    ("deepthought_folding_table", None, 2),
    ("minsky_folding_table_white_tray", None, 2),
    ("datacol2_toykitchen6", None, 2),
    ("datacol2_robot_desk", None, 2),
]

provenance = []
fid = 0

for scene, task, n_want in COMBOS:
    if task is None:
        tasks = [t.rstrip("/") for t in listing(f"{BASE}/{scene}/") if t.endswith("/")]
        if not tasks:
            continue
        task = tasks[0]
    print(f"[{scene}/{task}] want {n_want}")
    ids = [i.rstrip("/") for i in listing(f"{BASE}/{scene}/{task}/") if i.endswith("/")]
    got = 0
    for id_ in ids:
        if got >= n_want:
            break
        dates = [d.rstrip("/") for d in listing(f"{BASE}/{scene}/{task}/{id_}/") if d.endswith("/")]
        for date in dates:
            if got >= n_want:
                break
            root = f"{BASE}/{scene}/{task}/{id_}/{date}/raw/traj_group0"
            trajs = [t.rstrip("/") for t in listing(f"{root}/") if t.startswith("traj")]
            if not trajs:
                continue
            # take up to 2 well-separated trajs from this group
            pick = [trajs[0]] if len(trajs) < 4 else [trajs[0], trajs[len(trajs) // 2]]
            for traj in pick:
                if got >= n_want:
                    break
                turl = f"{root}/{traj}"
                lang = fetch_text(f"{turl}/lang.txt")
                if not lang or not lang.strip():
                    continue
                instruction = lang.strip().splitlines()[0].strip()
                if not instruction or instruction.startswith("confidence"):
                    continue
                imgs = listing(f"{turl}/images0/")
                idxs = sorted(
                    int(m.group(1)) for m in (re.match(r"im_(\d+)\.jpg", i) for i in imgs) if m
                )
                if len(idxs) < 10:
                    continue
                k = idxs[max(2, len(idxs) // 3)]
                k2_pos = min(len(idxs) - 1, max(2, len(idxs) // 3) + 3)
                k2 = idxs[k2_pos]
                if k2 == k:
                    continue
                name = f"frame{fid:02d}"
                ok1 = fetch(f"{turl}/images0/im_{k}.jpg", f"{OUT}/{name}.jpg")
                ok2 = fetch(f"{turl}/images0/im_{k2}.jpg", f"{OUT}/{name}_tplus.jpg")
                if not (ok1 and ok2):
                    continue
                provenance.append(
                    {
                        "frame": name,
                        "scene": scene,
                        "task_dir": task,
                        "traj_url": turl,
                        "frame_index": k,
                        "neighbor_index": k2,
                        "n_frames_in_episode": len(idxs),
                        "instruction": instruction,
                        "lang_raw": lang.strip(),
                    }
                )
                print(f"  {name}: im_{k} (+{k2}) '{instruction}'")
                fid += 1
                got += 1

with open(
    "/home/pairlab/DGAN/sceneforge/experiments/vla_probe/images/provenance.json", "w"
) as f:
    json.dump(provenance, f, indent=2)
print(f"\nTOTAL {fid} frames")
sys.exit(0 if fid >= 20 else 1)
