import argparse
import json
import time
import os
import sys

# Ensure 'cv' on sys.path
# Ensure repository root on sys.path so 'GPU' package resolves
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from GPU.prod.app import run_detection_only, run_tracking


def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark: detection and tracking on 1/3/11 cameras")
    p.add_argument("--duration", type=int, default=60, help="Seconds per run")
    p.add_argument("--sets", type=str, default="1,3,11", help="Which sets to run (comma sep)")
    p.add_argument("--cams1", type=str, default="8", help="Comma list for 1-cam set")
    p.add_argument("--cams3", type=str, default="8,9,10", help="Comma list for 3-cam set")
    p.add_argument("--cams11", type=str, default="1,2,3,4,5,6,7,8,9,10,11", help="Comma list for 11-cam set")
    p.add_argument("--with-gui", action="store_true", help="Enable GUI during runs")
    return p.parse_args()


def str_to_int_list(s: str):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def run_one(label: str, cams: list, duration: int, with_gui: bool):
    print(f"\n=== {label}: DETECTION-ONLY cams={cams} dur={duration}s gui={with_gui} ===")
    det_summary = run_detection_only(cams, duration_s=duration, show_gui=with_gui)
    print(json.dumps(det_summary, indent=2))

    print(f"\n=== {label}: TRACKING cams={cams} dur={duration}s gui={with_gui} ===")
    trk_summary = run_tracking(cams, duration_s=duration, show_gui=with_gui)
    print(json.dumps(trk_summary, indent=2))

    return {"label": label, "cams": cams, "detection": det_summary, "tracking": trk_summary}


def main():
    args = parse_args()
    sets = [s.strip() for s in args.sets.split(',')]

    cfg = {
        "1": str_to_int_list(args.cams1),
        "3": str_to_int_list(args.cams3),
        "11": str_to_int_list(args.cams11),
    }

    results = []
    start_all = time.time()

    for s in sets:
        cams = cfg.get(s)
        if not cams:
            print(f"Skipping set {s}: no cameras")
            continue
        label = f"SET-{s}"
        res = run_one(label, cams, args.duration, args.with_gui)
        results.append(res)

    elapsed_all = time.time() - start_all
    print("\n================= SUMMARY REPORT =================")
    # Compact comparison
    for res in results:
        lbl = res["label"]
        cams = res["cams"]
        det = res["detection"]
        trk = res["tracking"]
        print(f"\n[{lbl}] cams={cams}")
        print(f"  DET: total={det.get('total_msgs')} rate={det.get('avg_msgs_per_sec'):.2f}/s max_q={det.get('max_queue_size')}")
        print(f"  TRK: total={trk.get('total_msgs')} rate={trk.get('avg_msgs_per_sec'):.2f}/s max_q={trk.get('max_queue_size')}")

    print(f"\nTotal wall time: {elapsed_all:.1f}s")


if __name__ == "__main__":
    main()

