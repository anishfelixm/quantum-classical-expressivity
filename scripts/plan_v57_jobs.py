"""
Plan the Amendment 20, 21 and 23 runs: the gate (quick checks), wave 1 and
wave 2, with a runtime projection made from the recorded timings of the very
runs being repeated.

    python scripts/plan_v57_jobs.py                     # projection only, writes nothing
    python scripts/plan_v57_jobs.py --write logs_v57    # job scripts for the gate and wave 1
    python scripts/plan_v57_jobs.py --write logs_v57 --wave 2
                                                        # after wave 1: unclipped re-executions
                                                        # of every run whose R_ record shows a
                                                        # step above 20

Standard library only. Reads shards; writes only under --write.

WHAT IS PLANNED, IN PRIORITY ORDER (jobs start in this order; within a tier,
longest first, which minimises the time to finish a tier for a fixed J)
--------------------------------------------------------------------------
  tier 1  R_01_frozen_tuned       every primary-namespace run, original protocol
  tier 2  B_01_frozen_tuned       Amendment 20: five arms, every n, ten seeds, 1,000 epochs
  tier 3  R_ of 26_bottleneck_tuned, 27_capacity_tuned, 14_readout,
          03_robustness_v2, 10_capacity, 12_bottleneck, 25_encoder
  tier 4  R_ of 15_dim8, 16_dim16, 21_ansatz, 18_depth, 19_angle, 20_tanh,
          29_compression_frozen, 09_lr_selection
  tier 5  33_hardware_tuned       Amendment 23: 07's cells at the tuned rate
  tier 6  R_17_fulldata, then B_17_fulldata (quantum_vqc and the matched head)
  tier 7  C_17_fulldata           the full-data runs re-executed on CPU, the device
                                  they were recorded on (added 17 October): an
                                  identity test like every R_, launched with CUDA
                                  hidden
  wave 2  U_<source>              clipping removed, for runs whose R_ record has a
                                  step above 20 (known only after wave 1)

Every R_, B_ and U_ run is executed with the code of the commit that recorded
the run it repeats (replay.py --code-at), so jobs are grouped by commit too.

THE PROJECTION
--------------
  R_ and U_ runs: the recorded run's own training time (epochs x mean epoch
     time for runs of 01; recorded wall time for 03, 07, 09, 10, 12), x 1.10.
  B_ runs: the first 100 epochs repeat the 100-epoch run exactly, so the best
     epoch is at least the recorded one and training lasts at least
     max(best + 200, floor) + 1 epochs (lower bound) and at most 1,000 (upper
     bound). Planning value = lower bound x the largest ratio actual/lower
     bound observed in Amendment 19 (printed).
  17_fulldata ran on CPU, so its recorded time does not transfer. Its GPU time
     is projected from the primary runs' GPU cost per training-plus-validation
     image-epoch for the same arm (printed, with that assumption).

The makespan figures assume J jobs run at the speed one job ran alone; they
are a floor. The first jobs' logs show the real rate within minutes.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.environ.get("QML_ARTIFACT_ROOT", os.path.join(REPO, "artifacts"))
SHARDS = os.path.join(ART, "shards")
OVERHEAD = 1.10
ALL_SEEDS = [42, 123, 2026, 777, 888, 31337, 8, 271828, 161803, 1414]
VAL_FULL = {"breastmnist": 78, "pneumoniamnist": 524, "bloodmnist": 1712, "pathmnist": 10004}
TIERS = [
    (1, "R", ["01_frozen_tuned"]),
    (2, "B", ["01_frozen_tuned"]),
    (3, "R", ["26_bottleneck_tuned", "27_capacity_tuned", "14_readout", "03_robustness_v2",
              "10_capacity", "12_bottleneck", "25_encoder"]),
    (4, "R", ["15_dim8", "16_dim16", "21_ansatz", "18_depth", "19_angle", "20_tanh",
              "29_compression_frozen", "09_lr_selection"]),
    (5, "H", ["07_hardware_noise"]),
    (6, "R", ["17_fulldata"]),
    (6, "B", ["17_fulldata"]),
    (7, "C", ["17_fulldata"]),
]
B_ARMS = {"01_frozen_tuned": None, "17_fulldata": {"quantum_vqc", "matched_param_fullrank"}}
BUDGET = "1000:200"


def name(ns, keys):
    st = lambda v: str(v).replace("/", "-").replace(" ", "")   # noqa: E731  (= shards._stringify)
    return "__".join([ns] + [f"{k}{st(v)}" for k, v in sorted(keys.items())]) + ".json"


def load(ns):
    d = os.path.join(SHARDS, ns)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def run_hours(r):
    m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
    if m.get("epochs_run") is not None and m.get("mean_epoch_time") is not None:
        return m["epochs_run"] * m["mean_epoch_time"] / 3600.0
    if r.get("wall_time") is not None:
        return r["wall_time"] / 3600.0
    return None


def floor_epochs(n_train, max_epochs=1000):
    b = (int(n_train) + 31) // 32
    return min(max(20, 200 // max(b, 1)), max_epochs // 2)


def regime_n(k):
    return int(k["regime"]) if str(k["regime"]).isdigit() else 10 ** 9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", default=None)
    p.add_argument("--wave", type=int, default=1, choices=[1, 2])
    p.add_argument("--target", type=float, default=1.0, help="max planning hours per job")
    p.add_argument("--kinds", default="R,B,H,C",
                   help="wave 1 only: which kinds go into the job list (R, B, H, C); the "
                        "gate is always written from all of them")
    p.add_argument("--tag", default=None,
                   help="name of the job list (default wave1 / wave2): jobs_<tag>.txt")
    args = p.parse_args()
    tag = args.tag or f"wave{args.wave}"
    kinds = set(args.kinds.split(",")) if args.wave == 1 else {"U"}
    py = "python -u scripts/replay.py"
    # Hardware runs evaluate density matrices on CPU, one image at a time. With
    # torch's default of one thread per core (256 here) four such jobs crawled:
    # 17 h for one run that took 15 min alone (17 October). Their numbers do not
    # depend on the thread count beyond floating-point order, and no identity is
    # tested for them, so they run with a small, fixed thread count.
    HW_ENV = "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "

    # ---- GPU cost per image-epoch, per arm, from the primary runs (for 17_fulldata)
    pri = load("01_frozen_tuned")
    rate = collections.defaultdict(lambda: [0.0, 0.0])
    for r in pri:
        m, meta = r["metrics"], r.get("meta", {})
        nt = meta.get("n_train")
        if nt is None or m.get("mean_epoch_time") is None:
            continue
        rate[r["keys"]["arm"]][0] += m["epochs_run"] * m["mean_epoch_time"]
        rate[r["keys"]["arm"]][1] += m["epochs_run"] * 3 * nt        # n_val = 2n per class
    gpu_rate = {a: v[0] / v[1] for a, v in rate.items() if v[1]}

    # ---- Amendment 19's ratio actual / lower bound
    by = {name("x", r["keys"]): r for r in pri}
    ratios = []
    for r in load("30_budget_check"):
        q = by.get(name("x", r["keys"]))
        if q:
            m = q["metrics"]
            lb = min(1000, max(m["best_epoch"] + 200, floor_epochs(q["meta"]["n_train"])) + 1)
            ratios.append(r["metrics"]["epochs_run"] / lb)
    mult = max(ratios) if ratios else 2.0
    print(f"Amendment 19 (n=5): actual/lower-bound epochs over {len(ratios)} runs: max {mult:.2f} "
          f"-> B_ planning value = lower bound x {mult:.2f}")
    if gpu_rate:
        print("GPU seconds per training+validation image-epoch (primary runs): "
              + ", ".join(f"{a} {v * 1e3:.3f} ms" for a, v in sorted(gpu_rate.items())))

    units = []      # one per run
    if args.wave == 1:
        for tier, kind, sources in TIERS:
            for src in sources:
                recs = load(src)
                if not recs:
                    print(f"WARNING: MISSING SOURCE {src}: no shards under {SHARDS}; not planned")
                    continue
                for r in recs:
                    k = r["keys"]
                    sha = str(r.get("git_sha", "?"))
                    if kind == "B" and B_ARMS[src] is not None and k["arm"] not in B_ARMS[src]:
                        continue
                    if kind == "B" and src == "01_frozen_tuned" and int(k["seed"]) not in ALL_SEEDS:
                        continue
                    tgt = {"R": "R_", "B": "B_", "C": "C_", "H": ""}[kind] + (src if kind != "H" else "33_hardware_tuned")
                    done = os.path.exists(os.path.join(SHARDS, tgt, name(tgt, k)))
                    h = run_hours(r)
                    m = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
                    if src == "17_fulldata" and kind != "C":
                        nt = r.get("meta", {}).get("n_train", 0)
                        per = gpu_rate.get(k["arm"])
                        h = (per * m.get("epochs_run", 100) * (nt + VAL_FULL.get(k["dataset"], 0)) / 3600
                             if per else h)
                    lo = hi = plan = (h or 0.0) * OVERHEAD
                    if kind == "B":
                        per_epoch = (h or 0.0) * OVERHEAD / max(m.get("epochs_run", 100), 1)
                        lb = min(1000, max(m.get("best_epoch", 99) + 200,
                                           floor_epochs(r["meta"]["n_train"])) + 1)
                        lo, hi, plan = lb * per_epoch, 1000 * per_epoch, lb * per_epoch * mult
                    units.append(dict(tier=tier, kind=kind, src=src, tgt=tgt, sha=sha, keys=k,
                                      done=done, lo=lo, hi=hi, plan=plan,
                                      gb=4 * int((r.get("capacity") or {}).get("total", 0) or 0) / 1e9))
    else:
        for tier, kind, sources in TIERS:
            if kind != "R":
                continue
            for src in sources:
                for r in load("R_" + src):
                    rec = r.get("recorded", {})
                    if (rec.get("above_threshold") or 0) == 0 and (rec.get("nonfinite") or 0) == 0:
                        continue
                    k = r["keys"]
                    orig = [q for q in load(src) if name("x", q["keys"]) == name("x", k)]
                    sha = str(orig[0].get("git_sha", "?")) if orig else str(r.get("git_sha"))
                    tgt = "U_" + src
                    h = (run_hours(r) or 0.0) * OVERHEAD
                    units.append(dict(tier=tier, kind="U", src=src, tgt=tgt, sha=sha, keys=k,
                                      done=os.path.exists(os.path.join(SHARDS, tgt, name(tgt, k))),
                                      lo=h, hi=h, plan=h, gb=0.0))

    # ---- projection
    tot = collections.OrderedDict()
    for u in units:
        t = tot.setdefault((u["tier"], u["tgt"]), [0, 0, 0.0, 0.0, 0.0, 0.0])
        t[0] += 1
        t[1] += u["done"]
        if not u["done"]:
            t[2] += u["lo"]
            t[3] += u["plan"]
            t[4] += u["hi"]
            t[5] += u["gb"]
    print(f"\n{'tier':>4s} {'target namespace':30s} {'runs':>6s} {'done':>6s} {'lower h':>8s} "
          f"{'planning h':>10s} {'upper h':>8s} {'weights GB':>10s}")
    for (tier, tgt), t in tot.items():
        print(f"{tier:4d} {tgt:30s} {t[0]:6d} {t[1]:6d} {t[2]:8.1f} {t[3]:10.1f} {t[4]:8.1f} {t[5]:10.2f}")
    print(f"{'':4s} {'TOTAL':30s} {sum(t[0] for t in tot.values()):6d} "
          f"{sum(t[1] for t in tot.values()):6d} {sum(t[2] for t in tot.values()):8.1f} "
          f"{sum(t[3] for t in tot.values()):10.1f} {sum(t[4] for t in tot.values()):8.1f} "
          f"{sum(t[5] for t in tot.values()):10.2f}")

    # ---- jobs: per (tier, kind, src, sha, arm, dataset, regime), chunked by seeds
    groups = collections.OrderedDict()
    for u in units:
        if u["done"] or u["kind"] not in kinds:
            continue
        k = u["keys"]
        g = (u["tier"], u["kind"], u["src"], u["sha"], k["arm"], k["dataset"], str(k["regime"]))
        groups.setdefault(g, []).append(u)
    jobs = []
    for g, us in groups.items():
        total = sum(u["plan"] for u in us)
        nchunk = max(1, min(len({u["keys"]["seed"] for u in us}), math.ceil(total / args.target)))
        seeds = sorted({int(u["keys"]["seed"]) for u in us})
        chunks = [seeds[i::nchunk] for i in range(nchunk)]
        for c in chunks:
            cu = [u for u in us if int(u["keys"]["seed"]) in set(c)]
            jobs.append((g, c, cu))
    jobs.sort(key=lambda j: (j[0][0], -sum(u["plan"] for u in j[2])))
    print(f"\n{len(jobs)} jobs; longest planning value "
          f"{max((sum(u['plan'] for u in j[2]) for j in jobs), default=0):.2f} h")
    print("Makespan if parallel jobs do not slow each other (a floor):")
    for J in (4, 6, 8, 12, 16):
        span = {}
        for key in ("plan", "hi"):
            lanes = [0.0] * J
            for j in jobs:
                i = lanes.index(min(lanes))
                lanes[i] += sum(u[key] for u in j[2])
            span[key] = max(lanes)
        print(f"  J={J:2d}  planning {span['plan']:6.1f} h   upper {span['hi']:6.1f} h")
    shas = sorted({u["sha"] for u in units if u["kind"] != "H"})
    print(f"\ncommits whose code is needed: {', '.join(s[:8] for s in shas)}")

    if not args.write:
        return
    os.makedirs(args.write, exist_ok=True)
    jd = os.path.join(args.write, f"jobs_{tag}")
    os.makedirs(jd, exist_ok=True)
    status = os.path.join(args.write, "status.txt")
    lines = []

    def script(fname, cmd, note):
        path = os.path.join(jd, fname + ".sh")
        log = os.path.join(args.write, fname + ".log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\n# " + note + "\n")
            f.write(f"echo \"$(date '+%F %T') START {fname}\" >> {status}\n")
            f.write(f"if {cmd} > {log} 2>&1; then\n")
            f.write(f"  echo \"$(date '+%F %T') OK    {fname}\" >> {status}\n")
            f.write("else\n")
            f.write(f"  echo \"$(date '+%F %T') FAIL  {fname} (see {log})\" >> {status}\n")
            f.write("fi\n")
        return path

    for i, (g, seeds, cu) in enumerate(jobs):
        tier, kind, src, sha, arm, ds, reg = g
        where = f"--where arm={arm} dataset={ds} regime={reg}"
        sd = "--seeds " + " ".join(map(str, seeds))
        if kind == "H":
            cmd = f"{HW_ENV}{py} --hardware-tuned {where} {sd}"
        elif kind == "B":
            cmd = f"{py} --source {src} --code-at {sha} --budget {BUDGET} {where} {sd}"
        elif kind == "U":
            cmd = f"{py} --source {src} --code-at {sha} --clip inf {where} {sd}"
        elif kind == "C":
            cmd = f"CUDA_VISIBLE_DEVICES= {py} --source {src} --code-at {sha} --clip 20 {where} {sd}"
        else:
            cmd = f"{py} --source {src} --code-at {sha} --clip 20 {where} {sd}"
        fname = f"{i:04d}_t{tier}{kind}_{src[:2]}_{arm}_{ds}_n{reg}_{len(seeds)}s"
        note = (f"planning {sum(u['plan'] for u in cu):.2f} h, upper {sum(u['hi'] for u in cu):.2f} h, "
                f"{len(cu)} runs, commit {sha[:8]}")
        lines.append(script(fname, cmd, note))
    with open(os.path.join(args.write, f"jobs_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.write, f"commits_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(shas) + "\n")

    if args.wave == 1:
        gate = []
        seen = set()
        for u in units:
            if u["kind"] not in ("R", "C") or (u["src"], u["sha"], u["kind"]) in seen:
                continue
            if u["kind"] == "C":
                seen.add((u["src"], u["sha"], "C"))
                gate.append(script(f"gate_cpu_{u['src']}_{u['sha'][:8]}",
                                   f"CUDA_VISIBLE_DEVICES= {py} --source {u['src']} --code-at {u['sha']} --quick",
                                   "CPU re-execution of a CPU-recorded source: must be bit-identical"))
                continue
            seen.add((u["src"], u["sha"], "R"))
            fname = f"gate_{u['src']}_{u['sha'][:8]}"
            gate.append(script(fname, f"{py} --source {u['src']} --code-at {u['sha']} --quick",
                               "quick check: re-executions must be bit-identical"))
        # Runs the original logs show built their own feature cache
        # (docs/cold_cache_prediction.json): the quick set above draws from ALL_SEEDS
        # and so rarely contains one. One cheapest such run per source and commit
        # must also be bit-identical, re-executed in that state.
        import predict_cold_runs as pcr
        table = pcr.load_table()
        cold = {}
        for u in units:
            if u["kind"] != "R" or u["src"] == "17_fulldata":
                continue
            if pcr.predicted(table, u["src"], {"keys": u["keys"], "git_sha": u["sha"]}) == "cold":
                key = (u["src"], u["sha"])
                if key not in cold or u["plan"] < cold[key]["plan"]:
                    cold[key] = u
        for (src, sha), u in sorted(cold.items()):
            lst = os.path.join(args.write, f"gate_cold_{src}_{sha[:8]}.list")
            with open(lst, "w", encoding="utf-8") as f:
                f.write(name(src, u["keys"]) + "\n")
            gate.append(script(f"gate_cold_{src}_{sha[:8]}",
                               f"{py} --source {src} --code-at {sha} --from-list {lst} --gate",
                               "a run the logs show built its feature cache: must be bit-identical"))
        gate.append(script("gate_hardware_tuned", f"{HW_ENV}{py} --hardware-tuned --quick",
                           "smoke test of the Amendment 23 path"))
        b01 = sorted({u["sha"] for u in units if u["kind"] == "B" and u["src"] == "01_frozen_tuned"})
        if b01:
            gate.append(script("gate_budget_01", f"{py} --source 01_frozen_tuned --code-at {b01[0]} "
                                                 f"--budget {BUDGET} --quick",
                               "Amendment 20 path: the first epochs must repeat the recorded run exactly"))
        with open(os.path.join(args.write, "jobs_gate.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(gate) + "\n")
        print(f"wrote {len(gate)} gate jobs and {len(lines)} wave-1 jobs under {args.write}")
    else:
        print(f"wrote {len(lines)} wave-2 jobs under {args.write}")


if __name__ == "__main__":
    main()
