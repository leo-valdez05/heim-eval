"""
Run many persona runs at once, each on its own test database.

Example:
  python3 batch.py --personas p02 p03 p04 p05 --runs 1 --backend anthropic --model claude-sonnet-4-6

It is safe to stop and start again: runs that already finished are skipped.
It never touches Heim's real database (each worker uses heim_test, heim_test2, ...).
"""
import argparse, json, os, re, subprocess, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", required=True, help="e.g. p02 p03 (file names without .json)")
    ap.add_argument("--runs", nargs="+", type=int, default=[1])
    ap.add_argument("--backend", default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=600)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--system", default="heim", choices=["heim", "full", "flat"],
                    help="heim = Heim itself; full / flat = the two simple baselines")
    ap.add_argument("--script", default=None)
    ap.add_argument("--dbs", nargs="+", default=["heim_test", "heim_test2", "heim_test3", "heim_test4"])
    args = ap.parse_args()

    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    model_dir = safe_model if args.system == "heim" else f"{args.system}__{safe_model}"
    script = args.script or ("run_persona.py" if args.system == "heim" else "run_baseline.py")
    os.makedirs("results/batch_logs", exist_ok=True)

    jobs = []
    for name in args.personas:
        path = f"personas/{name}.json"
        pid = json.load(open(path))["id"]
        for r in args.runs:
            done = os.path.exists(f"results/{pid}/{model_dir}/run{r}/probes.json")
            jobs.append({"path": path, "pid": pid, "run": r, "done": done})
    todo = [j for j in jobs if not j["done"]]
    if args.system == "heim":
        free = list(args.dbs[:args.workers])  # Heim needs one test database per worker
    else:
        free = [f"slot{i}" for i in range(args.workers)]  # baselines use no database
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already finished, {len(todo)} to run, "
          f"{len(free)} at a time ({args.system})", flush=True)
    running = []  # (proc, job, db, started)
    finished = []
    t0 = time.time()
    while todo or running:
        while todo and free:
            j = todo.pop(0)
            db = free.pop(0)
            log = open(f"results/batch_logs/{j['pid']}_{model_dir}_run{j['run']}.log", "w")
            cmd = [sys.executable, "-u", script, j["path"], "--backend", args.backend,
                   "--model", args.model, "--max-calls", str(args.max_calls),
                   "--min-tokens", str(args.min_tokens), "--run", str(j["run"])]
            if args.system != "heim":
                cmd += ["--system", args.system]
            env = dict(os.environ)
            if args.system == "heim":
                env["HEIM_TEST_DB"] = db
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, j, db, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] started {j['pid']} run{j['run']} on {db}", flush=True)
        time.sleep(5)
        still = []
        for p, j, db, started in running:
            code = p.poll()
            if code is None:
                still.append((p, j, db, started))
                continue
            free.append(db)
            ok = os.path.exists(f"results/{j['pid']}/{model_dir}/run{j['run']}/probes.json")
            finished.append((j, ok, code, time.time() - started))
            print(f"[{time.strftime('%H:%M:%S')}] {'finished' if ok else 'FAILED'} {j['pid']} run{j['run']} "
                  f"(exit {code}, {int((time.time() - started) / 60)} min)", flush=True)
        running = still

    bad = [f for f in finished if not f[1]]
    print(f"\nAll done in {int((time.time() - t0) / 60)} min. OK: {len(finished) - len(bad)}  failed: {len(bad)}",
          flush=True)
    for j, ok, code, secs in bad:
        print("  failed:", j["pid"], "run", j["run"], "- see results/batch_logs/")


if __name__ == "__main__":
    main()