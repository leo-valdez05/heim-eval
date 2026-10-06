"""Judge: scores every saved answer against the expected answer.

The judge only sees: question, expected answer, answer. It never sees which system
(Heim / full / flat) wrote it, and it is a different model from the backbone.

Usage (run from ~/heim-eval, venv on):
  python3 judge.py score            # scores everything not yet scored (resumable)
  python3 judge.py summary          # prints the tables
  python3 judge.py sample 50        # writes 50 blinded rows for hand-checking
  python3 judge.py agree labels.csv # compares your labels with the judge
"""
import csv, glob, json, os, random, re, sys, time, statistics as st

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
OUT = "results/judge_scores.jsonl"
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}
SYSTEMS = {"claude-sonnet-4-6": "HEIM", "full__claude-sonnet-4-6": "FULL", "flat__claude-sonnet-4-6": "FLAT"}

SYSTEM_PROMPT = """You are a strict grader. You get a question, the expected answer, and an answer written by an AI companion.
Decide whether the answer is correct, judging only against the expected answer.

CORRECT: it gives the key fact(s) of the expected answer and says nothing that contradicts them.
PARTIAL: it gets some of the expected content but misses an important part, or mixes the right fact with a wrong one.
WRONG: it contradicts the expected answer, gives the old/outdated fact, claims not to know something it should know, or invents a change that did not happen.

If the question asks for a label (RECURRING or ONE-OFF) and a confidence (HIGH, MEDIUM or LOW), the answer is CORRECT only if BOTH match the expected answer.
Extra friendly detail is fine as long as it does not contradict the expected answer.

Reply with ONLY a JSON object: {"verdict": "CORRECT" | "PARTIAL" | "WRONG", "reason": "<one short sentence>"}"""


def load_env():
    try:
        from dotenv import load_dotenv
        for p in ("/home/ubuntu/ai-continuity-assistant/.env", ".env"):
            if os.path.exists(p):
                load_dotenv(p)
    except Exception:
        pass


def gather():
    """Yield one record per (persona, system, run, probe)."""
    rows = []
    for pdir in sorted(glob.glob("results/P*")):
        pname = os.path.basename(pdir)
        pf = f"personas/{pname[0].lower()}{pname[1:]}.json"
        if not os.path.exists(pf):
            continue
        probes = {p["id"]: p for p in json.load(open(pf))["probes"]}
        for sdir, label in SYSTEMS.items():
            for rd in sorted(glob.glob(f"{pdir}/{sdir}/run*")):
                f = os.path.join(rd, "probes.json")
                if not os.path.exists(f):
                    continue
                data = json.load(open(f))
                items = data if isinstance(data, list) else data.get("probes", data)
                for x in items:
                    if not (isinstance(x, dict) and "probe" in x) or x["probe"] not in probes:
                        continue
                    pr = probes[x["probe"]]
                    rows.append({"persona": pname, "system": label, "run": os.path.basename(rd),
                                 "probe": x["probe"], "kind": pr.get("kind", ""),
                                 "question": pr["question"], "expected": pr["expected"],
                                 "answer": x.get("answer") or ""})
    return rows


def key(r):
    return f'{r["persona"]}|{r["system"]}|{r["run"]}|{r["probe"]}'


def judge_one(client, r):
    msg = (f"Question: {r['question']}\n\nExpected answer: {r['expected']}\n\n"
           f"Answer to grade: {r['answer']}\n\nGrade it.")
    if os.environ.get("JUDGE_STUB"):
        return {"verdict": random.choice(list(SCORE)), "reason": "stub"}
    last = None
    for attempt in range(5):
        try:
            resp = client.messages.create(model=JUDGE_MODEL, max_tokens=200, temperature=0,
                                          system=SYSTEM_PROMPT,
                                          messages=[{"role": "user", "content": msg}])
            text = resp.content[0].text
            m = re.search(r"\{.*\}", text, re.S)
            out = json.loads(m.group(0))
            if out.get("verdict") in SCORE:
                return out
            last = f"bad verdict: {text[:100]}"
        except Exception as e:
            last = str(e)[:150]
            time.sleep(2 * (attempt + 1))
    return {"verdict": "ERROR", "reason": last}


def cmd_score():
    load_env()
    client = None
    if not os.environ.get("JUDGE_STUB"):
        import anthropic
        client = anthropic.Anthropic()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l)
                if d["verdict"] in SCORE:
                    done.add(d["key"])
            except Exception:
                pass
    rows = [r for r in gather() if key(r) not in done]
    print(f"{len(rows)} answers to score (judge model: {JUDGE_MODEL})")
    with open(OUT, "a") as fo:
        for i, r in enumerate(rows, 1):
            out = judge_one(client, r)
            rec = {"key": key(r), "persona": r["persona"], "system": r["system"], "run": r["run"],
                   "probe": r["probe"], "kind": r["kind"], "verdict": out["verdict"], "reason": out["reason"]}
            fo.write(json.dumps(rec) + "\n");
            fo.flush()
            if i % 20 == 0:
                print(f"  scored {i}/{len(rows)}")
    print("done. Next: python3 judge.py summary")


def load_scores():
    best = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l)
            if d["verdict"] in SCORE:
                best[d["key"]] = d  # last good one wins
    return list(best.values())


def ms(xs):
    return f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}" if len(xs) > 1 else (f"{xs[0]:.2f}" if xs else "-")


def cmd_summary():
    sc = load_scores()
    if not sc:
        print("no scores yet");
        return
    print(f"{len(sc)} scored answers\n")
    print("MEAN SCORE (1 = correct, 0.5 = partial, 0 = wrong); mean ± spread across people")
    print("Each person's runs are averaged first, so every person counts once.\n")
    kinds = sorted({d["kind"] for d in sc})
    header = f'{"system":<8}{"OVERALL":>16}' + "".join(f"{k[:14]:>16}" for k in kinds)
    print(header)
    for sysname in ("HEIM", "FULL", "FLAT"):
        line = f"{sysname:<8}"
        for k in [None] + kinds:
            per = {}
            for d in sc:
                if d["system"] == sysname and (k is None or d["kind"] == k):
                    per.setdefault(d["persona"], []).append(SCORE[d["verdict"]])
            line += f"{ms([st.mean(v) for v in per.values()]):>16}"
        print(line)
    print("\nPEOPLE COVERED PER SYSTEM:")
    for sysname in ("HEIM", "FULL", "FLAT"):
        ps = sorted({d["persona"] for d in sc if d["system"] == sysname})
        runs = len({(d["persona"], d["run"]) for d in sc if d["system"] == sysname})
        print(f"  {sysname}: {len(ps)} people, {runs} runs")
    print("\nWRONG ANSWERS BY SYSTEM AND QUESTION KIND (counts):")
    for sysname in ("HEIM", "FULL", "FLAT"):
        cnt = {}
        for d in sc:
            if d["system"] == sysname and d["verdict"] == "WRONG":
                cnt[d["kind"]] = cnt.get(d["kind"], 0) + 1
        print(f"  {sysname}: {cnt or 'none'}")


def cmd_sample(n):
    sc = {d["key"]: d for d in load_scores()}
    rows = [r for r in gather() if key(r) in sc]
    random.Random(7).shuffle(rows)
    rows = rows[:n]
    with open("results/handcheck_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "expected", "answer", "your_verdict (CORRECT/PARTIAL/WRONG)"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["question"], r["expected"], r["answer"], ""])
    json.dump({i: sc[key(r)]["verdict"] for i, r in enumerate(rows, 1)},
              open("results/handcheck_key.json", "w"))
    print(f"wrote results/handcheck_sample.csv ({len(rows)} rows, system names hidden)")
    print("Fill the last column, save, then: python3 judge.py agree results/handcheck_sample.csv")


def cmd_agree(path):
    key_ = json.load(open("results/handcheck_key.json"))
    same = tot = 0
    for row in csv.DictReader(open(path)):
        mine = (row.get("your_verdict (CORRECT/PARTIAL/WRONG)") or "").strip().upper()
        if mine in SCORE:
            tot += 1
            same += (mine == key_[row["id"]])
    print(f"you labelled {tot}; agreement with the judge = {same}/{tot}" + (f" = {same / tot:.0%}" if tot else ""))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "score":
        cmd_score()
    elif a[0] == "summary":
        cmd_summary()
    elif a[0] == "sample":
        cmd_sample(int(a[1]) if len(a) > 1 else 50)
    elif a[0] == "agree":
        cmd_agree(a[1])
    else:
        print(__doc__)
