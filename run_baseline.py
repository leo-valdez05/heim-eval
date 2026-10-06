"""
Run one persona through a SIMPLE memory system (a baseline), for comparison with Heim.

  --system full   Full context: the model is given the whole dated conversation history every time.
  --system flat   Flat memory store (Mem0-style, our own implementation): after each message an LLM
                  pulls out facts, and each fact is ADDed / UPDATEd / DELETEd / ignored against the
                  most similar stored facts. Answers use the top-matching stored facts only.

Neither baseline uses Heim's code, database, profile, diaries, follow-ups or prompts.
They use one short neutral prompt (below), the same for both.
Outputs: results/<persona>/<system>__<model>/run<N>/{transcript.jsonl, probes.json, cost.json, ...}
"""
import argparse, json, math, os, re, sys, time

COMPANION = ("You are a warm, attentive AI companion in an ongoing relationship with the user. "
             "Reply naturally and briefly. If the user asks about something they never told you, "
             "say you don't know instead of guessing.")

EXTRACT = ("You maintain a memory of facts about the user. From the user's latest message, extract atomic, "
           "self-contained facts about the user worth remembering (identity, situation, plans, events with dates, "
           "preferences, worries, corrections). Resolve relative dates using today's date. "
           "If the user corrects something earlier, extract the corrected fact. "
           "Return ONLY a JSON list of strings. Return [] if there is nothing worth remembering.")

DECIDE = ("You update a memory store. For each NEW fact, compare it with the SIMILAR existing memories shown "
          "(each has an id). Choose one operation per new fact:\n"
          "ADD (new information), UPDATE (replace the text of an existing memory with newer or more complete "
          "information; give its id and the new text), DELETE (an existing memory is wrong or contradicted; give its id), "
          "NONE (already stored).\n"
          "Return ONLY a JSON list like: "
          '[{"fact": 0, "op": "ADD", "text": "..."}, {"fact": 1, "op": "UPDATE", "id": 3, "text": "..."}, '
          '{"fact": 2, "op": "DELETE", "id": 5}, {"fact": 3, "op": "NONE"}]')


# ---------- small helpers ----------
def embed(text):
    """Same embedding model Heim uses (gemini-embedding-001, 768 dims)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    for attempt in range(4):
        try:
            r = client.models.embed_content(model="gemini-embedding-001", contents=text,
                                            config=types.EmbedContentConfig(output_dimensionality=768))
            return list(r.embeddings[0].values)
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding failed: {err}")


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a));
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def parse_json(text, default):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\[.*\]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return default


def ask(llm, system, messages, max_tokens=700):
    return llm.create(max_tokens=max_tokens, system=system, messages=messages).content[0].text.strip()


# ---------- the two systems ----------
class FullContext:
    name = "full"

    def __init__(self, llm):
        self.llm = llm
        self.sessions = []  # list of (date, [(role, text), ...])

    def start_session(self, date):
        self.sessions.append((date, []))

    def _history_text(self, upto_current):
        parts = []
        for i, (d, turns) in enumerate(self.sessions):
            if i == len(self.sessions) - 1 and not upto_current:
                continue
            body = "\n".join(f"{'User' if r == 'user' else 'You'}: {t}" for r, t in turns)
            parts.append(f"[Conversation on {d}]\n{body}")
        return "\n\n".join(parts)

    def reply(self, date, text):
        cur = self.sessions[-1][1]
        cur.append(("user", text))
        sys_prompt = (COMPANION + f"\n\nToday's date is {date}.\n\nYour earlier conversations with the user:\n"
                      + (self._history_text(upto_current=False) or "(none yet)"))
        msgs = [{"role": r if r == "user" else "assistant", "content": t} for r, t in cur]
        out = ask(self.llm, sys_prompt, msgs, 600)
        cur.append(("assistant", out))
        return out

    def answer(self, date, question):
        sys_prompt = (COMPANION + f"\n\nToday's date is {date}.\n\nYour earlier conversations with the user:\n"
                      + (self._history_text(upto_current=True) or "(none yet)"))
        return ask(self.llm, sys_prompt, [{"role": "user", "content": question}], 700)

    def dump(self):
        return {"sessions": [{"date": d, "turns": t} for d, t in self.sessions]}


class FlatStore:
    name = "flat"
    TOP_REPLY, TOP_ANSWER, TOP_SIMILAR = 5, 10, 3

    def __init__(self, llm):
        self.llm = llm
        self.mem = {}  # id -> {"text", "created", "updated", "emb"}
        self.next_id = 1
        self.session_turns = []
        self.log = []  # what the store did, for the audit trail

    def start_session(self, date):
        self.session_turns = []

    def _top(self, query, k):
        if not self.mem:
            return []
        q = embed(query)
        scored = sorted(self.mem.items(), key=lambda kv: cosine(q, kv[1]["emb"]), reverse=True)
        return scored[:k]

    def _fmt(self, items):
        return "\n".join(f"- [{m['updated']}] {m['text']}" for _, m in items) or "(nothing stored yet)"

    def reply(self, date, text):
        recalled = self._top(text, self.TOP_REPLY)
        sys_prompt = (COMPANION + f"\n\nToday's date is {date}.\n\nThings you remember about the user:\n"
                      + self._fmt(recalled))
        self.session_turns.append({"role": "user", "content": text})
        out = ask(self.llm, sys_prompt, list(self.session_turns), 600)
        self.session_turns.append({"role": "assistant", "content": out})
        self._remember(date, text)
        return out

    def _remember(self, date, text):
        raw = ask(self.llm, EXTRACT + f"\nToday's date is {date}.", [{"role": "user", "content": text}], 500)
        facts = [f for f in parse_json(raw, []) if isinstance(f, str) and f.strip()]
        if not facts:
            return
        block = []
        for i, f in enumerate(facts):
            sim = self._top(f, self.TOP_SIMILAR)
            lines = "\n".join(f"    id {mid}: {m['text']}" for mid, m in sim) or "    (none)"
            block.append(f"NEW FACT {i}: {f}\n  similar existing memories:\n{lines}")
        raw2 = ask(self.llm, DECIDE, [{"role": "user", "content": "\n\n".join(block)}], 800)
        ops = parse_json(raw2, [])
        for op in ops if isinstance(ops, list) else []:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).upper()
            try:
                fi = int(op.get("fact", -1))
            except Exception:
                fi = -1
            if kind == "ADD":
                t = op.get("text") or (facts[fi] if 0 <= fi < len(facts) else None)
                if t:
                    self.mem[self.next_id] = {"text": t, "created": date, "updated": date, "emb": embed(t)}
                    self.log.append({"date": date, "op": "ADD", "id": self.next_id, "text": t})
                    self.next_id += 1
            elif kind == "UPDATE" and op.get("id") in self.mem and op.get("text"):
                m = self.mem[op["id"]]
                self.log.append({"date": date, "op": "UPDATE", "id": op["id"], "old": m["text"], "text": op["text"]})
                m.update(text=op["text"], updated=date, emb=embed(op["text"]))
            elif kind == "DELETE" and op.get("id") in self.mem:
                self.log.append({"date": date, "op": "DELETE", "id": op["id"], "old": self.mem[op["id"]]["text"]})
                del self.mem[op["id"]]

    def answer(self, date, question):
        recalled = self._top(question, self.TOP_ANSWER)
        sys_prompt = (COMPANION + f"\n\nToday's date is {date}.\n\nThings you remember about the user:\n"
                      + self._fmt(recalled))
        return ask(self.llm, sys_prompt, [{"role": "user", "content": question}], 700)

    def dump(self):
        return {"memories": [{"id": i, "text": m["text"], "created": m["created"], "updated": m["updated"]}
                             for i, m in self.mem.items()], "operations": self.log}


# ---------- runner ----------
def run(persona, system, llm, outdir):
    os.makedirs(outdir, exist_ok=True)
    transcript, probe_rows = [], []
    for si, session in enumerate(persona["sessions"], start=1):
        system.start_session(session["date"])
        for mi, text in enumerate(session["messages"]):
            try:
                reply, err = system.reply(session["date"], text), None
            except Exception as e:
                reply, err = "", f"{type(e).__name__}: {str(e)[:200]}"
            transcript.append({"kind": "session", "session": si, "date": session["date"], "message_index": mi,
                               "user": text, "reply": reply, "error": err,
                               "watch": session.get("watch") if mi == len(session["messages"]) - 1 else None,
                               "calls_so_far": llm.totals()["calls"]})
            print(f"  session {si} msg {mi + 1}/{len(session['messages'])} [{session['date']}] "
                  f"calls={llm.totals()['calls']}" + (f"  ERROR {err}" if err else ""), flush=True)
    for pr in persona["probes"]:
        try:
            ans, err = system.answer(pr["date"], pr["question"]), None
        except Exception as e:
            ans, err = "", f"{type(e).__name__}: {str(e)[:200]}"
        probe_rows.append({"probe": pr["id"], "kind": pr["kind"], "date": pr["date"],
                           "question": pr["question"], "answer": ans, "error": err})
        print(f"  probe {pr['id']} answered ({len(ans)} chars)", flush=True)
    with open(os.path.join(outdir, "transcript.jsonl"), "w") as f:
        for row in transcript:
            f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(outdir, "memory.json"), "w") as f:
        json.dump(system.dump(), f, indent=1, default=str)
    with open(os.path.join(outdir, "cost.json"), "w") as f:
        json.dump(llm.totals(), f, indent=2)
    with open(os.path.join(outdir, "probes.json"), "w") as f:  # written last: marks the run as finished
        json.dump({"persona": persona["id"], "system": system.name,
                   "answer_key": persona.get("key", {}), "probes": probe_rows}, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("persona")
    ap.add_argument("--system", required=True, choices=["full", "flat"])
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "openrouter", "groq"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=600)
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from harness_core import LLM, HEIM_PATH
    try:  # keys (Anthropic, Google) live in Heim's .env
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HEIM_PATH, ".env"))
    except Exception:
        pass

    persona = json.load(open(args.persona))
    llm = LLM(args.backend, args.model, temperature=args.temperature,
              min_tokens=args.min_tokens, max_calls=args.max_calls)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    outdir = os.path.join("results", persona["id"], f"{args.system}__{safe_model}", f"run{args.run}")
    os.makedirs(outdir, exist_ok=True)
    llm.log_path = os.path.join(outdir, "llm_calls.jsonl")
    llm.full_log_path = os.path.join(outdir, "calls_full.jsonl")
    for f in (llm.log_path, llm.full_log_path):
        if os.path.exists(f):
            os.remove(f)
    system = FullContext(llm) if args.system == "full" else FlatStore(llm)
    print(f"Running {persona['id']} on {args.system} / {args.backend}:{args.model} (run {args.run})", flush=True)
    run(persona, system, llm, outdir)
    print("Done.", llm.totals(), flush=True)
    print("Results in", outdir, flush=True)


if __name__ == "__main__":
    main()
