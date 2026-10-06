"""
run_persona.py - play one made-up person through Heim, then ask the probe questions.

Usage (on the server, from ~/heim-eval):
  ~/ai-continuity-assistant/venv/bin/python3 run_persona.py personas/P01.json \
      --backend anthropic --model claude-sonnet-4-6 --run 1

It uses only the heim_test database and the fake clock. Nothing here edits Heim.
Results are written to results/<persona>/<model>/run<N>/.
"""
import argparse
import json
import os
import re
import sys
import time

TABLES = ["concerns", "life_events", "recurring_patterns", "contradiction_log",
          "conversation_sessions", "daily_diaries", "user_profile",
          "ontology_triples", "mood_shifts", "profile_archive"]


def snapshot(db, uid):
    """What Heim has stored about this person right now (for the audit trail)."""
    conn = db.get_connection()
    cur = conn.cursor()
    out = {}
    for t in TABLES:
        try:
            cur.execute(f"SELECT * FROM {t} WHERE user_id = %s", (uid,))
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                row.pop("embedding", None)
                rows.append(row)
            out[t] = rows
        except Exception as e:
            conn.rollback()
            out[t] = {"error": str(e)[:120]}
    conn.close()
    return out


def end_conversation(db, ai, uid, conv):
    """Same steps as the /end_conversation route in app.py."""
    history = ai.get_chat_history(uid)
    hist_len = len(history)  # Heim only checks for contradictions if this is 16 or more
    if history:
        summary = ai.generate_conversation_summary(history)
        if summary:
            embedding = ai.generate_embedding(summary)
            db.save_conversation_session(conv, uid, summary, embedding)
    ai.check_for_contradictions(uid, history)
    try:
        ai.extract_ontology_triples(uid)
    except Exception as e:
        print("extract_ontology_triples failed:", e)
    return hist_len


def send(db, ai, clock, uid, conv, text):
    """One user message, in the same order as the /chat route in app.py."""
    db.update_last_seen(uid)
    reply, emotion, floor_color, leaving, _ = ai.handle_message(
        text, uid, conv, clock.date_str(), "text", clock.iso(), None)
    db.save_message(conv, "user", text)
    db.save_message(conv, "ai", reply)
    db.update_conversation_title(conv, text[:50])
    return reply, emotion


def run_persona(persona, db, ai, llm, clock, outdir, run_id=1):
    os.makedirs(outdir, exist_ok=True)
    transcript, snapshots, probe_rows = [], [], []

    username = f"{persona['id']}_run{run_id}_{int(time.time())}"
    uid = db.create_user(username, "pw-not-secret")
    db.create_initial_profile(uid)

    # ---- the story, session by session ----
    for si, session in enumerate(persona["sessions"], start=1):
        clock.set(session["date"], session.get("time", "20:00"))
        ai.reset_chat_history(uid)  # a new visit: the in-memory chat starts empty
        conv = db.create_conversation("New Chat", uid)
        for mi, text in enumerate(session["messages"]):
            if mi > 0:
                clock.advance(minutes=2)
            try:
                reply, emotion = send(db, ai, clock, uid, conv, text)
                err = None
            except Exception as e:
                reply, emotion, err = "", "", f"{type(e).__name__}: {str(e)[:200]}"
            transcript.append({
                "kind": "session", "session": si, "date": clock.date_str(),
                "message_index": mi, "user": text, "reply": reply, "emotion": emotion,
                "watch": session.get("watch") if mi == len(session["messages"]) - 1 else None,
                "error": err, "calls_so_far": llm.totals()["calls"]})
            print(f"  session {si} msg {mi + 1}/{len(session['messages'])} "
                  f"[{clock.date_str()}] calls={llm.totals()['calls']}"
                  + (f"  ERROR {err}" if err else ""))
        clock.advance(minutes=3)
        hist_len = None
        try:
            hist_len = end_conversation(db, ai, uid, conv)
        except Exception as e:
            print("  end_conversation failed:", e)
        print(f"  session {si} ended: chat history entries = {hist_len}")
        snapshots.append({"after_session": si, "date": clock.date_str(),
                          "history_len_at_end": hist_len,
                          "memory": snapshot(db, uid)})

    # ---- the probe questions ----
    for probe in persona["probes"]:
        clock.set(probe["date"], probe.get("time", "10:00"))
        ai.reset_chat_history(uid)
        conv = db.create_conversation("New Chat", uid)
        try:
            reply, _ = send(db, ai, clock, uid, conv, probe["question"])
            err = None
        except Exception as e:
            reply, err = "", f"{type(e).__name__}: {str(e)[:200]}"
        probe_rows.append({"probe": probe["id"], "kind": probe["kind"],
                           "date": clock.date_str(), "question": probe["question"],
                           "answer": reply, "error": err})
        print(f"  probe {probe['id']} answered ({len(reply)} chars)")

    snapshots.append({"after_session": "probes", "date": clock.date_str(),
                      "memory": snapshot(db, uid)})

    with open(os.path.join(outdir, "transcript.jsonl"), "w") as f:
        for row in transcript:
            f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(outdir, "probes.json"), "w") as f:
        json.dump({"persona": persona["id"], "answer_key": persona.get("key", {}),
                   "probes": probe_rows}, f, indent=2, default=str)
    with open(os.path.join(outdir, "snapshots.json"), "w") as f:
        json.dump(snapshots, f, indent=1, default=str)
    with open(os.path.join(outdir, "cost.json"), "w") as f:
        json.dump(llm.totals(), f, indent=2)
    return probe_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("persona")
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "openrouter"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="raise tiny max_tokens (needed for models that think before answering)")
    ap.add_argument("--max-calls", type=int, default=600,
                    help="hard stop after this many model calls")
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    from harness_core import FakeClock, LLM, setup, reset_test_db
    persona = json.load(open(args.persona))
    clock = FakeClock(persona["sessions"][0]["date"] + " 09:00")
    llm = LLM(args.backend, args.model, temperature=args.temperature,
              min_tokens=args.min_tokens, max_calls=args.max_calls)
    db, ai = setup(clock, llm)
    reset_test_db(db)

    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    outdir = os.path.join("results", persona["id"], safe_model, f"run{args.run}")
    os.makedirs(outdir, exist_ok=True)
    llm.log_path = os.path.join(outdir, "llm_calls.jsonl")
    llm.full_log_path = os.path.join(outdir, "calls_full.jsonl")
    for f in (llm.log_path, llm.full_log_path):  # a re-run replaces the old logs
        if os.path.exists(f):
            os.remove(f)
    print(f"Running {persona['id']} on {args.backend}:{args.model} (run {args.run})")
    run_persona(persona, db, ai, llm, clock, outdir, args.run)
    print("Done.", llm.totals())
    print("Results in", outdir)


if __name__ == "__main__":
    main()
