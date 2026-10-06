"""
Self-test 1: does the fake clock move both Python and the database?
No AI calls are made, so this costs nothing. It only touches the heim_test database.

Run on the server:   cd ~/heim-eval && python3 selftest_clock.py
"""
from harness_core import FakeClock, LLM, setup, reset_test_db

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


clock = FakeClock("2026-10-01 09:00")
llm = LLM("anthropic", "claude-sonnet-4-6")  # never called in this test
db, ai = setup(clock, llm)
reset_test_db(db)

uid = db.create_user("selftest", "pw12345")
db.create_initial_profile(uid)
conv = db.create_conversation("selftest chat", uid)

db.save_life_event({
    "emotion": "anxious", "concern": "database exam", "state": "worried",
    "resolved": False, "severity": "medium", "event_worthy": True,
    "is_future_event": True, "followup_date": "2026-10-10",
    "message": "I have my database exam on Oct 10",
    "date": clock.date_str(), "user_id": uid, "conversation_id": conv,
    "local_time": clock.iso(),
})

# Day 1: the exam is in the future
ctx = ai.get_temporal_context(uid, clock.iso())
check("day 1: exam is labelled FUTURE", "FUTURE" in ctx and "PAST" not in ctx)
check("day 1: nothing is due for follow-up yet",
      db.get_followups(uid, clock.date_str(), None) == [])

# Database clock: a saved message gets the FAKE time, not the real time
db.save_message(conv, "user", "hello")
conn = db.get_connection();
cur = conn.cursor()
cur.execute("SELECT time FROM messages WHERE conversation_id = %s", (conv,))
stamp = str(cur.fetchone()[0])[:19]
conn.close()
check("database writes use the fake time", stamp == "2026-10-01 09:00:00", stamp)

# Day 12: the exam date has passed
clock.set("2026-10-12", "09:00")
ctx = ai.get_temporal_context(uid, clock.iso())
check("day 12: exam is labelled PAST", "PAST" in ctx and "FUTURE" not in ctx)
due = db.get_followups(uid, clock.date_str(), conv)
check("day 12: the exam is due for follow-up", len(due) == 1)

# Counting: at most once per day
if due:
    event_id = due[0][1]
    db.increment_followup_count(event_id, clock.date_str(), conv)
    db.increment_followup_count(event_id, clock.date_str(), conv)  # same day again
    conn = db.get_connection();
    cur = conn.cursor()
    cur.execute("SELECT followup_count FROM life_events WHERE id = %s", (event_id,))
    n1 = cur.fetchone()[0];
    conn.close()
    check("same day counts only once", n1 == 1, f"count={n1}")
    clock.set("2026-10-13", "09:00")
    db.increment_followup_count(event_id, clock.date_str(), conv)
    conn = db.get_connection();
    cur = conn.cursor()
    cur.execute("SELECT followup_count FROM life_events WHERE id = %s", (event_id,))
    n2 = cur.fetchone()[0];
    conn.close()
    check("next day counts again", n2 == 2, f"count={n2}")

# 15-day profile cycle uses the fake clock
clock.set("2026-10-17", "09:00")
last = db.get_last_evolved(uid)
days = (ai.datetime.now() - last).days
check("15-day cycle sees 16 days pass", days >= 15, f"days={days}")

check("no AI calls were made", llm.totals()["calls"] == 0)

reset_test_db(db)
print()
print("ALL PASS" if all(results) else "SOME FAILED - paste this whole output to Claude")
