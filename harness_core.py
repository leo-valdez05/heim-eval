"""
harness_core.py - shared pieces of the Heim evaluation harness.

It does three jobs. It never edits Heim's code or prompts.

1. FakeClock + install_clock()
   Lets us replay weeks of conversation in minutes.
   - Python side: datetime.datetime is replaced by a version whose now() returns the fake time.
   - Database side: NOW() and CURRENT_DATE inside every SQL statement are rewritten
     to the fake time before the statement runs.

2. connect_test_db()
   Points Heim's database module at the database named heim_test, and refuses to run
   if the connection is to anything else. The live database (heim_db) cannot be reached.

3. LLM
   A stand-in for the `client` object Heim calls. It can talk to Claude or to any
   OpenRouter model (for example openai/gpt-oss-20b), and it logs every call.
"""
import os
import re
import sys
import json
import time
import datetime as _dt
import urllib.request
import urllib.error

HEIM_PATH = os.path.expanduser(os.environ.get("HEIM_PATH", "~/ai-continuity-assistant"))
TEST_DB = "heim_test"

_REAL_DATETIME = _dt.datetime
_clock = None


# ----------------------------------------------------------------------------
# 1. Fake clock
# ----------------------------------------------------------------------------
class FakeClock:
    def __init__(self, start="2026-10-01 09:00"):
        self.set(start)

    def set(self, date_str, time_str=None):
        if time_str is None:
            parts = date_str.replace("T", " ").split(" ")
            date_str = parts[0]
            time_str = parts[1] if len(parts) > 1 else "09:00"
        fmt = "%Y-%m-%d %H:%M:%S" if time_str.count(":") == 2 else "%Y-%m-%d %H:%M"
        self._now = _REAL_DATETIME.strptime(f"{date_str} {time_str}", fmt)

    def now(self):
        return self._now

    def date_str(self):
        return self._now.strftime("%Y-%m-%d")

    def iso(self):
        return self._now.isoformat()

    def advance(self, **kwargs):
        self._now = self._now + _dt.timedelta(**kwargs)


def install_clock(clock):
    """Call this BEFORE importing Heim's modules."""
    global _clock
    _clock = clock

    class FakeDatetime(_REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return _REAL_DATETIME.now(tz)
            n = _clock.now()
            return cls(n.year, n.month, n.day, n.hour, n.minute, n.second)

        @classmethod
        def today(cls):
            return cls.now()

        @classmethod
        def utcnow(cls):
            return cls.now()

    _dt.datetime = FakeDatetime
    for name in ("emotion_ai2", "database"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "datetime", None) is _REAL_DATETIME:
            m.datetime = FakeDatetime
    return FakeDatetime


_NOW_RE = re.compile(r"\bNOW\(\)", re.I)
_TODAY_RE = re.compile(r"\bCURRENT_DATE\b", re.I)


def rewrite_sql(sql):
    """Replace the database's own clock with the fake clock."""
    if _clock is None or not isinstance(sql, str):
        return sql
    ts = _clock.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = _NOW_RE.sub(f"TIMESTAMP '{ts}'", sql)
    sql = _TODAY_RE.sub(f"DATE '{ts[:10]}'", sql)
    return sql


# ----------------------------------------------------------------------------
# 2. Safe test database
# ----------------------------------------------------------------------------
def test_database_url():
    """Heim's DATABASE_URL with the database name swapped to heim_test.
    Returns None if DATABASE_URL is not set. The URL is never printed."""
    from urllib.parse import urlparse, urlunparse
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    parts = urlparse(url)
    return urlunparse(parts._replace(path="/" + TEST_DB))


def connect_test_db():
    import psycopg2
    import psycopg2.extensions
    from dotenv import load_dotenv

    if HEIM_PATH not in sys.path:
        sys.path.insert(0, HEIM_PATH)
    load_dotenv(os.path.join(HEIM_PATH, ".env"))

    class _Cursor(psycopg2.extensions.cursor):
        def execute(self, sql, vars=None):
            return super().execute(rewrite_sql(sql), vars)

    def get_test_connection():
        url = test_database_url()
        if url:
            return psycopg2.connect(url, cursor_factory=_Cursor)
        return psycopg2.connect(
            host="localhost",
            port="5432",
            database=TEST_DB,
            user="postgres",
            password=os.environ.get("DB_PASSWORD"),
            cursor_factory=_Cursor,
        )

    conn = get_test_connection()
    cur = conn.cursor()
    cur.execute("SELECT current_database()")
    name = cur.fetchone()[0]
    conn.close()
    if name != TEST_DB:
        raise SystemExit(f"REFUSING TO RUN: connected to '{name}', expected '{TEST_DB}'")

    import database
    database.get_connection = get_test_connection
    return database


def reset_test_db(db):
    """Empty every table in heim_test (and nothing else)."""
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT current_database()")
    if cur.fetchone()[0] != TEST_DB:
        conn.close()
        raise SystemExit("REFUSING TO RESET: not the test database")
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    tables = [r[0] for r in cur.fetchall()]
    if tables:
        cur.execute("TRUNCATE " + ", ".join(tables) + " RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()
    ai = sys.modules.get("emotion_ai2")
    if ai is not None:
        ai.chat_history_by_user.clear()


# ----------------------------------------------------------------------------
# 3. Model stand-in
# ----------------------------------------------------------------------------
class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]


class LLM:
    """Looks like anthropic.Anthropic() to Heim: client.messages.create(...)."""

    def __init__(self, backend, model, temperature=0.0, min_tokens=0,
                 log_path="llm_calls.jsonl", max_calls=None):
        assert backend in ("anthropic", "openrouter")
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.min_tokens = min_tokens
        self.log_path = log_path
        self.max_calls = max_calls
        self.messages = self  # so client.messages.create(...) reaches create()
        self._client = None
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self.failures = 0

    # --- public ---
    def create(self, model=None, max_tokens=1024, system=None, messages=None, **_ignored):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise SystemExit(f"CALL BUDGET REACHED ({self.max_calls} calls). Stopping to protect your credit.")
        start = time.time()
        text, tin, tout, error = "", 0, 0, None
        try:
            if self.backend == "anthropic":
                text, tin, tout = self._call_anthropic(max_tokens, system, messages)
            else:
                text, tin, tout = self._call_openrouter(max_tokens, system, messages)
        except Exception as e:
            error = str(e)[:300]
            self.failures += 1
            raise
        finally:
            secs = time.time() - start
            self.calls += 1
            self.tokens_in += tin
            self.tokens_out += tout
            self.seconds += secs
            self._log({"backend": self.backend, "model": self.model,
                       "max_tokens": max_tokens, "tokens_in": tin,
                       "tokens_out": tout, "seconds": round(secs, 2),
                       "error": error})
        return _Response(text)

    def totals(self):
        return {"calls": self.calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "seconds": round(self.seconds, 1),
                "failures": self.failures}

    # --- backends ---
    def _call_anthropic(self, max_tokens, system, messages):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      system=system or "", messages=messages)
        try:
            resp = self._client.messages.create(temperature=self.temperature, **kwargs)
        except Exception as e:
            if "temperature" in str(e).lower():
                resp = self._client.messages.create(**kwargs)
            else:
                raise
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _call_openrouter(self, max_tokens, system, messages):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max(max_tokens, self.min_tokens),
            "temperature": self.temperature,
            "reasoning": {"effort": "low"},
        }
        body = json.dumps(payload).encode("utf-8")
        last_error = None
        for attempt in range(4):
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read().decode("utf-8"))
                text = (data["choices"][0]["message"].get("content") or "")
                usage = data.get("usage", {})
                return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as e:
                last_error = str(e)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter failed after retries: {last_error}")

    def _log(self, row):
        try:
            row["time"] = _REAL_DATETIME.now().isoformat(timespec="seconds")
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# One call that wires everything together
# ----------------------------------------------------------------------------
def setup(clock, llm):
    """Order matters: clock first, then database, then Heim, then the model stand-in."""
    install_clock(clock)
    db = connect_test_db()  # also loads Heim's .env
    os.environ.setdefault("ANTHROPIC_API_KEY", "unused-placeholder")
    import emotion_ai2
    emotion_ai2.client = llm
    return db, emotion_ai2
