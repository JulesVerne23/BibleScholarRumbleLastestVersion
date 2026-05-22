import requests
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import re
import json
import os
import sqlite3
from queue import Queue, Empty
from collections import OrderedDict
import threading
import time
import datetime
# python-dotenv is optional but recommended for credential management.
# Install with: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Selenium imports — install with: pip install selenium webdriver-manager
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL = "RumbleBot"
OLLAMA_URL = "http://localhost:11434/api/chat"

# ── Rumble credentials ──────────────────────────────────────────────────────
# Set RUMBLE_USERNAME and RUMBLE_PASSWORD in a .env file or as environment
# variables. Never hard-code credentials in source. Example .env file:
#   RUMBLE_USERNAME=you@example.com
#   RUMBLE_PASSWORD=yourpassword
RUMBLE_USERNAME = os.getenv("RUMBLE_USERNAME", "")
RUMBLE_PASSWORD = os.getenv("RUMBLE_PASSWORD", "")

# The @-handle the bot should watch for (without the @).
BOT_NAME = os.getenv("RUMBLE_BOT_NAME", "BibleScholar23")

# How often (seconds) to poll the chat for new @-mentions
RUMBLE_POLL_INTERVAL = 4

# Hard character ceiling for Rumble responses (total across all messages).
# Editable at runtime via the UI — the bot loop always reads this variable.
RUMBLE_CHAR_LIMIT = 780

# Maximum characters per individual chat entry (Rumble's per-message limit).
# The bot splits the full response into chunks of this size before posting.
RUMBLE_ENTRY_CHARS = 200

def _calc_max_parts() -> int:
    """How many 200-char chunks fit in the current RUMBLE_CHAR_LIMIT, minimum 1."""
    return max(1, -(-RUMBLE_CHAR_LIMIT // RUMBLE_ENTRY_CHARS))  # ceiling division

# ── Ollama tuning ────────────────────────────────────────────────────────────
# RUMBLE_NUM_CTX must be large enough to hold:
#   - the injected system prompt (~400 tokens)
#   - Heiser context blocks (~300 tokens)
#   - the user question (~50 tokens)
#   - the full 3-message response output (~1100 tokens)
# 4096 was too tight — the model was hitting the wall mid-sentence.
# 8192 gives comfortable headroom for input + full output.
RUMBLE_NUM_CTX = 8192

def _calc_num_predict(char_limit: int) -> int:
    """
    Convert the total character ceiling into a token budget for the full
    multi-message response.

    char_limit here is RUMBLE_CHAR_LIMIT (the total per-response budget),
    NOT RUMBLE_ENTRY_CHARS (the per-message chunk size). The two are separate:
      - RUMBLE_CHAR_LIMIT controls how much the AI writes in total.
      - RUMBLE_ENTRY_CHARS controls how that total is split across chat messages.

    ~3 chars per token for English prose. We add 100% headroom (2x) so the model
    always has room to finish its final sentence. num_predict is a ceiling not a
    target — the model stops naturally when done. Too-low causes mid-sentence cuts.
    sanitize_rumble_response clips runaway output afterward.

    Floor at 768 so even small limits get a usable budget.

    Previous version mistakenly multiplied char_limit by max_parts before
    converting to tokens. max_parts is derived FROM char_limit, so that
    caused the budget to scale quadratically (e.g. 780-char limit → 4 parts
    → 3120 chars used as input instead of 780). The fix: convert char_limit
    directly to tokens with 2x headroom.
    """
    tokens_needed = char_limit / 3               # chars -> tokens (~3 chars/token)
    return max(768, int(tokens_needed * 2.0))    # 2x headroom; floor at 768

RUMBLE_NUM_PREDICT = _calc_num_predict(RUMBLE_CHAR_LIMIT)

# How many conversation turns to keep in UI history (prevents context overflow)
MAX_HISTORY_TURNS = 20

# How many consecutive poll errors before auto-disconnect
MAX_POLL_ERRORS = 10

# Activity log file — saved next to the Python script
ACTIVITY_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rumble_activity.log")

def write_activity_log(event_type: str, author: str, content: str, response: str = "") -> bool:
    """
    Append a timestamped entry to the activity log file.
    event_type: MENTION | RESPONSE | SUSPICIOUS | ERROR | BLOCKED
    Returns True if the message is suspicious (caller can skip Ollama).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prompt injection detection — expanded to cover ChatML and role-injection patterns
    injection_patterns = [
        "ignore previous", "ignore your", "ignore all", "disregard",
        "new instructions", "system prompt", "you are now", "pretend you",
        "act as", "jailbreak", "dan mode", "<tool_call>", "tool_call",
        "{{", "}}", "<|im_start|>", "<|im_end|>", "<|system|>", "override",
        "forget everything", "your real instructions",
        "assistant:", "system:", "role:", "new role",
        "ignore the above", "ignore all previous",
        "disregard the above", "disregard all previous",
    ]
    lower_content = content.lower()
    suspicious = any(p in lower_content for p in injection_patterns)
    if suspicious and event_type == "MENTION":
        event_type = "SUSPICIOUS"

    line = (
        f"[{timestamp}] [{event_type}]\n"
        f"  FROM    : {author}\n"
        f"  MESSAGE : {content}\n"
    )
    if response:
        line += f"  RESPONSE: {response}\n"
    if suspicious:
        line += "  *** POSSIBLE PROMPT INJECTION DETECTED ***\n"
    line += "-" * 60 + "\n"

    try:
        with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[Warning] Could not write to activity log: {e}", flush=True)

    return suspicious

# -----------------------------
# NET BIBLE JSON LOADING & LOOKUP
# -----------------------------
# net_structured.json format: { "BookName": { "chapter_str": { "verse_str": "text" } } }
NET_LOOKUP = {}

def load_net_json(path="net_structured.json"):
    global NET_LOOKUP

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, path)

    if not os.path.exists(json_path):
        print(f"[Warning] net_structured.json not found at {json_path}. Verse lookup will be unavailable.")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            NET_LOOKUP = json.load(f)
        print(f"[Info] NET Bible loaded: {len(NET_LOOKUP)} books.")
    except Exception as e:
        print(f"[Warning] Could not load net_structured.json: {e}. Verse lookup will be unavailable.")

load_net_json()

# -----------------------------
# STRONG'S JSON LOADING & LOOKUP
# -----------------------------
STRONGS_LOOKUP = {}

def normalize_strongs_key(raw_key: str) -> str:
    raw_key = str(raw_key).strip().upper()

    if raw_key.startswith(("H", "G")):
        prefix = raw_key[0]
        number = raw_key[1:]
    else:
        prefix = "H"
        number = raw_key

    number = number.lstrip("0")
    if number == "":
        number = "0"

    return f"{prefix}{number}"

def load_strongs_json(path="strongs.json"):
    global STRONGS_LOOKUP

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, path)

    if not os.path.exists(json_path):
        print(f"[Warning] strongs.json not found at {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        STRONGS_LOOKUP = json.load(f)

def lookup_strongs(key_or_number):
    normalized = normalize_strongs_key(key_or_number)
    return STRONGS_LOOKUP.get(normalized)

load_strongs_json(path="strongs3.json")

# -----------------------------
# HEISER KNOWLEDGE BASE LOADING & SEARCH
# (Divine Council / ANE / exegetical entries — Michael Heiser)
# -----------------------------
HEISER_KNOWLEDGE = []

# Heiser-specific files: Divine Council, ANE exegesis, supernatural worldview.
_HEISER_FILES = [
    "BibleScholar_Knowledge_Demons_Unclean_Spirits.json",
    "Jesus_and_the_Gates_of_Hell.json",
    "Elisha_and_the_Bear.json",
]

# -----------------------------
# APOLOGETICS KNOWLEDGE BASE LOADING & SEARCH
# (Classical and evidentialist Christian apologetics)
# -----------------------------
APOLOGETICS_KNOWLEDGE = []

# Apologetics-specific files: Lewis, Geisler/Turek, etc.
_APOLOGETICS_FILES = [
    "Mere_Christianity_Knowledge.json",
    "I_dont_have_enough_faith_to_be_an_atheist.json",
]


def _load_knowledge_files(paths, label: str) -> list:
    """Shared loader for any list of knowledge JSON files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_entries = []

    for path in paths if isinstance(paths, (list, tuple)) else [paths]:
        json_path = os.path.join(base_dir, path)
        if not os.path.exists(json_path):
            print(f"[Warning] {label} knowledge file not found: {path}")
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_entries.extend(data)
            else:
                all_entries.append(data)
            print(f"[Info] Loaded {len(data) if isinstance(data, list) else 1} entries from {path}")
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")

    print(f"[Info] Total {label} knowledge entries loaded: {len(all_entries)}")
    return all_entries


def load_heiser_knowledge(paths=None):
    """Load Heiser (Divine Council / ANE) knowledge JSON files."""
    global HEISER_KNOWLEDGE
    HEISER_KNOWLEDGE = _load_knowledge_files(
        paths if paths is not None else _HEISER_FILES,
        label="Heiser"
    )


def load_apologetics_knowledge(paths=None):
    """Load apologetics knowledge JSON files."""
    global APOLOGETICS_KNOWLEDGE
    APOLOGETICS_KNOWLEDGE = _load_knowledge_files(
        paths if paths is not None else _APOLOGETICS_FILES,
        label="Apologetics"
    )


def _search_knowledge(
    knowledge: list,
    query: str,
    max_results: int,
    header: str,
    footer: str,
) -> str:
    """Shared keyword scorer used by both knowledge base search functions."""
    if not knowledge:
        return ""

    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w{4,}\b', query_lower))

    scored = []
    for entry in knowledge:
        score = 0
        haystack = " ".join([
            entry.get("title", ""),
            entry.get("summary", ""),
            entry.get("embedding_text", ""),
            " ".join(entry.get("tags", [])),
        ]).lower()

        for word in query_words:
            if word in haystack:
                score += 1
        for word in query_words:
            if word in [t.lower() for t in entry.get("tags", [])]:
                score += 2

        if score > 0:
            scored.append((score, entry))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_results]

    lines = [header]
    for _, entry in top:
        title   = entry.get("title", "Untitled")
        chapter = entry.get("chapter", "")
        summary = entry.get("summary", "").strip()
        etext   = entry.get("embedding_text", "").strip()
        source  = entry.get("source", "")

        detail = etext if etext else summary
        if len(detail) > 500:
            detail = detail[:500] + "…"

        lines.append(f"• [{chapter}] {title} ({source})")
        lines.append(f"  {detail}")

    lines.append(footer)
    return "\n".join(lines)


def search_heiser_knowledge(query: str, max_results: int = 3) -> str:
    """Search the Heiser / Divine Council / ANE knowledge base."""
    return _search_knowledge(
        HEISER_KNOWLEDGE,
        query,
        max_results,
        header="[HEISER KNOWLEDGE BASE — relevant entries found]",
        footer="[End of Heiser knowledge — use this to inform your answer if relevant]",
    )


# =============================================================================
# APOLOGETICS TRIGGER LIST
# Organized by category. Any phrase match gives every apologetics entry a
# flat APOLOGETICS_TRIGGER_BONUS before the keyword scorer runs, ensuring
# that even short or vague questions pull in the right material.
# The bonus is intentionally large so trigger matches beat marginal keyword
# overlaps from the Heiser base.
# =============================================================================

APOLOGETICS_TRIGGER_BONUS = 6  # added to every entry's score on any trigger match

_APOLOGETICS_TRIGGERS: dict[str, list[str]] = {

    # 1. Existence of God
    "existence_of_god": [
        "cosmological argument", "teleological argument", "fine-tuning",
        "first cause", "uncaused cause", "necessary being",
        "contingent universe", "moral lawgiver", "objective morality",
        "intelligent design", "irreducible complexity",
        "origin of the universe", "origin of life",
        "why is there something rather than nothing",
    ],

    # 2. Jesus & Resurrection
    "resurrection": [
        "minimal facts", "empty tomb", "eyewitness testimony",
        "hallucination theory", "swoon theory", "legend theory",
        "early creed", "1 corinthians 15", "resurrection evidence",
        "historical jesus", "extra-biblical sources",
        "tacitus", "josephus",
        # natural-language variants
        "jesus rose", "jesus risen", "rose from the dead",
        "risen from the dead", "did jesus rise", "resurrection of jesus",
        "resurrection of christ", "did christ rise",
    ],

    # 3. Bible Reliability
    "bible_reliability": [
        "manuscript evidence", "textual criticism", "variants",
        "dead sea scrolls", "canon formation", "gospels contradiction",
        "inerrancy", "authorship", "eyewitness accounts",
        "archaeological confirmation", "historical reliability",
    ],

    # 4. Moral & Philosophical
    "moral_philosophy": [
        "objective morality", "moral relativism", "moral argument",
        "free will", "determinism", "moral ontology",
        "evil exists", "problem of evil", "suffering",
        "meaning and purpose", "human dignity", "value of human life",
    ],

    # 5. Science & Faith
    "science_and_faith": [
        "big bang origin", "fine-tuning constants", "multiverse",
        "abiogenesis", "information in dna", "genetic code",
        "irreducible complexity", "cambrian explosion",
        "fossil record", "naturalism", "materialism", "scientism",
        # natural-language variants
        "science and faith", "science disproves", "science vs religion",
        "science versus religion", "science vs faith",
        "evolution and god", "evolution disproves", "can you believe science and",
        "does science", "religion and science",
    ],

    # 6. Atheist / Agnostic Claims
    "atheist_claims": [
        "no evidence for god", "religion is man-made",
        "god of the gaps", "flying spaghetti monster",
        "burden of proof", "extraordinary claims",
        "faith is irrational", "problem of suffering",
        "contradictions in the bible", "science disproves religion",
        "evolution disproves god",
        # natural-language variants
        "why believe in god", "why should i believe",
        "prove god exists", "god doesn't exist", "god does not exist",
        "there is no god", "atheist", "agnostic", "i don't believe",
        "i dont believe in god", "religion is just",
    ],

    # 7. Worldview & Cultural
    "worldview": [
        "worldview comparison", "secular humanism", "postmodernism",
        "relativism", "pluralism", "exclusive claims", "all religions",
        "tolerance", "truth claim",
    ],

    # 8. Historical Evidence
    "historical_evidence": [
        "historical evidence", "ancient sources", "roman records",
        "non-christian sources", "secular historians", "early church",
        "church fathers", "martyrdom",
    ],
}

# Flattened list of (phrase, category) pairs for fast scanning.
_APOLOGETICS_TRIGGER_PAIRS: list[tuple[str, str]] = [
    (phrase, cat)
    for cat, phrases in _APOLOGETICS_TRIGGERS.items()
    for phrase in phrases
]


def _apologetics_trigger_bonus(query: str) -> tuple[int, list[str]]:
    """
    Scan query for apologetics trigger phrases.
    Returns (bonus_score, matched_categories).
    bonus_score is APOLOGETICS_TRIGGER_BONUS if any phrase matched, else 0.
    matched_categories is deduplicated for logging.
    """
    q = query.lower()
    matched_cats: list[str] = []
    for phrase, cat in _APOLOGETICS_TRIGGER_PAIRS:
        if phrase in q:
            if cat not in matched_cats:
                matched_cats.append(cat)
    bonus = APOLOGETICS_TRIGGER_BONUS if matched_cats else 0
    return bonus, matched_cats


def search_apologetics_knowledge(query: str, max_results: int = 3) -> str:
    """
    Search the apologetics knowledge base (Lewis, Geisler/Turek, etc.).

    Scoring:
      - Standard keyword overlap score (4+ letter words, 2x bonus for tag hits)
      - Flat APOLOGETICS_TRIGGER_BONUS added to every entry when any trigger
        phrase is detected in the query — ensures category-relevant questions
        always surface apologetics material even with loose wording.
    """
    if not APOLOGETICS_KNOWLEDGE:
        return ""

    bonus, matched_cats = _apologetics_trigger_bonus(query)
    if matched_cats:
        print(f"[Apologetics] Trigger match — categories: {', '.join(matched_cats)} "
              f"(+{bonus} base score)")

    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))  # 3+ chars (catches God, sin, law)

    scored = []
    for entry in APOLOGETICS_KNOWLEDGE:
        score = bonus  # start from trigger bonus (0 if no trigger matched)

        haystack = " ".join([
            entry.get("title", ""),
            entry.get("summary", ""),
            entry.get("embedding_text", ""),
            " ".join(entry.get("tags", [])),
        ]).lower()

        for word in query_words:
            if word in haystack:
                score += 1
        for word in query_words:
            if word in [t.lower() for t in entry.get("tags", [])]:
                score += 2

        if score > 0:
            scored.append((score, entry))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_results]

    lines = ["[APOLOGETICS KNOWLEDGE BASE — relevant entries found]"]
    for _, entry in top:
        title   = entry.get("title", "Untitled")
        chapter = entry.get("chapter", "")
        summary = entry.get("summary", "").strip()
        etext   = entry.get("embedding_text", "").strip()
        source  = entry.get("source", "")

        detail = etext if etext else summary
        if len(detail) > 500:
            detail = detail[:500] + "…"

        lines.append(f"• [{chapter}] {title} ({source})")
        lines.append(f"  {detail}")

    lines.append("[End of apologetics knowledge — use this to inform your answer if relevant]")
    return "\n".join(lines)


# Load both knowledge bases at startup
load_heiser_knowledge()
load_apologetics_knowledge()

# =============================================================================
# MIDDLE ENGLISH DICTIONARY — SQLite-backed for instant Rumble chat lookups
# =============================================================================
#
# The raw JSONL file (kaikki_org-dictionary-MiddleEnglish-words.jsonl) is 62 MB
# with 52,736 entries. Loading it all into memory like the Heiser knowledge base
# would be slow and wasteful. Instead, we build a small SQLite database once at
# startup (takes ~5–8 seconds the first time, then reuses the .db file forever).
# Lookups are then instant (<1 ms), which is critical for live Rumble chat.
#
# Schema:
#   words(word TEXT, alt_form TEXT, pos TEXT, glosses TEXT,
#         etymology TEXT, sounds TEXT)
#
#   - word       : the headword (e.g. "cat")
#   - alt_form   : alternative spellings and inflected forms (e.g. "catt|catte|kat")
#                  indexed separately so we can find entries via any spelling
#   - pos        : part of speech (noun, verb, adj, …)
#   - glosses    : pipe-joined list of all sense definitions
#   - etymology  : etymology_text (first 300 chars)
#   - sounds     : IPA pronunciation string(s), pipe-joined
#
# Two indexes are created: one on the canonical headword, one on alt_form
# so queries like "what does 'catte' mean?" still resolve correctly.
# =============================================================================

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ME_JSONL    = os.path.join(_BASE_DIR, "kaikki_org-dictionary-MiddleEnglish-words.jsonl")
ME_DB_PATH  = os.path.join(_BASE_DIR, "middle_english_dict.db")

# Module-level connection — opened once, reused for all lookups.
_me_db_conn: sqlite3.Connection | None = None


def _build_me_db(jsonl_path: str, db_path: str) -> None:
    """
    Parse the JSONL file and write a fresh SQLite database.
    Called only when the .db doesn't exist yet (or needs rebuilding).
    Runs in the background startup thread so the UI opens immediately.
    """
    print("[ME Dict] Building SQLite index — this takes ~5–10 seconds on first run…")
    start = time.time()

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS words;
        CREATE TABLE words (
            word      TEXT NOT NULL,
            alt_form  TEXT,
            pos       TEXT,
            glosses   TEXT,
            etymology TEXT,
            sounds    TEXT
        );
    """)

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue

            headword = e.get("word", "").strip().lower()
            if not headword:
                continue

            # Collect all alternative spellings and inflected forms
            alt_forms = []
            for form_obj in e.get("forms", []):
                form = form_obj.get("form", "").strip().lower()
                if form and form != headword:
                    alt_forms.append(form)

            # Flatten all sense glosses into a single readable string
            gloss_parts = []
            for sense in e.get("senses", []):
                g = sense.get("glosses", [])
                if g:
                    gloss_parts.append(g[-1])   # deepest / most specific gloss

            rows.append((
                headword,
                "|".join(alt_forms) if alt_forms else None,
                e.get("pos", ""),
                "; ".join(gloss_parts[:6]) if gloss_parts else None,   # cap at 6 senses
                (e.get("etymology_text") or "")[:300] or None,
                "|".join(
                    s.get("ipa", "") for s in e.get("sounds", []) if s.get("ipa")
                ) or None,
            ))

            # Batch inserts for speed
            if len(rows) >= 2000:
                cur.executemany(
                    "INSERT INTO words VALUES (?,?,?,?,?,?)", rows
                )
                rows.clear()

    if rows:
        cur.executemany("INSERT INTO words VALUES (?,?,?,?,?,?)", rows)

    # Indexes: headword lookup + alt-form lookup (for variant spellings)
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_word     ON words(word);
        CREATE INDEX IF NOT EXISTS idx_alt_form ON words(alt_form);
    """)

    conn.commit()
    conn.close()
    elapsed = time.time() - start
    print(f"[ME Dict] SQLite index built in {elapsed:.1f}s — {db_path}")


def _init_me_db() -> None:
    """
    Called once at startup (in a background thread).
    Builds the DB if missing, then opens the shared connection.
    """
    global _me_db_conn

    if not os.path.exists(ME_JSONL):
        print(f"[ME Dict] JSONL not found at {ME_JSONL} — Middle English lookup disabled.")
        return

    # Build the index if the .db doesn't exist yet
    if not os.path.exists(ME_DB_PATH):
        try:
            _build_me_db(ME_JSONL, ME_DB_PATH)
        except Exception as ex:
            print(f"[ME Dict] Failed to build index: {ex}")
            return

    try:
        _me_db_conn = sqlite3.connect(ME_DB_PATH, check_same_thread=False)
        # WAL mode: allows concurrent reads without blocking the UI thread
        _me_db_conn.execute("PRAGMA journal_mode=WAL")
        print("[ME Dict] Ready.")
    except Exception as ex:
        print(f"[ME Dict] Could not open database: {ex}")


def lookup_middle_english(word: str, max_entries: int = 4) -> str:
    """
    Look up a word (or variant spelling) in the Middle English SQLite database.
    Returns a compact, Rumble-friendly summary string, or "" if nothing found.

    The function checks:
      1. Exact headword match  (e.g. "soule" → soule)
      2. Alt-form match        (e.g. "sowle" → alt_form of soule)
    Results are deduplicated and capped at max_entries to keep context short.
    """
    if _me_db_conn is None:
        return ""

    word_clean = word.strip().lower()
    if not word_clean:
        return ""

    try:
        cur = _me_db_conn.cursor()

        # 1. Exact headword match
        cur.execute(
            "SELECT word, pos, glosses, etymology, sounds FROM words "
            "WHERE word = ? LIMIT ?",
            (word_clean, max_entries)
        )
        rows = cur.fetchall()

        # 2. Alt-form match (variant spellings / inflections) — only if headword missed.
        #
        # alt_form is stored pipe-separated (e.g. "catt|catte|kat").  A plain LIKE
        # '%word%' over-matches — "cat" would hit "catter", "scat", etc.  Instead we
        # bracket the column with sentinel pipes so every token is surrounded by '|',
        # then search for '|word|'.  This is an exact whole-token match entirely inside
        # SQLite — no Python post-filter or per-row sub-query needed.
        #
        #   stored   : "catt|catte|kat"
        #   bracketed: "|catt|catte|kat|"
        #   pattern  : "|cat|"  → no match  (correct — "cat" is not in that list)
        #   pattern  : "|catt|" → match     (correct)
        if not rows:
            cur.execute(
                "SELECT word, pos, glosses, etymology, sounds FROM words "
                "WHERE instr('|' || alt_form || '|', ?) > 0 LIMIT ?",
                (f"|{word_clean}|", max_entries),
            )
            rows = cur.fetchall()

        if not rows:
            return ""

        lines = [f"[MIDDLE ENGLISH DICT — '{word}']"]
        seen_headwords = set()
        for headword, pos, glosses, etymology, sounds in rows:
            if headword in seen_headwords:
                continue
            seen_headwords.add(headword)

            parts = [f"• {headword}"]
            if pos:
                parts[0] += f" ({pos})"
            if glosses:
                # Keep it concise for chat context
                gloss_short = glosses if len(glosses) <= 200 else glosses[:197] + "…"
                parts.append(f"  Meaning: {gloss_short}")
            if etymology:
                etym_short = etymology if len(etymology) <= 150 else etymology[:147] + "…"
                parts.append(f"  Origin: {etym_short}")
            if sounds:
                parts.append(f"  Pronunciation: /{sounds.split('|')[0]}/")
            lines.append("\n".join(parts))

        lines.append("[End of Middle English data — use this to inform your answer]")
        return "\n".join(lines)

    except Exception as ex:
        print(f"[ME Dict] Lookup error: {ex}")
        return ""


def _detect_me_query(question: str) -> str | None:
    """
    Heuristic: detect whether the user is asking about a Middle English word.
    Returns the candidate word/phrase to look up, or None.

    Triggers on patterns like:
      - "what does 'soule' mean in Middle English"
      - "middle english word X"
      - "what is the middle english for X"
      - "translate X from middle english"
      - explicit quote markers: "what does 'X' mean"
    """
    q = question.lower()

    # Explicit Middle English mention + nearby word
    me_pattern = re.compile(
        r'(?:middle\s+english|middle-english|me\s+word)\s+(?:word\s+)?["\']?(\w[\w\s]{0,30}?)["\']?'
        r'|["\'](\w[\w\s]{0,30}?)["\'].*middle\s+english'
        r'|middle\s+english\s+for\s+["\']?(\w[\w\s]{0,30}?)["\']?'
        r'|translate\s+["\']?(\w[\w\s]{0,30}?)["\']?\s+(?:from\s+)?middle\s+english',
        re.IGNORECASE,
    )
    m = me_pattern.search(question)
    if m:
        candidate = next((g for g in m.groups() if g), None)
        if candidate:
            return candidate.strip().split()[0]  # take first token of multi-word

    # "what does 'X' mean" — quoted word, no explicit ME mention
    quote_pattern = re.compile(r"""what\s+does\s+['"\u2018\u2019\u201c\u201d](\w+)['"\u2018\u2019\u201c\u201d]\s+mean""", re.IGNORECASE)
    m2 = quote_pattern.search(question)
    if m2:
        return m2.group(1).strip()

    return None


# Kick off DB init in a daemon thread so the UI opens instantly
threading.Thread(target=_init_me_db, daemon=True).start()

# -----------------------------
# BOOK NORMALIZATION
# -----------------------------
BOOK_NORMALIZATION = {
    # ── Genesis ──
    "Genesis": "Genesis", "Gen": "Genesis", "Ge": "Genesis", "Gn": "Genesis",
    # ── Exodus ──
    "Exodus": "Exodus", "Ex": "Exodus", "Exo": "Exodus", "Exod": "Exodus",
    # ── Leviticus ──
    "Leviticus": "Leviticus", "Lev": "Leviticus", "Le": "Leviticus", "Lv": "Leviticus",
    # ── Numbers ──
    "Numbers": "Numbers", "Num": "Numbers", "Nu": "Numbers", "Nm": "Numbers", "Numb": "Numbers",
    # ── Deuteronomy ──
    "Deuteronomy": "Deuteronomy", "Deut": "Deuteronomy", "Dt": "Deuteronomy", "Deu": "Deuteronomy",
    # ── Joshua ──
    "Joshua": "Joshua", "Josh": "Joshua", "Jos": "Joshua",
    # ── Judges ──
    "Judges": "Judges", "Judg": "Judges", "Jdg": "Judges", "Jg": "Judges",
    # ── Ruth ──
    "Ruth": "Ruth", "Rth": "Ruth",
    # ── 1 Samuel ──
    "1 Samuel": "1 Samuel", "1Sam": "1 Samuel", "1 Sam": "1 Samuel",
    "I Samuel": "1 Samuel", "1Samuel": "1 Samuel",
    # ── 2 Samuel ──
    "2 Samuel": "2 Samuel", "2Sam": "2 Samuel", "2 Sam": "2 Samuel",
    "II Samuel": "2 Samuel", "2Samuel": "2 Samuel",
    # ── 1 Kings ──
    "1 Kings": "1 Kings", "1Kgs": "1 Kings", "1 Kgs": "1 Kings",
    "I Kings": "1 Kings", "1Kings": "1 Kings", "1Kin": "1 Kings",
    # ── 2 Kings ──
    "2 Kings": "2 Kings", "2Kgs": "2 Kings", "2 Kgs": "2 Kings",
    "II Kings": "2 Kings", "2Kings": "2 Kings", "2Kin": "2 Kings",
    # ── 1 Chronicles ──
    "1 Chronicles": "1 Chronicles", "1Chr": "1 Chronicles", "1 Chr": "1 Chronicles",
    "I Chronicles": "1 Chronicles", "1Chron": "1 Chronicles", "1 Chron": "1 Chronicles",
    # ── 2 Chronicles ──
    "2 Chronicles": "2 Chronicles", "2Chr": "2 Chronicles", "2 Chr": "2 Chronicles",
    "II Chronicles": "2 Chronicles", "2Chron": "2 Chronicles", "2 Chron": "2 Chronicles",
    # ── Ezra ──
    "Ezra": "Ezra",
    # ── Nehemiah ──
    "Nehemiah": "Nehemiah", "Neh": "Nehemiah", "Ne": "Nehemiah",
    # ── Esther ──
    "Esther": "Esther", "Esth": "Esther", "Est": "Esther",
    # ── Job ──
    "Job": "Job",
    # ── Psalms ──
    "Psalms": "Psalms", "Psalm": "Psalms", "Ps": "Psalms", "Psa": "Psalms", "Pss": "Psalms",
    # ── Proverbs ──
    "Proverbs": "Proverbs", "Prov": "Proverbs", "Pro": "Proverbs", "Prv": "Proverbs",
    # ── Ecclesiastes ──
    "Ecclesiastes": "Ecclesiastes", "Eccl": "Ecclesiastes", "Ecc": "Ecclesiastes", "Ec": "Ecclesiastes",
    # ── Song of Solomon ──
    "Song of Solomon": "Song of Solomon", "Song": "Song of Solomon",
    "Song of Songs": "Song of Solomon", "SOS": "Song of Solomon",
    "Sos": "Song of Solomon", "SS": "Song of Solomon",
    "Solomon's Song": "Song of Solomon", "Cant": "Song of Solomon",
    # ── Isaiah ──
    "Isaiah": "Isaiah", "Isa": "Isaiah", "Is": "Isaiah",
    # ── Jeremiah ──
    "Jeremiah": "Jeremiah", "Jer": "Jeremiah", "Je": "Jeremiah",
    # ── Lamentations ──
    "Lamentations": "Lamentations", "Lam": "Lamentations", "La": "Lamentations",
    # ── Ezekiel ──
    "Ezekiel": "Ezekiel", "Ezek": "Ezekiel", "Eze": "Ezekiel", "Ezk": "Ezekiel",
    # ── Daniel ──
    "Daniel": "Daniel", "Dan": "Daniel", "Da": "Daniel", "Dn": "Daniel",
    # ── Hosea ──
    "Hosea": "Hosea", "Hos": "Hosea", "Ho": "Hosea",
    # ── Joel ──
    "Joel": "Joel", "Joe": "Joel", "Jl": "Joel",
    # ── Amos ──
    "Amos": "Amos", "Am": "Amos",
    # ── Obadiah ──
    "Obadiah": "Obadiah", "Obad": "Obadiah", "Ob": "Obadiah",
    # ── Jonah ──
    "Jonah": "Jonah", "Jon": "Jonah", "Jnh": "Jonah",
    # ── Micah ──
    "Micah": "Micah", "Mic": "Micah", "Mc": "Micah",
    # ── Nahum ──
    "Nahum": "Nahum", "Nah": "Nahum", "Na": "Nahum",
    # ── Habakkuk ──
    "Habakkuk": "Habakkuk", "Hab": "Habakkuk", "Hb": "Habakkuk",
    # ── Zephaniah ──
    "Zephaniah": "Zephaniah", "Zeph": "Zephaniah", "Zep": "Zephaniah", "Zp": "Zephaniah",
    # ── Haggai ──
    "Haggai": "Haggai", "Hag": "Haggai", "Hg": "Haggai",
    # ── Zechariah ──
    "Zechariah": "Zechariah", "Zech": "Zechariah", "Zec": "Zechariah", "Zc": "Zechariah",
    # ── Malachi ──
    "Malachi": "Malachi", "Mal": "Malachi", "Ml": "Malachi",
    # ── Matthew (+ common misspellings) ──
    "Matthew": "Matthew", "Matt": "Matthew", "Mat": "Matthew", "Mt": "Matthew",
    "Mathew": "Matthew", "Mathew": "Matthew", "Matth": "Matthew",
    # ── Mark ──
    "Mark": "Mark", "Mrk": "Mark", "Mk": "Mark",
    # ── Luke ──
    "Luke": "Luke", "Luk": "Luke", "Lk": "Luke",
    # ── John ──
    "John": "John", "Jhn": "John", "Jn": "John",
    # ── Acts ──
    "Acts": "Acts", "Act": "Acts", "Ac": "Acts",
    # ── Romans ──
    "Romans": "Romans", "Rom": "Romans", "Ro": "Romans", "Rm": "Romans",
    # ── 1 Corinthians ──
    "1 Corinthians": "1 Corinthians", "1Cor": "1 Corinthians", "1 Cor": "1 Corinthians",
    "I Corinthians": "1 Corinthians", "1Corinthians": "1 Corinthians",
    # ── 2 Corinthians ──
    "2 Corinthians": "2 Corinthians", "2Cor": "2 Corinthians", "2 Cor": "2 Corinthians",
    "II Corinthians": "2 Corinthians", "2Corinthians": "2 Corinthians",
    # ── Galatians ──
    "Galatians": "Galatians", "Gal": "Galatians", "Ga": "Galatians",
    # ── Ephesians ──
    "Ephesians": "Ephesians", "Eph": "Ephesians", "Ep": "Ephesians",
    # ── Philippians ──
    "Philippians": "Philippians", "Phil": "Philippians", "Php": "Philippians", "Pp": "Philippians",
    # ── Colossians ──
    "Colossians": "Colossians", "Col": "Colossians",
    # ── 1 Thessalonians ──
    "1 Thessalonians": "1 Thessalonians", "1Thess": "1 Thessalonians", "1 Thess": "1 Thessalonians",
    "I Thessalonians": "1 Thessalonians", "1Th": "1 Thessalonians", "1 Th": "1 Thessalonians",
    # ── 2 Thessalonians ──
    "2 Thessalonians": "2 Thessalonians", "2Thess": "2 Thessalonians", "2 Thess": "2 Thessalonians",
    "II Thessalonians": "2 Thessalonians", "2Th": "2 Thessalonians", "2 Th": "2 Thessalonians",
    # ── 1 Timothy ──
    "1 Timothy": "1 Timothy", "1Tim": "1 Timothy", "1 Tim": "1 Timothy",
    "I Timothy": "1 Timothy", "1Ti": "1 Timothy",
    # ── 2 Timothy ──
    "2 Timothy": "2 Timothy", "2Tim": "2 Timothy", "2 Tim": "2 Timothy",
    "II Timothy": "2 Timothy", "2Ti": "2 Timothy",
    # ── Titus ──
    "Titus": "Titus", "Tit": "Titus", "Ti": "Titus",
    # ── Philemon ──
    "Philemon": "Philemon", "Philem": "Philemon", "Phm": "Philemon", "Pm": "Philemon",
    # ── Hebrews ──
    "Hebrews": "Hebrews", "Heb": "Hebrews", "He": "Hebrews",
    # ── James ──
    "James": "James", "Jas": "James", "Jm": "James",
    # ── 1 Peter ──
    "1 Peter": "1 Peter", "1Pet": "1 Peter", "1 Pet": "1 Peter",
    "I Peter": "1 Peter", "1Pe": "1 Peter", "1Pt": "1 Peter",
    # ── 2 Peter ──
    "2 Peter": "2 Peter", "2Pet": "2 Peter", "2 Pet": "2 Peter",
    "II Peter": "2 Peter", "2Pe": "2 Peter", "2Pt": "2 Peter",
    # ── 1 John ──
    "1 John": "1 John", "1Jn": "1 John", "1 Jn": "1 John",
    "I John": "1 John", "1Jo": "1 John", "1Jhn": "1 John",
    # ── 2 John ──
    "2 John": "2 John", "2Jn": "2 John", "2 Jn": "2 John",
    "II John": "2 John", "2Jo": "2 John",
    # ── 3 John ──
    "3 John": "3 John", "3Jn": "3 John", "3 Jn": "3 John",
    "III John": "3 John", "3Jo": "3 John",
    # ── Jude ──
    "Jude": "Jude", "Jud": "Jude",
    # ── Revelation (+ common misspellings) ──
    "Revelation": "Revelation", "Rev": "Revelation", "Re": "Revelation",
    "Revelations": "Revelation", "Rv": "Revelation",
}

# -----------------------------
# PERSISTENCE
# -----------------------------
SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.json")

def load_saved_conversations():
    if not os.path.exists(SAVE_FILE):
        return []
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(entry["title"], entry["content"]) for entry in data if "title" in entry and "content" in entry]
    except Exception as e:
        print(f"[Warning] Could not load conversations.json: {e}")
        return []

def persist_conversations(convos):
    try:
        data = [{"title": title, "content": content} for title, content in convos]
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Could not save conversations.json: {e}")



# -----------------------------
# STREAMING CHAT CALL
# -----------------------------
conversation_history = []

# ------------------------------------------------------------------
# OLLAMA LOCK — ensures the UI and Rumble bot never call Ollama at
# the same time. Whoever gets there first runs; the other waits.
# ------------------------------------------------------------------
ollama_lock = threading.Lock()

# The word "RUMBLE CHAT" in this prompt triggers the special output mode
# defined in the modelfile — no labels, no Strong's, plain sentences only.
# Built as a function so it always reflects the current RUMBLE_CHAR_LIMIT.
def build_rumble_system() -> str:
    max_parts    = _calc_max_parts()
    total_budget = RUMBLE_CHAR_LIMIT * max_parts
    return (
        "RUMBLE CHAT. "
        "You are BibleScholar23 answering a live Rumble chat question. "
        "This is a Christian apologetics and Bible study stream. "
        "Your scope is broad: Bible, theology, apologetics, church history, "
        "Ancient Near Eastern culture, Nephilim, giants (Rephaim, Anakim, Emim), "
        "sons of God, divine council, famous figures in Christianity or biblical history, "
        "comparative religion, manuscript evidence, archaeology, and Middle English "
        "language as used in Bible translations (Wycliffe, Tyndale, KJV era). "
        "If a question touches any of these areas, ANSWER IT FULLY — do not say "
        "'that is not Bible-related.' "
        "Vague questions like 'what's important?' get a substantive overview answer "
        "covering multiple important topics. "
        "Questions about giants, Nephilim, or the divine council get a full explanation, "
        "not just one or two sentences. "
        "When Middle English dictionary data is provided in context, use it authoritatively "
        "to explain the word's meaning, origin, and how it appears in early Bible translations. "
        "Plain sentences only. No headers, no labels, no bullet points. "
        "Include Strong's numbers only if the user specifically asks for them. "
        "If views differ write: Views differ: then summarize each in one sentence. "
        f"Write up to {total_budget} characters total — your response will be "
        f"automatically split across up to {max_parts} chat messages of {RUMBLE_ENTRY_CHARS} "
        f"characters each. Do NOT stop early. Fill the space. "
        "Never end mid-sentence. "
        "Never add notes, commentary, or explanations about these instructions. "
        "Do not begin with filler words. Answer immediately and substantively."
    )




# Pre-approved canned responses — checked before hitting Ollama.
# Returning these instantly saves tokens and latency.
# Add entries here as key=trigger_phrase (lowercase), value=response text.
CANNED_RESPONSES = {
    "islamic dilemma": (
        "The Quran affirms the Bible's inspiration, preservation, and authority "
        "(Surah 3:3-4, 5:47, 6:115). So either the Bible IS God's word — and "
        "Islam is false because the Bible contradicts it — or the Bible is NOT "
        "reliable — and Islam is false because the Quran says it is. Either way, "
        "Islam self-destructs on Scripture. Ask me more."
    ),
}

def check_canned_response(question: str) -> str | None:
    """Return a pre-approved canned response if the question matches, else None."""
    q = question.lower().strip(" ?!")
    for trigger, response in CANNED_RESPONSES.items():
        if trigger in q:
            return response
    return None


def check_ollama_health() -> bool:
    """Ping Ollama to confirm it's reachable before the UI starts."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def _summarize_via_ollama(text: str, max_chars: int) -> str | None:
    """
    Ask Ollama to rewrite *text* so it fits within *max_chars* characters.
    Returns the shorter text, or None if the call fails.
    Called only when sanitize_rumble_response finds the text is too long.
    """
    prompt = (
        f"Rewrite the following answer so it fits within {max_chars} characters total. "
        "Keep the most important biblical facts. Plain sentences only. "
        "No headers, no bullet points. End at a complete sentence. "
        f"Output ONLY the rewritten text — nothing else.\n\n{text}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": RUMBLE_NUM_CTX,
            "num_predict": _calc_num_predict(max_chars),
        },
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json().get("message", {}).get("content", "").strip()
        # Strip any stray non-ASCII/CJK artifacts before returning
        result = _strip_non_latin(result)
        return result if result else None
    except Exception as e:
        rumble_log(f"[Summarize error] {e}")
        return None


def _strip_non_latin(text: str) -> str:
    """
    Remove runs of CJK / non-Latin characters that occasionally leak into
    Ollama output (encoding artefacts, stray BOM bytes, tokenizer noise).
    Keeps standard Latin, Greek letters (for transliterations), punctuation,
    digits, and common Unicode used in Bible scholarship (e.g. dagesh dots).
    """
    # Remove CJK Unified Ideographs, CJK Extension blocks, Hangul, Hiragana,
    # Katakana, and other East-Asian script ranges.
    cjk_pattern = re.compile(
        r'[\u2E80-\u2EFF'   # CJK Radicals Supplement
        r'\u2F00-\u2FDF'   # Kangxi Radicals
        r'\u3000-\u303F'   # CJK Symbols and Punctuation
        r'\u3040-\u309F'   # Hiragana
        r'\u30A0-\u30FF'   # Katakana
        r'\u3100-\u312F'   # Bopomofo
        r'\u3130-\u318F'   # Hangul Compatibility Jamo
        r'\u3190-\u319F'   # Kanbun
        r'\u31A0-\u31BF'   # Bopomofo Extended
        r'\u31F0-\u31FF'   # Katakana Phonetic Extensions
        r'\u3200-\u32FF'   # Enclosed CJK Letters and Months
        r'\u3300-\u33FF'   # CJK Compatibility
        r'\u3400-\u4DBF'   # CJK Extension A
        r'\u4E00-\u9FFF'   # CJK Unified Ideographs (core)
        r'\uA000-\uA48F'   # Yi Syllables
        r'\uA490-\uA4CF'   # Yi Radicals
        r'\uAC00-\uD7AF'   # Hangul Syllables
        r'\uF900-\uFAFF'   # CJK Compatibility Ideographs
        r'\uFE10-\uFE1F'   # Vertical forms
        r'\uFE30-\uFE4F'   # CJK Compatibility Forms
        r'\uFF00-\uFFEF'   # Halfwidth and Fullwidth Forms
        r'\U00020000-\U0002A6DF'  # CJK Extension B
        r'\U0002A700-\U0002B73F'  # CJK Extension C
        r'\U0002B740-\U0002B81F'  # CJK Extension D
        r']+',
        re.UNICODE
    )
    cleaned = cjk_pattern.sub('', text)
    # Collapse any double-spaces left behind
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned


def sanitize_rumble_response(text: str, max_chars: int = 780, max_parts: int = 3) -> str:
    """
    Enforce a hard character ceiling on Rumble responses.

    The ceiling is max_chars * max_parts — the full budget across all chat
    messages — NOT just one message. split_into_chat_entries() handles
    dividing the result into per-message chunks afterward.

    1. Strip CJK / non-Latin garbage characters.
    2. Strip parenthetical meta-commentary.
    3. If already fits in the total budget, return as-is.
    4. If too long, ask Ollama to summarize to fit.
    5. Fallback: truncate at last complete sentence within budget.
    """
    total_budget = max_chars * max_parts

    # Step 1 — strip CJK / garbage characters
    text = _strip_non_latin(text)

    # Step 2 — strip parenthetical meta-commentary
    text = re.sub(
        r'\s*\([^)]*(?:instruct|Note:|asked|request)[^)]*\)',
        '', text, flags=re.IGNORECASE
    ).strip()

    # Step 3 — already fits total budget
    if len(text) <= total_budget:
        return text

    # Step 4 — ask Ollama to summarize down to total budget
    rumble_log(f"Response too long ({len(text)} chars > {total_budget} total budget) — asking Ollama to summarize...")
    summarized = _summarize_via_ollama(text, total_budget)
    if summarized and len(summarized) <= total_budget:
        rumble_log(f"Summarized to {len(summarized)} chars.")
        return summarized

    # Step 5 — fallback: trim to last complete sentence within total budget
    source = summarized if summarized else text
    clipped = source[:total_budget]
    last_end = max(clipped.rfind('.'), clipped.rfind('!'), clipped.rfind('?'))
    if last_end != -1:
        return clipped[:last_end + 1].strip()
    return clipped.strip()

def stream_from_ollama(user_message):
    """Streaming call used by the UI. Tries to acquire the shared ollama_lock
    with a 30-second timeout so a busy Rumble request doesn't permanently
    block the UI. Yields text chunks as they arrive."""
    # Trim conversation history to prevent context overflow
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    # Store the user message in history and build the request
    conversation_history.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL,
        "messages": conversation_history,
        "stream": True,
    }

    ai_reply_chunks = []

    acquired = ollama_lock.acquire(timeout=30)
    if not acquired:
        yield "[Ollama is busy — please try again in a moment.]"
        return

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                chunk = data.get("message", {}).get("content", "")

                if chunk:
                    ai_reply_chunks.append(chunk)
                    yield chunk

                if data.get("done"):
                    full_reply = "".join(ai_reply_chunks)
                    conversation_history.append({"role": "assistant", "content": full_reply})
                    break

    except Exception as e:
        yield f"[Error communicating with model: {e}]"
    finally:
        ollama_lock.release()


def split_into_chat_entries(text, max_chars=200, max_parts=3):
    """
    Split text into at most max_parts chunks, each <= max_chars characters,
    breaking only at word boundaries. Overflow beyond max_parts is logged and dropped.
    """
    parts = []
    remaining = text.strip()

    while remaining and len(parts) < max_parts:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break

        cut = remaining.rfind(" ", 0, max_chars)
        if cut == -1:
            # No space before the limit -- scan forward for the next space so we
            # don't hard-cut mid-word. Fall back to hard cut only if no space exists.
            next_space = remaining.find(" ", max_chars)
            cut = next_space if next_space != -1 else max_chars

        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining and len(parts) >= max_parts:
        rumble_log(
            f"[Warning] Unexpected overflow: {len(remaining)} chars dropped after {max_parts} parts. "
            f"(sanitize_rumble_response should have prevented this — check budget alignment.)"
        )

    return parts


def inject_strongs_facts(user_message: str) -> str:
    """
    Look up any Strong's numbers in the question from STRONGS_LOOKUP
    and prepend verified definitions so the model cannot hallucinate.
    Also injects Middle English dictionary data when the question asks
    about a Middle English word.
    """
    pattern = re.compile(
        r'\b([HGhg]\d{1,5}[a-z]?)\b'
        r'|(?:strong\'?s?\s*#?\s*)(\d{3,5})\b'
        r'|\bstrong\'?s?\s+number\s+(\d{3,5})\b',
        re.IGNORECASE
    )
    injections = []
    seen = set()
    for m in pattern.finditer(user_message):
        raw = m.group(1) or m.group(2) or m.group(3)
        if not raw:
            continue
        key = normalize_strongs_key(raw)
        if key in seen:
            continue
        seen.add(key)
        entry = STRONGS_LOOKUP.get(key)
        lang = "Hebrew" if key.startswith("H") else "Greek"
        if entry:
            if isinstance(entry, dict):
                lemma = entry.get("lemma", "")
                xlit  = entry.get("xlit") or entry.get("translit", "")
                pron  = entry.get("pron", "")
                sdef  = entry.get("strongs_def", "").strip()
                kjv   = entry.get("kjv_def", "").strip()
                line  = (f"[VERIFIED] {key} ({lang}): lemma={lemma} "
                         f"xlit={xlit} pron={pron} — {sdef} [KJV gloss: {kjv}]")
            else:
                line = f"[VERIFIED] {key} ({lang}): {entry}"
            injections.append(line)
        else:
            injections.append(
                f"[VERIFIED] {key}: NOT FOUND in Strong's lexicon. "
                f"Do not invent a definition. Tell the user this number has no entry."
            )

    # ── Middle English dictionary injection ──────────────────────────────────
    # Detect whether the question is about a Middle English word and, if so,
    # look it up in the SQLite index and prepend the verified data.
    me_block = ""
    me_candidate = _detect_me_query(user_message)
    if me_candidate:
        me_block = lookup_middle_english(me_candidate)
        if me_block:
            print(f"[ME Dict] Injecting data for '{me_candidate}'")

    # ── Heiser knowledge base context (Divine Council / ANE / exegetical) ────
    heiser_block = search_heiser_knowledge(user_message)

    # ── Apologetics knowledge base context (Lewis, Geisler/Turek, etc.) ──────
    apologetics_block = search_apologetics_knowledge(user_message)

    # ── Assemble enriched prompt ─────────────────────────────────────────────
    if injections:
        block = "\n".join(injections)
        enriched = (
            f"VERIFIED STRONG'S DATA (use only this — do not rely on memory):\n"
            f"{block}\n\n{user_message}"
        )
    else:
        enriched = user_message

    if me_block:
        enriched = f"{me_block}\n\n{enriched}"

    if heiser_block:
        enriched = f"{heiser_block}\n\n{enriched}"

    if apologetics_block:
        enriched = f"{apologetics_block}\n\n{enriched}"

    return enriched


# -----------------------------
# OLLAMA SYNC CALL (RUMBLE BOT)
# -----------------------------
def get_ollama_response_sync(user_message):
    """
    Non-streaming Ollama call used by the Rumble bot.
    Acquires the shared ollama_lock so it waits politely if the UI
    is already mid-conversation, then runs immediately when free.
    """
    grounded = inject_strongs_facts(user_message)
    rumble_prefixed = f"{build_rumble_system()}\n\n{grounded}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": rumble_prefixed}],
        "stream": False,
        # RUMBLE_NUM_CTX is set to 8192 to match the Modelfile.
        # RUMBLE_NUM_PREDICT caps token output.
        "options": {
            "num_ctx": RUMBLE_NUM_CTX,
            "num_predict": _calc_num_predict(RUMBLE_CHAR_LIMIT),
        },
    }
    rumble_log("Waiting for Ollama to be free…")
    with ollama_lock:
        rumble_log("Querying AI…")
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.HTTPError as e:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            rumble_log(f"[Ollama error]: {e} | Detail: {detail}")
            return None
        except Exception as e:
            rumble_log(f"[Ollama error]: {e}")
            return None


# -----------------------------
# REGEX PATTERNS
# -----------------------------
VERSE_PATTERN = re.compile(
    r'\b'
    r'(?:'
        r'Song\s+of\s+Solomon'  # 3-word book — must come before shorter matches
        r'|Song\s+of\s+Songs'
        r'|[1-3]\s?[A-Za-z]+'  # numbered books: 1 Kings, 2 Cor, 1 Corinthians, etc.
        r'|[A-Za-z]+'           # single-word books: Genesis, Psalms, Rev, etc.
    r')'
    r'\s+\d+:\d+(?:-\d+)?'
    r'\b',
    re.IGNORECASE
)

STRONGS_PATTERN = re.compile(
    r'(?:'
    r'\[STRONGS:\s*([HG]\d+[a-z]?)\]'
    r'|'
    r'\(([HG]\d+[a-z]?)\)'
    r'|'
    r'\b([HG]\d{1,5}[a-z]?)\b'
    r')'
)

# -----------------------------
# HIGHLIGHTING
# -----------------------------
def highlight_verses(text_widget, start_index="1.0"):
    text_widget.tag_remove("verse", start_index, tk.END)
    text = text_widget.get(start_index, tk.END)

    for match in VERSE_PATTERN.finditer(text):
        start = f"{start_index}+{match.start()}c"
        end = f"{start_index}+{match.end()}c"
        text_widget.tag_add("verse", start, end)

def highlight_strongs(text_widget, start_index="1.0"):
    text_widget.tag_remove("strongs", start_index, tk.END)
    text = text_widget.get(start_index, tk.END)

    for match in STRONGS_PATTERN.finditer(text):
        raw = match.group(1) or match.group(2) or match.group(3)
        if not raw:
            continue
        num_start = text.find(raw, match.start())
        if num_start == -1:
            continue
        num_end = num_start + len(raw)
        tag_start = f"{start_index}+{num_start}c"
        tag_end = f"{start_index}+{num_end}c"
        text_widget.tag_add("strongs", tag_start, tag_end)

def _highlight_new_content(text_widget):
    """
    Highlight only content added since the last 'highlight_marker' tag.
    Falls back to full scan if marker is absent (e.g. on conversation load).
    """
    try:
        ranges = text_widget.tag_ranges("highlight_marker")
        start = str(ranges[-1]) if ranges else "1.0"
    except Exception:
        start = "1.0"
    highlight_verses(text_widget, start_index=start)
    highlight_strongs(text_widget, start_index=start)
    # Advance the marker to the current end
    text_widget.tag_remove("highlight_marker", "1.0", tk.END)
    text_widget.tag_add("highlight_marker", tk.END)

def copy_verses_to_clipboard():
    ranges = chat_window.tag_ranges("verse")
    if not ranges:
        messagebox.showinfo("Copy Verses", "No verse references detected.")
        return

    verses = []
    for i in range(0, len(ranges), 2):
        verses.append(chat_window.get(ranges[i], ranges[i + 1]))

    combined = "\n".join(sorted(set(verses)))
    root.clipboard_clear()
    root.clipboard_append(combined)
    messagebox.showinfo("Copy Verses", "Verse references copied.")

# -----------------------------
# REFERENCE EXTRACTION
# -----------------------------
def extract_references(ai_text):
    refs = []
    for match in VERSE_PATTERN.finditer(ai_text):
        raw = match.group(0).strip()
        # Parse out the book portion and expand it to its canonical name so that
        # abbreviations like "Song", "Ps", "Rev" become the full book name in
        # both the chat panel list and the NET Bible verse lookup.
        parsed = parse_reference(raw)
        if parsed:
            canonical_book = BOOK_NORMALIZATION.get(
                parsed["book"],
                BOOK_NORMALIZATION.get(parsed["book"].title(), parsed["book"])
            )
            verse_range = parsed["verses"]
            if len(verse_range) > 1:
                cv = f'{parsed["chapter"]}:{verse_range[0]}-{verse_range[-1]}'
            else:
                cv = f'{parsed["chapter"]}:{verse_range[0]}'
            refs.append(f'{canonical_book} {cv}')
        else:
            refs.append(raw)
    return refs

# -----------------------------
# STRONG'S EXTRACTION
# -----------------------------
def extract_strongs_numbers(ai_text):
    seen = set()
    ordered = []
    for match in STRONGS_PATTERN.finditer(ai_text):
        raw = match.group(1) or match.group(2) or match.group(3)
        if not raw:
            continue
        num = normalize_strongs_key(raw)
        if num not in seen:
            seen.add(num)
            ordered.append(num)
    return ordered

# -----------------------------
# RANGE-AWARE PARSER
# -----------------------------
def parse_reference(ref):
    ref = ref.strip()

    if " " not in ref or ":" not in ref:
        return None

    book_part, cv = ref.rsplit(" ", 1)

    if "-" in cv:
        chapter_str, verse_range = cv.split(":", 1)
        start_verse_str, end_verse_str = verse_range.split("-", 1)

        try:
            chapter = int(chapter_str)
            start_verse = int(start_verse_str)
            end_verse = int(end_verse_str)
        except ValueError:
            return None

        return {
            "book": book_part.strip(),
            "chapter": chapter,
            "verses": list(range(start_verse, end_verse + 1))
        }

    chapter_str, verse_str = cv.split(":", 1)

    try:
        chapter = int(chapter_str)
        verse = int(verse_str)
    except ValueError:
        return None

    return {
        "book": book_part.strip(),
        "chapter": chapter,
        "verses": [verse]
    }

# -----------------------------
# DISPLAY NET BIBLE VERSES
# -----------------------------
def display_net_verses(refs):
    verse_output.config(state="normal")
    verse_output.delete("1.0", tk.END)

    if not refs:
        verse_output.insert(tk.END, "No verse references detected.\n")
        verse_output.config(state="disabled")
        return

    for ref in refs:
        parsed = parse_reference(ref)

        if not parsed:
            verse_output.insert(tk.END, f"{ref} — [Could not parse reference]\n\n")
            continue

        book = BOOK_NORMALIZATION.get(parsed["book"], parsed["book"])
        chapter = str(parsed["chapter"])

        for verse in parsed["verses"]:
            text = NET_LOOKUP.get(book, {}).get(chapter, {}).get(str(verse))

            if text:
                verse_output.insert(tk.END, f"{book} {chapter}:{verse} — {text}\n\n")
            else:
                verse_output.insert(tk.END, f"{book} {chapter}:{verse} — [Verse not found in NET Bible]\n\n")

    verse_output.config(state="disabled")


# -----------------------------
# DISPLAY STRONG'S WORDS
# -----------------------------
def display_strongs_words(strongs_numbers):
    strongs_output.config(state="normal")
    strongs_output.delete("1.0", tk.END)

    if not strongs_numbers:
        strongs_output.insert(tk.END, "No Strong's numbers detected.\n")
        strongs_output.config(state="disabled")
        return

    for num in strongs_numbers:
        word = STRONGS_LOOKUP.get(num)
        lang = "Heb." if num.startswith("H") else "Grk."

        if word:
            strongs_output.insert(tk.END, f"{num}  ", "strongs_num")
            strongs_output.insert(tk.END, f"({lang})  ", "strongs_lang")
            strongs_output.insert(tk.END, f"{word}\n\n", "strongs_word")
        else:
            strongs_output.insert(tk.END, f"{num}  ", "strongs_num")
            strongs_output.insert(tk.END, "[Not found in Strong's]\n\n", "strongs_missing")

    strongs_output.config(state="disabled")


# -----------------------------
# AI STOP FLAG
# -----------------------------
# Set to True to abort the current AI stream mid-response.
# _stream_to_chat checks this on every queue poll and bails out if set.
_ai_stop_flag = {"stop": False}

def stop_ai_response():
    """Called by the AI Stop button — signals the streaming loop to abort."""
    _ai_stop_flag["stop"] = True


# -----------------------------
# SHARED STREAMING HELPER
# -----------------------------
def _stream_to_chat(prompt: str, on_done=None):
    """
    Shared helper: stream an Ollama response into the chat window.
    Spawns a worker thread, then polls the queue every 30 ms on the
    main thread so Tkinter stays responsive.

    on_done(ai_full_text) is called once the stream is complete.
    The AI Stop button sets _ai_stop_flag["stop"] = True to abort early.
    """
    # Reset stop flag at the start of every new request
    _ai_stop_flag["stop"] = False

    chat_window.config(state="normal")
    # Mark position just before the new bot response for incremental highlighting
    chat_window.tag_add("highlight_marker", tk.END)
    chat_window.insert(tk.END, "BibleScholar:\n ", "bot")
    chat_window.config(state="disabled")
    chat_window.see(tk.END)

    send_button.config(state="disabled")
    if 'ai_stop_button' in globals():
        ai_stop_button.config(state="normal")
    thinking_label.config(text="BibleScholar is thinking…")
    root.update_idletasks()

    q = Queue()
    ai_chunks = []

    def worker():
        for chunk in stream_from_ollama(prompt):
            if _ai_stop_flag["stop"]:
                break
            q.put(chunk)
        q.put(None)  # sentinel

    def _finish_stream(stopped: bool = False):
        """Shared teardown — runs whether we finished or were stopped."""
        thinking_label.config(text="[Stopped]" if stopped else "")
        chat_window.config(state="normal")
        chat_window.insert(tk.END, "\n\n")
        _highlight_new_content(chat_window)
        chat_window.config(state="disabled")
        send_button.config(state="normal")
        if 'ai_stop_button' in globals():
            ai_stop_button.config(state="disabled")
        ai_full = "".join(ai_chunks)
        if on_done and not stopped:
            on_done(ai_full)

    def process_queue():
        # Check stop flag first — drain remaining queue items and bail
        if _ai_stop_flag["stop"]:
            try:
                while True:
                    q.get_nowait()
            except Empty:
                pass
            _finish_stream(stopped=True)
            return

        try:
            while True:
                item = q.get_nowait()

                if item is None:
                    # Stream finished naturally
                    _finish_stream(stopped=False)
                    return

                ai_chunks.append(item)
                chat_window.config(state="normal")
                chat_window.insert(tk.END, item, "bot")
                chat_window.config(state="disabled")
                chat_window.see(tk.END)

        except Empty:
            pass

        root.after(30, process_queue)

    threading.Thread(target=worker, daemon=True).start()
    root.after(30, process_queue)


# -----------------------------
# CHAT LOGIC
# -----------------------------
conversations = load_saved_conversations()

def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return

    chat_window.config(state="normal")
    chat_window.insert(tk.END, f"\nYou: {user_input}\n", "user")
    chat_window.config(state="disabled")
    entry.delete(0, tk.END)

    def on_done(ai_full):
        refs = extract_references(ai_full)
        display_net_verses(refs)
        strongs_nums = extract_strongs_numbers(ai_full)
        display_strongs_words(strongs_nums)

    _stream_to_chat(user_input, on_done=on_done)


def save_conversation():
    content = chat_window.get("1.0", tk.END).strip()
    if not content:
        messagebox.showinfo("Save Conversation", "Nothing to save.")
        return

    title = simpledialog.askstring("Save Conversation", "Enter a title:")
    if not title:
        return

    conversations.append((title, content))
    convo_listbox.insert(tk.END, title)
    persist_conversations(conversations)


def delete_conversation():
    selection = convo_listbox.curselection()
    if not selection:
        return

    index = selection[0]
    title = conversations[index][0]
    if not messagebox.askyesno("Delete", f"Delete '{title}'?"):
        return

    conversations.pop(index)
    convo_listbox.delete(index)
    persist_conversations(conversations)


def load_conversation(event=None):
    selection = convo_listbox.curselection()
    if not selection:
        return

    index = selection[0]
    title, content = conversations[index]

    chat_window.config(state="normal")
    chat_window.delete("1.0", tk.END)
    chat_window.insert(tk.END, content)
    # Full scan needed here since we replaced all content
    highlight_verses(chat_window, "1.0")
    highlight_strongs(chat_window, "1.0")
    # Reset the incremental highlight marker
    chat_window.tag_remove("highlight_marker", "1.0", tk.END)
    chat_window.tag_add("highlight_marker", tk.END)
    chat_window.config(state="disabled")


def _clear_text_widget(widget):
    """Enable, clear, and re-disable a read-only ScrolledText widget."""
    widget.config(state="normal")
    widget.delete("1.0", tk.END)
    widget.config(state="disabled")


def new_conversation():
    conversation_history.clear()
    _clear_text_widget(chat_window)
    _clear_text_widget(verse_output)
    _clear_text_widget(strongs_output)



# -----------------------------
# VERSE LOOKUP (STREAMING)
# -----------------------------
def lookup_verse():
    ref = verse_entry.get().strip()
    if not ref:
        return

    chat_window.config(state="normal")
    chat_window.insert(tk.END, f"Verse lookup ({ref}):\n", "user")
    chat_window.config(state="disabled")
    chat_window.see(tk.END)
    verse_entry.delete(0, tk.END)

    # Try the local NET Bible JSON first — instant and accurate.
    # resolve_verse_from_net expects a clean reference with no extra words.
    verse_text = resolve_verse_from_net(ref)
    if verse_text is not None:
        # Hit — display directly without touching Ollama.
        chat_window.config(state="normal")
        chat_window.insert(tk.END, f"BibleScholar:\n {verse_text}\n\n", "bot")
        _highlight_new_content(chat_window)
        chat_window.config(state="disabled")
        chat_window.see(tk.END)
        # Show in the NET Bible panel too
        verse_output.config(state="normal")
        verse_output.delete("1.0", tk.END)
        verse_output.insert(tk.END, verse_text + "\n")
        verse_output.config(state="disabled")
        return

    # Miss (book not in JSON, or ref couldn't be parsed) — fall back to Ollama.
    prompt = (
        f"Quote the NET Bible (New English Translation) text of {ref} "
        f"word for word, exactly as written. "
        f"Give only the reference and the verse text — nothing else. "
        f"No commentary, no explanation, no paraphrasing. "
        f"End the reference with (NET)."
    )

    def on_done(ai_full):
        strongs_nums = extract_strongs_numbers(ai_full)
        display_strongs_words(strongs_nums)

    _stream_to_chat(prompt, on_done=on_done)


# =============================================================================
# VERSE-ONLY DETECTION & NET BIBLE LOOKUP FOR RUMBLE CHAT
# =============================================================================

# Pattern that matches a clean "Book Chapter:Verse" reference.
# Used by resolve_verse_from_net after "verse only" is stripped out.
_VERSE_REF_RE = re.compile(
    r'^\s*'
    r'(?P<book>[1-3]?\s?[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+'
    r'(?P<chapter>\d+):(?P<verse_start>\d+)(?:-(?P<verse_end>\d+))?\s*[.!?]?\s*$',
    re.IGNORECASE,
)


def _normalize_book(raw: str) -> str:
    """Return the canonical book name from BOOK_NORMALIZATION, or raw if unknown."""
    raw = raw.strip()
    for candidate in (raw, raw.title(), raw.capitalize()):
        if candidate in BOOK_NORMALIZATION:
            return BOOK_NORMALIZATION[candidate]
    collapsed = re.sub(r'\s+', ' ', raw).strip()
    for candidate in (collapsed, collapsed.title()):
        if candidate in BOOK_NORMALIZATION:
            return BOOK_NORMALIZATION[candidate]
    return raw


def resolve_verse_from_net(question: str):
    """
    Look up a verse reference in NET_LOOKUP and return formatted text.
    Called only after "verse only" has already been stripped from the message,
    so *question* should be a clean reference like "Genesis 1:2".
    Returns None if the reference doesn't match or isn't found in the NET data.
    Supports single verses ("Romans 11:11") and ranges ("1 Cor 13:4-7").
    """
    m = _VERSE_REF_RE.match(question)
    if not m:
        return None

    raw_book    = m.group("book")
    chapter     = int(m.group("chapter"))
    verse_start = int(m.group("verse_start"))
    verse_end   = int(m.group("verse_end")) if m.group("verse_end") else verse_start

    book      = _normalize_book(raw_book)
    book_data = NET_LOOKUP.get(book)
    if not book_data:
        return None

    ch_data = book_data.get(str(chapter))
    if not ch_data:
        return None

    collected = []
    for v in range(verse_start, verse_end + 1):
        text = ch_data.get(str(v))
        if text:
            collected.append(text.strip())

    if not collected:
        return None

    ref_label = (
        f"{book} {chapter}:{verse_start}"
        if verse_start == verse_end
        else f"{book} {chapter}:{verse_start}-{verse_end}"
    )
    return f"{ref_label} — {' '.join(collected)} (NET)"


# =============================================================================
# RUMBLE BOT — runs entirely in a background thread
# =============================================================================

# Shared state between the Rumble thread and the UI
rumble_state = {
    "driver": None,
    "running": False,
    "thread": None,
    "seen_ids": OrderedDict(),  # message ID -> timestamp; expired after 30 min TTL
    "log_queue": Queue(),   # messages to append to the Rumble log panel
}


def rumble_log(msg: str):
    """Thread-safe: queue a line for the Rumble log panel."""
    rumble_state["log_queue"].put(msg)


def rumble_login(driver, email: str, password: str) -> bool:
    """Navigate to Rumble login and sign in with email + password. Returns True on success."""
    try:
        rumble_log("Navigating to Rumble login…")
        driver.get("https://rumble.com/login")
        wait = WebDriverWait(driver, 20)

        # Rumble's login form uses an email field (type="email" or name="email")
        email_field = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,
             "input[type='email'], input[name='email'], input[name='username'], input[type='text']")
        ))
        email_field.clear()
        email_field.send_keys(email)

        pass_field = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        pass_field.clear()
        pass_field.send_keys(password)
        pass_field.send_keys(Keys.RETURN)

        # Wait until we're redirected away from /login
        wait.until(EC.url_changes("https://rumble.com/login"))
        time.sleep(2)
        rumble_log(f"Logged in as {email}.")
        return True
    except Exception as e:
        rumble_log(f"[Login error] {e}")
        return False


def rumble_navigate_to_stream(driver, stream_url: str):
    """Open the target stream URL."""
    rumble_log(f"Opening stream: {stream_url}")
    driver.get(stream_url)
    time.sleep(4)


def get_chat_messages(driver):
    """
    Return a list of dicts: {id, author, text}
    Rumble's live chat uses <li> items inside the chat container.

    IDs are synthesised from author + text only — NOT DOM position.

    Using DOM position was the source of a re-answering bug: after the bot
    sat idle for a while, Rumble evicts old messages and loads new ones,
    shifting every item's position in the list.  The same message would land
    at a new position, producing a brand-new ID that wasn't in seen_ids, so
    the bot would answer it again.

    Content-only IDs mean the same author+text pair is always the same ID
    regardless of where it sits in the DOM.  If a user genuinely asks the
    identical question twice in the same session the second copy will be
    silently skipped — that's an acceptable trade-off.  The 500-entry
    eviction window in the bot loop ensures very old IDs are eventually
    forgotten so a question asked hours later will still be answered.
    """
    messages = []
    try:
        all_items = driver.find_elements(
            By.CSS_SELECTOR,
            # Rumble chat list items — selector may need adjustment if Rumble updates its HTML
            "ul.chat-history--list li, .rumbles-vote li, .chat-history li"
        )
        items = all_items[-20:]  # only process the most recent visible messages
        for item in items:
            try:
                author_el = item.find_element(By.CSS_SELECTOR, ".chat-username, .username, [class*='username']")
                text_el   = item.find_element(By.CSS_SELECTOR, ".chat-message--body, .message-body, [class*='message']")
                author = author_el.text.strip()
                text   = text_el.text.strip()
                # Stable content-based ID — immune to DOM list shifts
                uid = f"{author}::{text}"
                messages.append({"id": uid, "author": author, "text": text})
            except Exception:
                pass
    except Exception as e:
        rumble_log(f"[Chat scrape error] {e}")
    return messages


def post_chat_reply(driver, reply_text: str, max_retries: int = 3) -> bool:
    """
    Type and submit a reply in the Rumble chat input box.

    Uses clipboard-paste (Ctrl+V) instead of JS value injection to avoid
    IME/composition events that can corrupt text into Chinese characters on
    React-controlled inputs.  Falls back to send_keys if pyperclip is absent.
    Retries with exponential backoff if throttled or the element is stale.
    """
    # Sanitize the reply one final time — catch any CJK that slipped through
    reply_text = _strip_non_latin(reply_text)

    for attempt in range(max_retries):
        try:
            wait = WebDriverWait(driver, 10)
            chat_input = wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR,
                 "textarea.chat-input, input.chat-input, [placeholder*='chat'], "
                 "[class*='chat-input'], textarea[name='message']")
            ))
            chat_input.click()

            # Clear existing content
            chat_input.send_keys(Keys.CONTROL, 'a')
            chat_input.send_keys(Keys.DELETE)

            # Prefer clipboard paste — bypasses React's synthetic IME/composition
            # events that cause JS-injected text to become garbled CJK characters.
            try:
                import pyperclip
                pyperclip.copy(reply_text)
                chat_input.send_keys(Keys.CONTROL, 'v')
                time.sleep(1.2)   # increased from 0.2s — gives Rumble time to register input
            except ImportError:
                # pyperclip not installed — fall back to send_keys
                # (slower but avoids the JS injection problem)
                chat_input.send_keys(reply_text)

            # Verify the value looks right before submitting.
            # Rumble uses a contenteditable div in some layouts, so check
            # both .value (input/textarea) and innerText (contenteditable).
            actual = (
                chat_input.get_attribute("value")
                or chat_input.get_attribute("innerText")
                or driver.execute_script("return arguments[0].innerText;", chat_input)
                or ""
            )
            if actual and _strip_non_latin(actual) != actual.strip():
                # CJK crept in — clear and retry
                chat_input.send_keys(Keys.CONTROL, 'a')
                chat_input.send_keys(Keys.DELETE)
                raise RuntimeError("CJK contamination detected in chat input — retrying")

            chat_input.send_keys(Keys.RETURN)
            time.sleep(1.5)   # increased from 0.5s — avoids Rumble rate-limit kick
            return True
        except Exception as e:
            wait_secs = 2 ** attempt  # 1s, 2s, 4s
            rumble_log(f"[Post reply error, attempt {attempt+1}/{max_retries}] {e} — retrying in {wait_secs}s")
            time.sleep(wait_secs)

    rumble_log("[Post reply failed] Gave up after all retries.")
    return False


def rumble_bot_loop(stream_url: str, username: str, password: str):
    """
    Main loop that runs in its own thread.
    Opens Chrome, logs in, navigates to the stream, and polls for @-mentions.
    Auto-disconnects after MAX_POLL_ERRORS consecutive failures.
    """
    if not SELENIUM_AVAILABLE:
        rumble_log("[Error] selenium / webdriver-manager not installed.")
        rumble_log("Run:  pip install selenium webdriver-manager")
        _set_rumble_status("Error — missing libraries")
        return

    # Launch Chrome
    rumble_log("Launching Chrome…")
    opts = Options()
    opts.add_argument("--start-maximized")
    # Comment out the next line if you want to see the browser window
    # opts.add_argument("--headless=new")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        rumble_state["driver"] = driver
    except Exception as e:
        rumble_log(f"[Chrome launch error] {e}")
        _set_rumble_status("Error — Chrome failed")
        return

    # Login
    if not rumble_login(driver, username, password):
        _set_rumble_status("Error — login failed")
        driver.quit()
        return

    # Navigate
    rumble_navigate_to_stream(driver, stream_url)
    _set_rumble_status("Connected — watching chat…")

    mention_re = re.compile(rf'@{re.escape(BOT_NAME)}\b', re.IGNORECASE)
    consecutive_errors = 0

    while rumble_state["running"]:
        try:
            messages = get_chat_messages(driver)
            consecutive_errors = 0  # reset on successful poll

            for msg in messages:
                uid = msg["id"]
                if uid in rumble_state["seen_ids"]:
                    continue
                rumble_state["seen_ids"][uid] = time.time()
                # Expire seen IDs older than 30 minutes so the same question
                # asked again after a long gap gets answered again, while still
                # suppressing rapid re-answering within a session.
                # Hard-cap at 1000 entries as a safety net against memory growth.
                _now = time.time()
                _TTL = 1800  # 30 minutes
                expired = [k for k, ts in rumble_state["seen_ids"].items()
                           if ts is not None and (_now - ts) > _TTL]
                for k in expired:
                    del rumble_state["seen_ids"][k]
                while len(rumble_state["seen_ids"]) > 1000:
                    rumble_state["seen_ids"].popitem(last=False)  # hard cap, drop oldest

                if mention_re.search(msg["text"]):
                    author   = msg["author"]
                    question = mention_re.sub("", msg["text"]).strip()
                    # Rumble sometimes prepends the sender's username and a newline
                    # before the message body. Strip any leading "Word\n" fragment
                    # so "JulesVerne23\nGenesis 1:2 verse only" becomes "Genesis 1:2 verse only".
                    question = re.sub(r'^\S+\s*\n\s*', '', question).strip(" ,")
                    rumble_log(f"@mention from {author}: {question}")

                    # Log and check for prompt injection — block if detected
                    is_suspicious = write_activity_log("MENTION", author, question)
                    if is_suspicious:
                        rumble_log(f"[BLOCKED] Prompt injection attempt from {author} — skipping Ollama.")
                        block_reply = f"@{author} I can't help with that."
                        post_chat_reply(driver, block_reply)
                        write_activity_log("BLOCKED", author, question, block_reply)
                        continue

                    # ── Verse-only shortcut ──────────────────────────────────
                    # Triggered by "verse only" anywhere in the message.
                    # Example: "@BibleScholar23 verse only Genesis 1:2"
                    # Step 1: check if "verse only" keyword is present
                    _vo_triggered = bool(re.search(r'\bverse\s+only\b', question, re.IGNORECASE))
                    if _vo_triggered:
                        _vo_question = re.sub(r'\bverse\s+only\b', '', question, flags=re.IGNORECASE).strip(" ,")
                        rumble_log(f"[Verse-only] triggered. Ref after strip: '{_vo_question}'")

                        # Step 2: try the local NET Bible lookup first (instant, no AI)
                        verse_text = resolve_verse_from_net(_vo_question)
                        if verse_text is not None:
                            rumble_log(f"[Verse-only] NET hit — posting directly.")
                            reply = f"@{author} {verse_text}"
                            parts = split_into_chat_entries(reply, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                            write_activity_log("RESPONSE", author, question, " | ".join(parts))
                            for i, part in enumerate(parts):
                                post_chat_reply(driver, part)
                                time.sleep(2.5)  # always sleep after each post — avoids Rumble rate-limit kick
                            continue

                        # Step 3: NET Bible lookup missed (book/verse not in JSON) —
                        # fall back to Ollama but instruct it to quote verbatim.
                        rumble_log(f"[Verse-only] NET miss for '{_vo_question}' — asking Ollama to quote verbatim.")
                        verse_prompt = (
                            f"Quote the NET Bible (New English Translation) text of {_vo_question} "
                            f"word for word, exactly as written. "
                            f"Give only the reference and the verse text — nothing else. "
                            f"No commentary, no explanation, no paraphrasing."
                        )
                        answer = get_ollama_response_sync(verse_prompt)
                        if answer:
                            answer = sanitize_rumble_response(answer, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                            prefixed = f"@{author} {answer}"
                            parts = split_into_chat_entries(prefixed, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                            rumble_log(f"[Verse-only] Ollama quoted — sending {len(parts)} message(s).")
                            write_activity_log("RESPONSE", author, question, " | ".join(parts))
                            for i, part in enumerate(parts):
                                post_chat_reply(driver, part)
                                time.sleep(2.5)  # always sleep after each post — avoids Rumble rate-limit kick
                        else:
                            rumble_log(f"[Verse-only] Ollama returned nothing for '{_vo_question}'.")
                        continue

                    # Check canned responses before hitting Ollama
                    canned = check_canned_response(question)
                    if canned:
                        rumble_log(f"Serving canned response for '{question[:40]}'")
                        prefixed = f"@{author} {canned}"
                        parts = split_into_chat_entries(prefixed, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                        write_activity_log("RESPONSE", author, question, " | ".join(parts))
                        for i, part in enumerate(parts):
                            post_chat_reply(driver, part)
                            time.sleep(2.5)  # always sleep after each post — avoids Rumble rate-limit kick
                        continue

                    # Ask the AI — runs synchronously on this background thread,
                    # queuing behind any active UI conversation automatically.
                    rumble_log(f"Waiting for AI response (Ollama may be busy)…")
                    answer = get_ollama_response_sync(
                        f"Rumble chat user {author} asks: {question}",
                    )

                    if answer is None:
                        rumble_log(f"[Gave up] Could not get a response for {author}'s question after all retries.")
                        write_activity_log("ERROR", author, question, "No response after timeout.")
                    else:
                        # Sanitize first — clip at last complete sentence, strip meta-commentary
                        answer = sanitize_rumble_response(answer, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                        # Prefix with @author, then split across however many chunks fit
                        prefixed = f"@{author} {answer}"
                        parts = split_into_chat_entries(prefixed, max_chars=RUMBLE_ENTRY_CHARS, max_parts=_calc_max_parts())
                        full_response = " | ".join(parts)
                        rumble_log(f"Sending {len(parts)} chat message(s)…")

                        # Show the full unsplit response in the response viewer
                        def _update_response_viewer(text=answer):
                            rumble_response_viewer.config(state="normal")
                            rumble_response_viewer.delete("1.0", tk.END)
                            rumble_response_viewer.insert(tk.END, text)
                            rumble_response_viewer.config(state="disabled")
                        root.after(0, _update_response_viewer)

                        # Log the full response alongside the original question
                        write_activity_log("RESPONSE", author, question, full_response)

                        for i, part in enumerate(parts):
                            rumble_log(f"  [{i+1}/{len(parts)}] {part[:60]}{'…' if len(part)>60 else ''}")
                            post_chat_reply(driver, part)
                            time.sleep(2.5)  # always sleep after each post — avoids Rumble rate-limit kick

                else:
                    continue  # no bot response for non-mention messages

        except Exception as e:
            consecutive_errors += 1
            rumble_log(f"[Poll error #{consecutive_errors}] {e}")
            if consecutive_errors >= MAX_POLL_ERRORS:
                rumble_log(f"[Auto-disconnect] {MAX_POLL_ERRORS} consecutive poll errors — stopping bot.")
                rumble_state["running"] = False
                break

        time.sleep(RUMBLE_POLL_INTERVAL)

    rumble_log("Bot stopped.")
    try:
        driver.quit()
    except Exception:
        pass
    rumble_state["driver"] = None
    _set_rumble_status("Disconnected")


# -----------------------------
# RUMBLE UI HELPERS (called from the main thread)
# -----------------------------

def _set_rumble_status(text: str):
    """Update the status label — safe to call from any thread via root.after."""
    def _update():
        rumble_status_label.config(text=text)
    root.after(0, _update)


def _poll_rumble_log():
    """Drain the log queue and append lines to the Rumble log widget."""
    try:
        while True:
            line = rumble_state["log_queue"].get_nowait()
            rumble_log_widget.config(state="normal")
            rumble_log_widget.insert(tk.END, line + "\n")
            rumble_log_widget.see(tk.END)
            rumble_log_widget.config(state="disabled")
    except Empty:
        pass
    root.after(500, _poll_rumble_log)


def connect_rumble():
    """Called by the Connect button."""
    if rumble_state["running"]:
        messagebox.showinfo("Rumble Bot", "Bot is already running.")
        return

    # Gather credentials
    username = rumble_username_var.get().strip() or RUMBLE_USERNAME
    password = rumble_password_var.get().strip() or RUMBLE_PASSWORD
    stream_url = rumble_url_var.get().strip()

    if not username:
        username = simpledialog.askstring("Rumble Login", "Rumble email address:")
        if not username:
            return
        rumble_username_var.set(username)

    if not password:
        password = simpledialog.askstring("Rumble Login", "Rumble password:", show="*")
        if not password:
            return
        # Don't save password in the var for security

    if not stream_url:
        messagebox.showwarning("Rumble Bot", "Please enter the stream URL first.")
        return

    rumble_state["running"] = True
    _set_rumble_status("Connecting…")

    t = threading.Thread(
        target=rumble_bot_loop,
        args=(stream_url, username, password),
        daemon=True
    )
    rumble_state["thread"] = t
    t.start()

    connect_btn.config(state="disabled")
    disconnect_btn.config(state="normal")
    if 'rumble_stop_btn' in globals():
        rumble_stop_btn.config(state="normal")


def disconnect_rumble():
    """Called by the Disconnect button."""
    rumble_state["running"] = False
    connect_btn.config(state="normal")
    disconnect_btn.config(state="disabled")
    if 'rumble_stop_btn' in globals():
        rumble_stop_btn.config(state="disabled")
    _set_rumble_status("Disconnecting…")



# =============================================================================
# UI SETUP  (original layout preserved — Rumble panel added as a new tab/frame)
# =============================================================================
root = tk.Tk()
root.title("BibleScholar — Qwen Edition")
root.geometry("1200x900")
root.configure(bg="#101015")

# ── Ollama startup health check ──────────────────────────────────────────────
if not check_ollama_health():
    messagebox.showerror(
        "Ollama Not Found",
        "Could not connect to Ollama at http://localhost:11434.\n\n"
        "Please ensure Ollama is running before using BibleScholar.\n"
        "Start it with:  ollama serve"
    )

accent_color = "#00b7ff"
accent_soft  = "#0078a0"
panel_bg     = "#15151c"
chat_bg      = "#1b1b23"
text_fg      = "#d4d4d4"
border_color = "#303038"

root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=0)
root.grid_rowconfigure(0, weight=3)   # main chat + Strong's top half
root.grid_rowconfigure(1, weight=1)   # NET Bible panel + analysis panel bottom half
root.grid_rowconfigure(2, weight=1)   # Rumble panel — resizable

# -----------------------------
# SIDEBAR
# -----------------------------
sidebar = tk.Frame(root, bg=panel_bg)
sidebar.grid(row=0, column=0, sticky="nsw")
sidebar.grid_rowconfigure(2, weight=1)

sidebar_title = tk.Label(
    sidebar, text="Conversations", bg=panel_bg, fg=accent_color,
    font=("Segoe UI", 11, "bold")
)
sidebar_title.pack(padx=10, pady=(10, 5), anchor="w")

convo_listbox = tk.Listbox(
    sidebar, bg="#121218", fg=text_fg,
    selectbackground=accent_soft, selectforeground="white",
    borderwidth=0, highlightthickness=0, font=("Segoe UI", 10)
)
convo_listbox.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
convo_listbox.bind("<<ListboxSelect>>", load_conversation)

sidebar_buttons_frame = tk.Frame(sidebar, bg=panel_bg)
sidebar_buttons_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

save_button = tk.Button(
    sidebar_buttons_frame, text="Save", command=save_conversation,
    font=("Segoe UI", 9), bg=accent_soft, fg="white",
    activebackground=accent_color, relief="flat", padx=6, pady=3
)
save_button.pack(side=tk.LEFT, padx=(0, 5))

new_button = tk.Button(
    sidebar_buttons_frame, text="New", command=new_conversation,
    font=("Segoe UI", 9), bg="#303040", fg="white",
    activebackground="#404050", relief="flat", padx=6, pady=3
)
new_button.pack(side=tk.LEFT, padx=(0, 5))

delete_button = tk.Button(
    sidebar_buttons_frame, text="Delete", command=delete_conversation,
    font=("Segoe UI", 9), bg="#5a2020", fg="white",
    activebackground="#7a3030", relief="flat", padx=6, pady=3
)
delete_button.pack(side=tk.LEFT)


for title, _ in conversations:
    convo_listbox.insert(tk.END, title)


# -----------------------------
# MAIN CHAT FRAME
# -----------------------------
main_frame = tk.Frame(root, bg="#101015")
main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 5), pady=10)
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

chat_window = scrolledtext.ScrolledText(
    main_frame, wrap=tk.WORD, state="disabled",
    font=("Consolas", 12), bg=chat_bg, fg=text_fg,
    insertbackground=text_fg, borderwidth=0, relief="flat",
    highlightthickness=1, highlightbackground=border_color
)
chat_window.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0, 8))
chat_window.tag_config("user", foreground=accent_color, font=("Consolas", 12, "bold"))
chat_window.tag_config("bot", foreground="#39FF14")
chat_window.tag_config("verse", foreground="#1E90FF", font=("Consolas", 12, "bold"))
chat_window.tag_config("strongs", foreground="#1E90FF", font=("Consolas", 12, "bold"))

thinking_label = tk.Label(
    main_frame, text="", bg="#101015", fg=accent_color,
    font=("Segoe UI", 9, "italic")
)
thinking_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 3))

# -----------------------------
# INPUT BAR
# -----------------------------
entry = tk.Entry(
    main_frame, font=("Consolas", 12),
    bg="#1a1a22", fg="#ffffff",
    insertbackground="#00b7ff",
    relief="flat",
    highlightthickness=2,
    highlightbackground="#303040",
    highlightcolor=accent_color
)
entry.grid(row=2, column=0, sticky="ew", padx=(0, 5), ipady=6)

send_button = tk.Button(
    main_frame, text="Send", command=send_message,
    font=("Segoe UI", 11, "bold"),
    bg=accent_color, fg="black",
    activebackground="#33c7ff",
    relief="flat",
    padx=14, pady=6
)
send_button.grid(row=2, column=1, sticky="ew")

ai_stop_button = tk.Button(
    main_frame, text="⏹ Stop AI", command=stop_ai_response,
    font=("Segoe UI", 10, "bold"),
    bg="#7a2020", fg="white",
    activebackground="#a03030",
    relief="flat",
    padx=10, pady=6,
    state="disabled"
)
ai_stop_button.grid(row=2, column=2, sticky="ew", padx=(5, 0))

copy_button = tk.Button(
    main_frame, text="Copy Verses", command=copy_verses_to_clipboard,
    font=("Segoe UI", 9),
    bg="#2a2a33", fg="white",
    activebackground="#404050",
    relief="flat",
    padx=10, pady=5
)
copy_button.grid(row=2, column=3, sticky="ew", padx=(5, 0))

# -----------------------------
# VERSE LOOKUP PANEL
# -----------------------------
verse_frame = tk.Frame(main_frame, bg="#101015")
verse_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
verse_frame.grid_columnconfigure(1, weight=1)

verse_label = tk.Label(
    verse_frame, text="Verse lookup:",
    bg="#101015", fg=text_fg, font=("Segoe UI", 9)
)
verse_label.grid(row=0, column=0, padx=(0, 5))

verse_entry = tk.Entry(
    verse_frame, font=("Consolas", 11),
    bg="#202028", fg="#ffffff", insertbackground="#ffffff",
    borderwidth=2, relief="flat"
)
verse_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

verse_button = tk.Button(
    verse_frame, text="Go", command=lookup_verse,
    font=("Segoe UI", 9), bg=accent_soft, fg="white",
    activebackground=accent_color, relief="flat",
    padx=8, pady=3
)
verse_button.grid(row=0, column=2)


# -----------------------------
# NET BIBLE PANEL (BOTTOM)
# -----------------------------
kjv_frame = tk.Frame(root, bg="#181820", height=160)
kjv_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
kjv_frame.grid_propagate(False)
kjv_frame.grid_columnconfigure(0, weight=1)
kjv_frame.grid_rowconfigure(3, weight=1)   # verse_output row expands

# ── drag handle for NET Bible panel ──────────────────────────────────────────
_kjv_drag_start_y = [0]
_kjv_drag_start_h = [0]

kjv_drag_handle = tk.Frame(kjv_frame, bg="#1a2a3a", height=6, cursor="sb_v_double_arrow")
kjv_drag_handle.grid(row=0, column=0, sticky="ew")
kjv_drag_handle.grid_propagate(False)

tk.Label(kjv_drag_handle, text="⠿  drag to resize", bg="#1a2a3a", fg="#336688",
         font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=8)

def _kjv_drag_start(event):
    _kjv_drag_start_y[0] = event.y_root
    _kjv_drag_start_h[0] = kjv_frame.winfo_height()

def _kjv_drag_motion(event):
    delta = _kjv_drag_start_y[0] - event.y_root   # drag up = taller panel
    new_h = max(60, _kjv_drag_start_h[0] + delta)
    kjv_frame.config(height=new_h)
    kjv_frame.grid_propagate(False)
    kjv_frame.update_idletasks()

kjv_drag_handle.bind("<ButtonPress-1>", _kjv_drag_start)
kjv_drag_handle.bind("<B1-Motion>",     _kjv_drag_motion)

# ── thin accent divider ───────────────────────────────────────────────────────
tk.Frame(kjv_frame, bg="#303038", height=2).grid(row=1, column=0, sticky="ew")

kjv_label = tk.Label(
    kjv_frame, text="NET Bible Verses", bg="#181820", fg=accent_color,
    font=("Segoe UI", 11, "bold")
)
kjv_label.grid(row=2, column=0, sticky="w", padx=10, pady=(5, 0))

verse_output = scrolledtext.ScrolledText(
    kjv_frame, height=6, wrap=tk.WORD,
    font=("Consolas", 11), bg="#121218", fg="#d4d4d4",
    insertbackground="#d4d4d4", borderwidth=0, relief="flat"
)
verse_output.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
verse_output.config(state="disabled")


# -----------------------------
# STRONG'S PANEL (RIGHT)
# -----------------------------
strongs_panel = tk.Frame(root, bg=panel_bg, width=220)
strongs_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)
strongs_panel.grid_propagate(False)
strongs_panel.grid_rowconfigure(1, weight=1)
strongs_panel.grid_columnconfigure(0, weight=1)

strongs_title = tk.Label(
    strongs_panel, text="Strong's Words", bg=panel_bg, fg="#ffcc44",
    font=("Segoe UI", 11, "bold")
)
strongs_title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

strongs_output = scrolledtext.ScrolledText(
    strongs_panel, wrap=tk.WORD,
    font=("Consolas", 10), bg="#121218", fg="#d4d4d4",
    insertbackground="#d4d4d4", borderwidth=0, relief="flat"
)
strongs_output.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 10))
strongs_output.tag_config("strongs_num", foreground="#ffcc44", font=("Consolas", 10, "bold"))
strongs_output.tag_config("strongs_lang", foreground="#888888", font=("Consolas", 10, "italic"))
strongs_output.tag_config("strongs_word", foreground="#d4d4d4", font=("Consolas", 10))
strongs_output.tag_config("strongs_missing", foreground="#ff6666", font=("Consolas", 10, "italic"))
strongs_output.config(state="disabled")



# =============================================================================
# RUMBLE BOT PANEL  (row 2, spans all columns) — resizable via drag handle
# =============================================================================

# Outer container that fills row 2 fully
rumble_outer = tk.Frame(root, bg="#0d1a0d", bd=0)
rumble_outer.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)
rumble_outer.grid_columnconfigure(0, weight=1)
rumble_outer.grid_rowconfigure(1, weight=1)   # log+response row expands

# ── drag handle / resize grip ──────────────────────────────────────────────
# A thin bar at the very top of the Rumble panel.  Dragging it up/down
# resizes the panel by adjusting the root row weights.
_drag_start_y   = [0]
_drag_start_h   = [0]

drag_handle = tk.Frame(rumble_outer, bg="#1a3a1a", height=6, cursor="sb_v_double_arrow")
drag_handle.grid(row=0, column=0, sticky="ew")
drag_handle.grid_propagate(False)

tk.Label(drag_handle, text="⠿  drag to resize", bg="#1a3a1a", fg="#336633",
         font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=8)

def _drag_start(event):
    _drag_start_y[0] = event.y_root
    _drag_start_h[0] = rumble_outer.winfo_height()

def _drag_motion(event):
    delta = _drag_start_y[0] - event.y_root          # positive = dragged up = bigger panel
    new_h = max(120, _drag_start_h[0] + delta)
    rumble_outer.config(height=new_h)
    rumble_outer.grid_propagate(False)

drag_handle.bind("<ButtonPress-1>",   _drag_start)
drag_handle.bind("<B1-Motion>",       _drag_motion)

# ── controls frame (all buttons / fields) ─────────────────────────────────
rumble_frame = tk.Frame(rumble_outer, bg="#0d1a0d", bd=0)
rumble_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
rumble_frame.grid_columnconfigure(3, weight=1)
rumble_frame.grid_rowconfigure(4, weight=1)   # log+response row expands

# ── title ──
tk.Label(
    rumble_frame, text="🎥  Rumble Bot", bg="#0d1a0d", fg="#00ff88",
    font=("Segoe UI", 10, "bold")
).grid(row=0, column=0, padx=(10, 6), pady=(6, 4), sticky="w")

# ── username ──
tk.Label(rumble_frame, text="Email:", bg="#0d1a0d", fg=text_fg,
         font=("Segoe UI", 9)).grid(row=0, column=1, sticky="e")
rumble_username_var = tk.StringVar(value=RUMBLE_USERNAME)
tk.Entry(rumble_frame, textvariable=rumble_username_var, width=14,
         bg="#1a2a1a", fg="#ffffff", insertbackground="#00ff88",
         relief="flat").grid(row=0, column=2, padx=(2, 8), sticky="ew")

# ── password ──
tk.Label(rumble_frame, text="Pass:", bg="#0d1a0d", fg=text_fg,
         font=("Segoe UI", 9)).grid(row=0, column=3, sticky="e", padx=(0, 2))
rumble_password_var = tk.StringVar()
tk.Entry(rumble_frame, textvariable=rumble_password_var, show="*", width=14,
         bg="#1a2a1a", fg="#ffffff", insertbackground="#00ff88",
         relief="flat").grid(row=0, column=4, padx=(2, 8), sticky="ew")

# ── connect / disconnect buttons ──
connect_btn = tk.Button(
    rumble_frame, text="Connect", command=connect_rumble,
    font=("Segoe UI", 9, "bold"), bg="#1a6e3a", fg="white",
    activebackground="#28a060", relief="flat", padx=10, pady=3
)
connect_btn.grid(row=0, column=5, padx=(0, 4), pady=(4, 2), sticky="ew")

disconnect_btn = tk.Button(
    rumble_frame, text="Disconnect", command=disconnect_rumble,
    font=("Segoe UI", 9), bg="#5a2020", fg="white",
    activebackground="#7a3030", relief="flat", padx=10, pady=3,
    state="disabled"
)
disconnect_btn.grid(row=0, column=6, padx=(0, 4), pady=(4, 2), sticky="ew")

rumble_stop_btn = tk.Button(
    rumble_frame, text="⏹ Stop Bot", command=disconnect_rumble,
    font=("Segoe UI", 9, "bold"), bg="#7a2020", fg="white",
    activebackground="#a03030", relief="flat", padx=10, pady=3,
    state="disabled"
)
rumble_stop_btn.grid(row=0, column=7, padx=(0, 10), pady=(4, 2), sticky="ew")

# ── stream URL ──
tk.Label(rumble_frame, text="Stream URL:", bg="#0d1a0d", fg=text_fg,
         font=("Segoe UI", 9)).grid(row=1, column=0, padx=(10, 4), pady=(0, 4), sticky="e")
rumble_url_var = tk.StringVar(value="https://rumble.com/live")
rumble_url_entry = tk.Entry(
    rumble_frame, textvariable=rumble_url_var,
    bg="#1a2a1a", fg="#ffffff", insertbackground="#00ff88",
    relief="flat", width=42
)
rumble_url_entry.grid(row=1, column=1, columnspan=4, padx=(0, 8), pady=(0, 4), sticky="ew")

# ── bot name ──
tk.Label(rumble_frame, text="Bot @name:", bg="#0d1a0d", fg=text_fg,
         font=("Segoe UI", 9)).grid(row=1, column=5, padx=(0, 4), sticky="e")
rumble_botname_var = tk.StringVar(value=BOT_NAME)

def _update_bot_name(*_):
    global BOT_NAME
    BOT_NAME = rumble_botname_var.get().strip()

rumble_botname_var.trace_add("write", _update_bot_name)
tk.Entry(rumble_frame, textvariable=rumble_botname_var, width=18,
         bg="#1a2a1a", fg="#ffffff", insertbackground="#00ff88",
         relief="flat").grid(row=1, column=6, padx=(0, 8), sticky="ew")

# ── char limit control ──
tk.Label(rumble_frame, text="Char limit:", bg="#0d1a0d", fg=text_fg,
         font=("Segoe UI", 9)).grid(row=2, column=0, padx=(10, 4), pady=(0, 4), sticky="e")

rumble_char_limit_var = tk.StringVar(value=str(RUMBLE_CHAR_LIMIT))
_char_limit_spinbox = tk.Spinbox(
    rumble_frame, from_=100, to=5000, increment=10,
    textvariable=rumble_char_limit_var, width=7,
    bg="#1a2a1a", fg="#ffffff", insertbackground="#00ff88",
    buttonbackground="#1a3a1a", relief="flat", font=("Segoe UI", 9)
)
_char_limit_spinbox.grid(row=2, column=1, padx=(0, 4), pady=(0, 4), sticky="w")

def _apply_char_limit():
    global RUMBLE_CHAR_LIMIT, RUMBLE_NUM_PREDICT
    try:
        val = int(rumble_char_limit_var.get().strip())
        if val < 50:
            messagebox.showwarning("Char Limit", "Minimum allowed limit is 50.")
            return
        RUMBLE_CHAR_LIMIT = val
        # Recalculate token budget to stay consistent with new char limit
        RUMBLE_NUM_PREDICT = _calc_num_predict(RUMBLE_CHAR_LIMIT)
        _char_limit_feedback.config(text=f"✔ Set to {RUMBLE_CHAR_LIMIT}", fg="#00ff88")
        root.after(2500, lambda: _char_limit_feedback.config(text=""))
    except ValueError:
        messagebox.showerror("Char Limit", "Please enter a whole number.")

tk.Button(
    rumble_frame, text="Set", command=_apply_char_limit,
    font=("Segoe UI", 9), bg="#1a4a2a", fg="white",
    activebackground="#28a060", relief="flat", padx=8, pady=2
).grid(row=2, column=2, padx=(0, 8), pady=(0, 4), sticky="w")

_char_limit_feedback = tk.Label(
    rumble_frame, text="", bg="#0d1a0d", fg="#00ff88",
    font=("Segoe UI", 8, "italic")
)
_char_limit_feedback.grid(row=2, column=3, columnspan=4, padx=(0, 10), pady=(0, 4), sticky="w")

# ── status label ──
rumble_status_label = tk.Label(
    rumble_frame, text="Disconnected", bg="#0d1a0d", fg="#888888",
    font=("Segoe UI", 8, "italic")
)
rumble_status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 2), sticky="w")


# ── log + last response — side by side, both scrollable and resizable ──────
log_response_frame = tk.Frame(rumble_frame, bg="#0d1a0d")
log_response_frame.grid(row=4, column=0, columnspan=7, sticky="nsew",
                        padx=(6, 6), pady=(0, 6))
log_response_frame.grid_columnconfigure(0, weight=1)
log_response_frame.grid_columnconfigure(2, weight=1)
log_response_frame.grid_rowconfigure(1, weight=1)

# activity log (left)
tk.Label(log_response_frame, text="Bot Log", bg="#0d1a0d", fg="#00ff88",
         font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w", padx=(4, 0))

rumble_log_widget = scrolledtext.ScrolledText(
    log_response_frame, height=8, wrap=tk.WORD, state="disabled",
    font=("Consolas", 9), bg="#081508", fg="#88ff88",
    insertbackground="#00ff88", borderwidth=0, relief="flat"
)
rumble_log_widget.grid(row=1, column=0, sticky="nsew", padx=(4, 2))

# vertical divider
tk.Frame(log_response_frame, bg="#1a3a1a", width=2).grid(
    row=0, column=1, rowspan=2, sticky="ns", padx=4)

# last full response viewer (right)
tk.Label(log_response_frame, text="Last Full Response (read-only)", bg="#0d1a0d",
         fg="#ffcc44", font=("Segoe UI", 8, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 4))

rumble_response_viewer = scrolledtext.ScrolledText(
    log_response_frame, height=8, wrap=tk.WORD, state="disabled",
    font=("Consolas", 9), bg="#0a0a18", fg="#d4d4d4",
    insertbackground="#d4d4d4", borderwidth=0, relief="flat"
)
rumble_response_viewer.grid(row=1, column=2, sticky="nsew", padx=(2, 4))


# -----------------------------
# KEYBOARD SHORTCUTS
# -----------------------------
def on_enter(event):
    send_message()
    return "break"

entry.bind("<Return>", on_enter)
verse_entry.bind("<Return>", lambda e: (lookup_verse(), "break"))


# -----------------------------
# START RUMBLE LOG POLLING
# -----------------------------
root.after(500, _poll_rumble_log)


# -----------------------------
# MAINLOOP
# -----------------------------
root.mainloop()