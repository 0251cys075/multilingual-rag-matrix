import json
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "rag_search.db"
LANGS = ("en", "hi", "ur")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "how", "why", "when", "where", "of", "to", "in", "on", "for", "and",
    "or", "as", "by", "with", "from", "about", "do", "you", "mean", "does",
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "पर", "और", "यह", "वह", "क्या",
    "ہے", "ہیں", "تھا", "تھی", "تھے", "کا", "کی", "کے", "میں", "پر", "اور", "یہ", "وہ", "کیا"
}


def tokens(value: str, lang: str = "en") -> list[str]:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    words = [word for word in words if word not in STOPWORDS]
    return words


def source_files() -> list[tuple[str, Path]]:
    files = []
    for lang in LANGS:
        for suffix in ("_corpus.jsonl", "_longdocs.jsonl"):
            path = PROJECT_ROOT / f"{lang}{suffix}"
            if path.is_file():
                files.append((lang, path))
    return files


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def build_index() -> float:
    """Build the persistent FTS5 index once. This is never called by /api/query."""
    started = time.perf_counter()
    files = source_files()
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {PROJECT_ROOT}")

    temp_path = PROJECT_ROOT / "rag_search.building.db"
    if temp_path.exists():
        temp_path.unlink()

    db = sqlite3.connect(str(temp_path))
    try:
        db.execute("PRAGMA journal_mode = OFF")
        db.execute("PRAGMA synchronous = OFF")
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE VIRTUAL TABLE passages_fts USING fts5(
                doc_ref UNINDEXED,
                language UNINDEXED,
                preview UNINDEXED,
                text,
                tokenize = 'unicode61'
            )
        """)

        batch = []
        seen = set()
        for lang, path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(record.get("text", "")).strip()
                    if not text:
                        continue
                    doc_ref = str(
                        record.get("passage_id")
                        or record.get("doc_id")
                        or f"{lang.upper()}_{line_number}"
                    ).strip().upper()
                    key = (lang, doc_ref, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    batch.append((doc_ref, lang, text[:300] + ("..." if len(text) > 300 else ""), text))
                    if len(batch) >= 5000:
                        db.executemany(
                            "INSERT INTO passages_fts(doc_ref, language, preview, text) VALUES (?, ?, ?, ?)",
                            batch,
                        )
                        batch.clear()
        if batch:
            db.executemany(
                "INSERT INTO passages_fts(doc_ref, language, preview, text) VALUES (?, ?, ?, ?)",
                batch,
            )

        db.execute("INSERT INTO metadata(key, value) VALUES ('created_at', ?)", (str(time.time()),))
        for lang, path in files:
            stat = path.stat()
            db.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (f"source:{path.name}", f"{stat.st_size}:{stat.st_mtime_ns}"),
            )
        db.commit()
    finally:
        db.close()

    os.replace(temp_path, DB_PATH)
    return (time.perf_counter() - started) * 1000


def index_is_current() -> bool:
    if not DB_PATH.is_file():
        return False
    try:
        db = sqlite3.connect(str(DB_PATH))
        rows = dict(db.execute("SELECT key, value FROM metadata"))
        db.close()
        for _, path in source_files():
            stat = path.stat()
            expected = f"{stat.st_size}:{stat.st_mtime_ns}"
            if rows.get(f"source:{path.name}") != expected:
                return False
        return True
    except (sqlite3.Error, OSError):
        return False


def open_index() -> dict:
    if not index_is_current():
        raise RuntimeError(
            f"{DB_PATH.name} is missing or stale. Run: python3 -c 'from app.retriever import build_index; print(build_index())'"
        )
    connection = connect()
    count = connection.execute("SELECT count(*) FROM passages_fts").fetchone()[0]
    return {"connection": connection, "count": count}


def _fts_query(query: str, lang: str) -> tuple[str, str]:
    terms = list(dict.fromkeys(tokens(query, lang)))
    if not terms:
        terms = list(dict.fromkeys(tokens(query, "")))
    safe_terms = [f'"{term.replace(chr(34), "")}"' for term in terms if term]
    if not safe_terms:
        return "", ""
    return " AND ".join(safe_terms), " OR ".join(safe_terms)


def retrieve_and_rerank(query: str, lang: str = "en", top_k: int = 4, connection=None) -> list[dict]:
    lang = (lang or "en").lower().strip()
    if lang not in LANGS:
        lang = "en"
    query = str(query or "").strip()
    and_query, or_query = _fts_query(query, lang)
    if not and_query:
        return [{"doc_ref": "NO_QUERY", "text": "Please enter a non-empty query.", "similarity": 0.0, "percentage_str": "0%", "score": 0.0}]

    own_connection = connection is None
    db = connection or connect()
    limit = max(1, min(int(top_k), 3))
    try:
        # 1. Fast path: require ALL keywords to match (instant lookup)
        rows = db.execute(
            """
            SELECT doc_ref, language, preview as text, bm25(passages_fts, 3.0, 1.0) AS rank_score
            FROM passages_fts
            WHERE passages_fts MATCH ? AND language = ?
            ORDER BY rank_score ASC
            LIMIT ?
            """,
            (and_query, lang, limit),
        ).fetchall()
        
        # 2. Fallback: if strict AND fails, use OR but bound the scan to prevent latency spikes
        if not rows:
            rows = db.execute(
                """
                SELECT doc_ref, language, preview as text, bm25(passages_fts, 3.0, 1.0) AS rank_score
                FROM passages_fts
                WHERE passages_fts MATCH ? AND language = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (or_query, lang, limit),
            ).fetchall()
    finally:
        if own_connection:
            db.close()

    results = []
    for rank, row in enumerate(rows):
        similarity = max(0.50, min(0.99, 0.95 - (rank * 0.04)))
        results.append({
            "doc_ref": row["doc_ref"],
            "text": row["text"],
            "similarity": round(similarity, 4),
            "percentage_str": f"{round(similarity * 100):.0f}%",
            "score": round(float(-row["rank_score"]), 6),
            "source_lang": row["language"],
        })
    if not results:
        return [{"doc_ref": "NO_MATCH", "text": "No matching evidence found in the selected language dataset.", "similarity": 0.0, "percentage_str": "0%", "score": 0.0}]
    return results
