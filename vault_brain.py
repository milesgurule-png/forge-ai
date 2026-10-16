#!/usr/bin/env python3
# Copyright 2026 Miles Gurule / Forge Labs
# SPDX-License-Identifier: Apache-2.0
"""Forge AI v0.1 - "vault brain": local RAG over the Forge Labs Obsidian vault.

Standard library only (sqlite3 + urllib). Vectors are stored as float32 blobs
and compared with plain-Python cosine similarity, which is fast enough at
vault scale (a few thousand chunks). Embeddings and answers both come from a
local Ollama server.

Commands:
    python3 vault_brain.py index [--full]          # (re)embed changed notes
    python3 vault_brain.py ask "question" [-k 8]   # answer with [[note]] citations
    python3 vault_brain.py links "Note Name" [-n 8] [--explain]
                                                   # connect the dots: similar notes in
                                                   # OTHER folders that aren't linked yet
    python3 vault_brain.py suggest [-n 20] [--explain]
                                                   # best unlinked cross-folder pairs, vault-wide
    python3 vault_brain.py stats

Environment: OLLAMA_HOST (default http://localhost:11434),
FORGE_MODEL (default qwen3.5:9b), FORGE_EMBED_MODEL (default qwen3-embedding:0.6b).
"""

import argparse
import array
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

AI_DIR = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.abspath(os.path.expanduser(os.environ.get("FORGE_VAULT", os.path.dirname(AI_DIR))))  # set FORGE_VAULT to point at any Obsidian vault
DB_PATH = os.environ.get("FORGE_DB", os.path.join(AI_DIR, "data", "vault.db"))

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith(("http://", "https://")):
    HOST = "http://" + HOST
CHAT_MODEL = os.environ.get("FORGE_MODEL", "qwen3.5:9b")
EMBED_MODEL = os.environ.get("FORGE_EMBED_MODEL", "qwen3-embedding:0.6b")

# Directory names skipped anywhere in the tree, and vault-relative path prefixes skipped.
SKIP_DIRS = {".git", ".venv", "lab", "ai", ".obsidian", ".claude", ".trash", "node_modules", "__pycache__"}
SKIP_PREFIXES = ("Forge Labs/Director/Status/",)  # live status files churn every few minutes
# Sections that are lists of URLs, not content: kept out of the index.
SKIP_SECTIONS = {"sources"}
# Note types that link to everything (logs, status) are not useful link suggestions.
NO_SUGGEST_TYPES = {"log", "status", "roadmap", "template", "index", "moc"}

MAX_CHUNK_CHARS = 1800
OVERLAP_CHARS = 200
EMBED_BATCH = 16
QUERY_INSTRUCTION = "Given a question, retrieve Obsidian vault note sections that answer it"

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]+")
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have how i if in into
is it its like me more most my no not of on or our so than that the their them then there these
they this to was we what when where which who whom why will with would you your about any all also
just very make made many much one some such use used using other only own same should each between
""".split())


# ---------------------------------------------------------------- Ollama

class OllamaError(RuntimeError):
    pass


def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            detail = json.loads(err.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = err.reason
        raise OllamaError("Ollama HTTP {}: {} (is the model pulled? `ollama pull {}`)".format(
            err.code, detail, payload.get("model")))
    except (urllib.error.URLError, ConnectionError, OSError) as err:
        raise OllamaError("Can't reach Ollama at {} ({}). Start it with `ollama serve`.".format(HOST, err))


def embed(texts):
    """Return a list of L2-normalised float lists, one per input text."""
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        data = _post("/api/embed", {"model": EMBED_MODEL, "input": batch, "truncate": True})
        for vec in data["embeddings"]:
            out.append(normalize(vec))
    return out


def chat(messages, think=False):
    data = _post("/api/chat", {
        "model": CHAT_MODEL, "messages": messages, "stream": False, "think": think,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    })
    return data.get("message", {}).get("content", "").strip(), data


# ---------------------------------------------------------------- vectors

def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def to_blob(vec):
    return array.array("f", vec).tobytes()


def from_blob(blob):
    arr = array.array("f")
    arr.frombytes(blob)
    return arr


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mean_vector(vectors):
    if not vectors:
        return None
    acc = [0.0] * len(vectors[0])
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return normalize(acc)


# ---------------------------------------------------------------- parsing

def parse_frontmatter(text):
    """Very small YAML-frontmatter reader: returns (meta dict, body)."""
    meta = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    block = text[3:end]
    body = text[end + 4:].lstrip("\n")
    for line in block.splitlines():
        if ":" not in line or line.startswith((" ", "-")):
            continue
        key, _, val = line.partition(":")
        val = val.split("#", 1)[0].strip() if not val.strip().startswith(("'", '"')) else val.strip()
        meta[key.strip()] = val.strip().strip("'\"")
    return meta, body


def area_of(relpath):
    """The 'neighbourhood' a note lives in, e.g. 'Forge Labs/History' or 'Cars'."""
    parts = relpath.split("/")[:-1]
    if not parts:
        return "(root)"
    if parts[0] in ("Forge Labs", "Forge Digital") and len(parts) > 1:
        return parts[0] + "/" + parts[1]
    return parts[0]


def split_long(text, limit=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    if len(text) <= limit:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    pieces, cur = [], ""
    for p in paras:
        while len(p) > limit:  # one giant paragraph: hard split
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(p[:limit])
            p = p[limit - overlap:]
        if cur and len(cur) + len(p) + 2 > limit:
            pieces.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        pieces.append(cur)
    return pieces


def chunk_note(title, body):
    """Split a note body by #/##/### headings (ignoring code fences).

    Returns a list of (heading_path, section_name, text) tuples.
    """
    sections = []
    stack = []  # [(level, heading)]
    current, in_fence = [], False
    current_heading = ""

    def flush():
        text = "\n".join(current).strip()
        if text and current_heading.lower() not in SKIP_SECTIONS:
            path = " > ".join(h for _, h in stack) or "(intro)"
            sections.append((path, current_heading or "(intro)", text))

    for line in body.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            current = []
            level, heading = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not (level == 1 and heading == title):
                stack.append((level, heading))
            current_heading = heading
            continue
        current.append(line)
    flush()

    chunks = []
    for path, name, text in sections:
        for piece in split_long(text):
            chunks.append((path, name, piece))
    return chunks


def iter_notes():
    for root, dirs, files in os.walk(VAULT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, VAULT).replace(os.sep, "/")
            if rel.startswith(SKIP_PREFIXES):
                continue
            yield rel, full


# ---------------------------------------------------------------- database

def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS notes (
            path TEXT PRIMARY KEY, title TEXT, area TEXT, type TEXT,
            mtime REAL, hash TEXT, links TEXT);
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, path TEXT, title TEXT, heading TEXT, section TEXT,
            text TEXT, vec BLOB);
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
    """)
    return db


def cmd_index(args):
    t0 = time.time()
    db = connect()
    row = db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
    if args.full or (row and row[0] != EMBED_MODEL):
        if row and row[0] != EMBED_MODEL:
            print("Embedding model changed ({} -> {}): full re-index.".format(row[0], EMBED_MODEL))
        db.execute("DELETE FROM notes")
        db.execute("DELETE FROM chunks")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (EMBED_MODEL,))

    known = {r[0]: (r[1], r[2]) for r in db.execute("SELECT path, mtime, hash FROM notes")}
    seen, n_new, n_skip, n_chunks = set(), 0, 0, 0
    for rel, full in iter_notes():
        seen.add(rel)
        mtime = os.path.getmtime(full)
        if rel in known and known[rel][0] == mtime:
            n_skip += 1
            continue
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if rel in known and known[rel][1] == digest:
            db.execute("UPDATE notes SET mtime=? WHERE path=?", (mtime, rel))
            n_skip += 1
            continue
        title = os.path.splitext(os.path.basename(rel))[0]
        meta, body = parse_frontmatter(text)
        links = sorted(set(m.strip() for m in WIKILINK_RE.findall(body)))
        chunks = chunk_note(title, body)
        texts = ["Note: {} > {}\n{}".format(title, h, t) for h, _, t in chunks]
        vecs = embed(texts) if texts else []
        db.execute("DELETE FROM chunks WHERE path=?", (rel,))
        for (heading, section, _), ctext, vec in zip(chunks, texts, vecs):
            db.execute("INSERT INTO chunks (path, title, heading, section, text, vec) VALUES (?,?,?,?,?,?)",
                       (rel, title, heading, section, ctext, to_blob(vec)))
        db.execute("INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?,?,?)",
                   (rel, title, area_of(rel), meta.get("type", ""), mtime, digest, json.dumps(links)))
        db.commit()
        n_new += 1
        n_chunks += len(chunks)
        print("  embedded {:<70} {:>3} chunks".format(rel[:70], len(chunks)))
    removed = set(known) - seen
    for rel in removed:
        db.execute("DELETE FROM chunks WHERE path=?", (rel,))
        db.execute("DELETE FROM notes WHERE path=?", (rel,))
    db.commit()
    total_notes = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    total_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db.close()
    print("Indexed {} changed notes ({} chunks), skipped {} unchanged, removed {}.".format(
        n_new, n_chunks, n_skip, len(removed)))
    print("Index: {} notes, {} chunks, {:.1f} MB at {} ({:.1f}s)".format(
        total_notes, total_chunks, os.path.getsize(DB_PATH) / 1e6,
        os.path.relpath(DB_PATH, VAULT), time.time() - t0))
    return 0


# ---------------------------------------------------------------- retrieval

def load_chunks(db):
    rows = db.execute("SELECT id, path, title, heading, section, text, vec FROM chunks").fetchall()
    return [dict(id=r[0], path=r[1], title=r[2], heading=r[3], section=r[4], text=r[5],
                 vec=from_blob(r[6])) for r in rows]


def keywords(text):
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def retrieve(db, question, k=8, per_note=2, kw_weight=0.15, title_weight=0.10):
    """Hybrid search: cosine similarity + a keyword-overlap boost (+ title-match boost)."""
    chunks = load_chunks(db)
    if not chunks:
        raise SystemExit("Index is empty. Run: python3 vault_brain.py index")
    qvec = embed(["Instruct: {}\nQuery: {}".format(QUERY_INSTRUCTION, question)])[0]
    qwords = keywords(question)
    for c in chunks:
        c["cos"] = dot(qvec, c["vec"])
        words = keywords(c["text"])
        overlap = len(qwords & words) / len(qwords) if qwords else 0.0
        title_hit = 1.0 if qwords & keywords(c["title"]) else 0.0
        c["kw"] = overlap
        c["score"] = c["cos"] + kw_weight * overlap + title_weight * title_hit
    chunks.sort(key=lambda c: c["score"], reverse=True)
    picked, per = [], {}
    for c in chunks:
        if per.get(c["path"], 0) >= per_note:
            continue
        per[c["path"]] = per.get(c["path"], 0) + 1
        picked.append(c)
        if len(picked) >= k:
            break
    return picked


SYSTEM_PROMPT = """You are Forge AI, the "vault brain" of Miles's Forge Labs Obsidian vault.
Answer ONLY from the note excerpts provided. Rules:
- Cite every claim with the note name in double brackets exactly as given, e.g. [[Augustus]].
- Only cite note names that appear in the excerpts. Never invent note names.
- If the excerpts don't contain the answer, say so plainly instead of guessing.
- Be concise: a short direct answer, then 2-5 supporting bullet points."""


def build_context(chunks):
    parts = []
    for c in chunks:
        body = c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]
        parts.append("[[{}]] (section: {})\n{}".format(c["title"], c["heading"], body))
    return "\n\n---\n\n".join(parts)


def check_citations(answer, chunks):
    cited = []
    for m in WIKILINK_RE.findall(answer):
        name = m.strip()
        if name not in cited:
            cited.append(name)
    allowed = {c["title"].lower() for c in chunks}
    bad = [n for n in cited if n.lower() not in allowed]
    return cited, bad


def cmd_ask(args):
    db = connect()
    t0 = time.time()
    chunks = retrieve(db, args.question, k=args.k)
    t_ret = time.time() - t0
    if args.show_chunks:
        for c in chunks:
            print("  {:.3f} (cos {:.3f}, kw {:.2f})  [[{}]] > {}".format(
                c["score"], c["cos"], c["kw"], c["title"], c["heading"]))
        print()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Note excerpts:\n\n{}\n\nQuestion: {}".format(
            build_context(chunks), args.question)},
    ]
    t1 = time.time()
    answer, raw = chat(messages, think=args.think)
    t_gen = time.time() - t1
    print(answer)
    cited, bad = check_citations(answer, chunks)
    print("\nSources (retrieved):")
    seen = set()
    for c in chunks:
        if c["path"] in seen:
            continue
        seen.add(c["path"])
        mark = "*" if c["title"] in cited else " "
        print("  {} [[{}]]  ({})  score {:.3f}".format(mark, c["title"], c["path"], c["score"]))
    print("  (* = cited in the answer)")
    if bad:
        print("\nWARNING: cited but NOT in the retrieved notes (possible hallucination): " +
              ", ".join("[[{}]]".format(b) for b in bad))
    else:
        print("\nCitation check: OK ({} cited, all from retrieved notes)".format(len(cited)))
    toks = raw.get("eval_count", 0)
    dur = raw.get("eval_duration", 0) / 1e9 or 1
    print("Timing: retrieval {:.2f}s, generation {:.1f}s ({} tokens, {:.1f} tok/s)".format(
        t_ret, t_gen, toks, toks / dur))
    return 0


# ---------------------------------------------------------------- connect the dots

def note_vectors(db):
    """One vector per note: mean of its content chunks (Connections sections excluded,
    so existing links don't inflate similarity)."""
    notes = {}
    for path, title, area, ntype, links in db.execute("SELECT path, title, area, type, links FROM notes"):
        notes[path] = dict(path=path, title=title, area=area, type=ntype.lower(),
                           links={l.lower() for l in json.loads(links)}, chunks=[])
    for path, section, text, blob in db.execute("SELECT path, section, text, vec FROM chunks"):
        if path in notes and section.lower() != "connections":
            notes[path]["chunks"].append((section, text))
            notes[path].setdefault("vecs", []).append(from_blob(blob))
    for n in notes.values():
        n["vec"] = mean_vector(n.pop("vecs", []))
    return {p: n for p, n in notes.items() if n["vec"] is not None}


def already_linked(a, b):
    return b["title"].lower() in a["links"] or a["title"].lower() in b["links"]


def find_note(notes, name):
    name_l = name.lower().strip().strip("[]")
    exact = [n for n in notes.values() if n["title"].lower() == name_l or n["path"].lower() == name_l]
    if exact:
        return exact[0]
    partial = [n for n in notes.values() if name_l in n["title"].lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise SystemExit("Ambiguous note name '{}': {}".format(name, ", ".join(n["title"] for n in partial)))
    raise SystemExit("No indexed note named '{}'.".format(name))


def summary(note, limit=900):
    text = "\n".join(t.split("\n", 1)[-1] for _, t in note["chunks"])
    return text[:limit]


def explain(a, b):
    prompt = ("Two notes from a personal knowledge vault are semantically similar but not linked.\n\n"
              "NOTE A: [[{}]] ({})\n{}\n\nNOTE B: [[{}]] ({})\n{}\n\n"
              "In ONE sentence (max 30 words), state the most specific, concrete idea that connects them. "
              "No preamble.").format(a["title"], a["area"], summary(a), b["title"], b["area"], summary(b))
    text, _ = chat([{"role": "user", "content": prompt}], think=False)
    return " ".join(text.split())


def suggestable(n):
    return n["type"] not in NO_SUGGEST_TYPES and not n["path"].endswith("README.md")


def cmd_links(args):
    db = connect()
    notes = note_vectors(db)
    src = find_note(notes, args.note)
    cands = []
    for n in notes.values():
        if n is src or not suggestable(n):
            continue
        if not args.same_area and n["area"] == src["area"]:
            continue
        if already_linked(src, n):
            continue
        cands.append((dot(src["vec"], n["vec"]), n))
    cands.sort(key=lambda x: x[0], reverse=True)
    print("Connect the dots for [[{}]] ({}). Unlinked notes in other folders:".format(src["title"], src["area"]))
    for sim, n in cands[:args.n]:
        line = "  {:.3f}  [[{}]]  ({})".format(sim, n["title"], n["area"])
        if args.explain:
            line += "\n         why: " + explain(src, n)
        print(line)
    if not cands:
        print("  (nothing: every similar note is already linked)")
    return 0


def cmd_suggest(args):
    db = connect()
    notes = [n for n in note_vectors(db).values() if suggestable(n)]
    pairs = []
    for i, a in enumerate(notes):
        for b in notes[i + 1:]:
            if a["area"] == b["area"] or already_linked(a, b):
                continue
            pairs.append((dot(a["vec"], b["vec"]), a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    out = []
    for sim, a, b in pairs[:args.n]:
        row = {"sim": round(sim, 3), "a": a["title"], "a_area": a["area"], "a_path": a["path"],
               "b": b["title"], "b_area": b["area"], "b_path": b["path"]}
        if args.explain:
            row["why"] = explain(a, b)
        out.append(row)
        print("  {:.3f}  [[{}]] ({})  <->  [[{}]] ({}){}".format(
            sim, a["title"], a["area"], b["title"], b["area"],
            ("\n         why: " + row["why"]) if args.explain else ""))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 0


def cmd_stats(args):
    db = connect()
    n = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    c = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print("{} notes, {} chunks, {:.1f} MB, embed model {}".format(n, c, os.path.getsize(DB_PATH) / 1e6, EMBED_MODEL))
    for area, cnt in db.execute("SELECT area, COUNT(*) FROM notes GROUP BY area ORDER BY 2 DESC"):
        print("  {:>3}  {}".format(cnt, area))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Forge AI vault brain: RAG over the Obsidian vault.")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("index", help="embed new/changed notes into ai/data/vault.db")
    s.add_argument("--full", action="store_true", help="drop the index and re-embed everything")
    s = sub.add_parser("ask", help="answer a question from the vault with [[note]] citations")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=8, help="chunks to retrieve (default 8)")
    s.add_argument("--think", action="store_true", help="enable Qwen thinking (slower)")
    s.add_argument("--show-chunks", action="store_true", help="print retrieval scores first")
    s = sub.add_parser("links", help="suggest unlinked, similar notes in other folders")
    s.add_argument("note")
    s.add_argument("-n", type=int, default=8)
    s.add_argument("--same-area", action="store_true", help="include notes in the same folder")
    s.add_argument("--explain", action="store_true", help="ask the LLM for a one-line reason per link")
    s = sub.add_parser("suggest", help="vault-wide top unlinked cross-folder pairs")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--explain", action="store_true")
    s.add_argument("--json", help="also write the pairs to this JSON file")
    sub.add_parser("stats", help="index size and notes per folder")
    args = p.parse_args(argv)
    handlers = {"index": cmd_index, "ask": cmd_ask, "links": cmd_links, "suggest": cmd_suggest, "stats": cmd_stats}
    if args.cmd not in handlers:
        p.print_help()
        return 1
    try:
        return handlers[args.cmd](args)
    except OllamaError as err:
        sys.stderr.write("Error: {}\n".format(err))
        return 1


if __name__ == "__main__":
    sys.exit(main())
