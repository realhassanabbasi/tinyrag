"""Ask your docs in the browser: a tiny local web UI over the index.

One command, zero dependencies — stdlib http.server only. The server loads
the index once and answers every question through the exact pipeline `ask`
uses (tokenize → vectorize → retrieve → answer), so the page can never
disagree with the CLI. It binds localhost only and serves no files from
disk — the page is one built-in string — so there is nothing to traverse
and nothing exposed to the network.

`serve --index-dir DIR` loads every *.json index in a folder instead of
one file: the page gets a picker, each question is answered from the
index you picked, and /health lists them all. Files in the folder that
are not valid indexes are skipped and named, never fatal.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .answerer import answer
from .evidence import evidence_payload
from .retriever import retrieve
from .store import load_index
from .vectorizer import tokenize, vectorize

DEFAULT_PORT = 8000


def ask_payload(index: dict, question: str, k: int = 3,
                name: str | None = None) -> dict:
    """The answer to one question, same shape as `ask --json`.

    Runs the exact CLI pipeline, so a browser answer and a terminal
    answer for the same question are always identical. `name` is the
    index the answer came from — included when the server knows it,
    so a multi-index answer is never ambiguous about its source.
    """
    q_terms = tokenize(question)
    q_vec = vectorize(q_terms, index["idf"])
    ranked = retrieve(q_vec, index["vectors"], k=k)
    result = answer(set(q_terms), index["idf"], ranked, index["chunks"])
    payload = {
        "question": question,
        "answer": result["text"],
        "doc": result["doc"],
        "chunk_id": result["chunk_id"],
        "confidence": result["confidence"],
        "sources": evidence_payload(ranked, index["chunks"], n=k),
    }
    if name is not None:
        payload["index"] = name
    return payload


def load_indexes(index_dir: str) -> tuple[dict[str, dict], list[str]]:
    """Every *.json index in a folder, keyed by file stem.

    Files that are not valid indexes (unreadable, bad JSON, or missing
    the chunks/idf/vectors shape) are skipped and returned as the
    second item — a stray file never stops the server.
    """
    indexes: dict[str, dict] = {}
    skipped: list[str] = []
    for filename in sorted(os.listdir(index_dir)):
        if not filename.endswith(".json"):
            continue
        name = filename[:-len(".json")]
        try:
            index = load_index(os.path.join(index_dir, filename))
        except (OSError, ValueError):
            skipped.append(filename)
            continue
        if (not isinstance(index, dict)
                or not {"chunks", "idf", "vectors"} <= set(index)):
            skipped.append(filename)
            continue
        indexes[name] = index
    return indexes, skipped


def render_page(docs: int, chunks: int,
                indexes: list[tuple[str, int, int]] | None = None) -> str:
    """The whole UI: one self-contained page, no external assets.

    Dark, teal accent, one question box. The counts are baked in at
    render time from the loaded index — the page never guesses.
    With more than one index, `indexes` is the (name, docs, chunks)
    list and the page gets a picker; switching it updates the counts
    and sends the chosen index with every question.
    """
    picker = ""
    index_map: dict[str, list[int]] = {}
    if indexes and len(indexes) > 1:
        options = "".join(
            f'<option value="{name}">{name} ({d} docs)</option>'
            for name, d, _ in indexes)
        picker = ('<label for="idx">Index</label>'
                  f'<select id="idx">{options}</select>')
        index_map = {name: [d, c] for name, d, c in indexes}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyRAG — ask your docs</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #111827; color: #e5e7eb;
         font-family: system-ui, -apple-system, sans-serif; }}
  main {{ max-width: 640px; margin: 0 auto; padding: 48px 20px; }}
  h1 {{ color: #fff; font-size: 28px; margin: 0 0 4px; }}
  h1 span {{ color: #0d9488; }}
  p.sub {{ color: #9ca3af; margin: 0 0 24px; }}
  label {{ display: block; color: #9ca3af; font-size: 13px;
          text-transform: uppercase; letter-spacing: 0.05em;
          margin: 0 0 6px; }}
  select {{ width: 100%; padding: 12px 14px; font-size: 16px;
           border-radius: 8px; border: 1px solid #374151;
           background: #1f2937; color: #fff; margin-bottom: 16px; }}
  select:focus {{ outline: 2px solid #0d9488; border-color: #0d9488; }}
  form {{ display: flex; gap: 8px; }}
  input {{ flex: 1; padding: 12px 14px; font-size: 16px; border-radius: 8px;
          border: 1px solid #374151; background: #1f2937; color: #fff; }}
  input:focus {{ outline: 2px solid #0d9488; border-color: #0d9488; }}
  button {{ padding: 12px 20px; font-size: 16px; border: 0; border-radius: 8px;
           background: #0d9488; color: #fff; cursor: pointer; }}
  button:hover {{ background: #0f766e; }}
  #answer {{ margin-top: 24px; padding: 20px; border-radius: 8px;
            background: #1f2937; display: none; }}
  #answer.show {{ display: block; }}
  #answer .text {{ font-size: 18px; line-height: 1.5; color: #fff; }}
  #answer .cite {{ margin-top: 12px; color: #9ca3af; font-size: 14px; }}
  #answer .cite b {{ color: #2dd4bf; font-weight: 600; }}
  #answer.miss .text {{ color: #9ca3af; }}
  #evidence {{ margin-top: 16px; border-top: 1px solid #374151;
              padding-top: 12px; }}
  #evidence h2 {{ font-size: 13px; text-transform: uppercase;
                  letter-spacing: 0.05em; color: #9ca3af; margin: 0 0 8px; }}
  #evidence ol {{ margin: 0; padding-left: 20px; color: #d1d5db;
                  font-size: 14px; line-height: 1.5; }}
  #evidence li {{ margin-bottom: 6px; }}
  #evidence li b {{ color: #2dd4bf; font-weight: 600; }}
  footer {{ margin-top: 32px; color: #6b7280; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>Tiny<span>RAG</span></h1>
  <p class="sub">Ask questions over your own documents — <span id="counts">{docs} documents,
  {chunks} chunks indexed</span>. Answers are quoted from your docs, never made up.</p>
  {picker}
  <form id="ask-form">
    <input id="q" type="text" placeholder="What is backpropagation?"
           autocomplete="off" autofocus>
    <button type="submit">Ask</button>
  </form>
  <div id="answer" role="status"></div>
  <footer>Local only — this page runs on your machine and talks to nothing
  but the index on disk.</footer>
</main>
<script>
const form = document.getElementById("ask-form");
const box = document.getElementById("answer");
const picker = document.getElementById("idx");
const counts = document.getElementById("counts");
const INDEXES = {json.dumps(index_map)};
if (picker) {{
  picker.addEventListener("change", () => {{
    const c = INDEXES[picker.value];
    counts.textContent = c[0] + " documents, " + c[1] + " chunks indexed";
  }});
}}
form.addEventListener("submit", async (e) => {{
  e.preventDefault();
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  box.className = "show";
  box.innerHTML = '<p class="text">Thinking…</p>';
  const body = {{ question }};
  if (picker) body.index = picker.value;
  const res = await fetch("/ask", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(body),
  }});
  const data = await res.json();
  if (!res.ok) {{
    box.className = "show miss";
    box.innerHTML = '<p class="text">' + data.error + "</p>";
    return;
  }}
  let html = '<p class="text">' + data.answer + "</p>";
  box.className = "show";
  if (data.doc) {{
    html += '<p class="cite">From <b>' + data.doc + "</b> (chunk "
      + data.chunk_id + ", confidence " + data.confidence + ")</p>";
    if (data.sources && data.sources.length > 1) {{
      html += "<div id='evidence'><h2>Evidence</h2><ol>";
      for (const s of data.sources) {{
        html += "<li><b>" + s.doc + "</b> (chunk " + s.chunk_id + ", "
          + s.score + ") — " + s.excerpt + "</li>";
      }}
      html += "</ol></div>";
    }}
  }} else {{
    box.className = "show miss";
  }}
  box.innerHTML = html;
}});
</script>
</body>
</html>
"""


def make_handler(indexes: dict | dict[str, dict], k: int = 3):
    """A request handler bound to one or more loaded indexes.

    A single index dict is accepted for the old call shape and served
    under the name "index". A dict of name → index serves them all:
    GET / shows the picker, GET /health lists every index, and
    POST /ask takes an optional "index" name — required when there
    is more than one, so a question can never silently land on the
    wrong documents. Anything else is a 404 — no file is ever read
    from disk for a request.
    """
    if "chunks" in indexes:  # a single loaded index, not a name → index map
        indexes = {"index": indexes}
    names = sorted(indexes)
    stats = {
        name: (len({c["doc"] for c in indexes[name]["chunks"]}),
               len(indexes[name]["chunks"]))
        for name in names
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path == "/":
                first = stats[names[0]]
                self._send_html(render_page(
                    first[0], first[1],
                    indexes=[(n, *stats[n]) for n in names]))
            elif self.path == "/health":
                self._send_json({
                    "ok": True,
                    "indexes": [
                        {"name": n, "docs": stats[n][0],
                         "chunks": stats[n][1]}
                        for n in names
                    ],
                    "version": __version__,
                })
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path != "/ask":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
            except (ValueError, TypeError):
                self._send_json({"error": "invalid JSON"}, status=400)
                return
            question = body.get("question") if isinstance(body, dict) else None
            if not isinstance(question, str) or not question.strip():
                self._send_json(
                    {"error": "question must be a non-empty string"},
                    status=400)
                return
            name = body.get("index") if isinstance(body, dict) else None
            if name is None:
                if len(names) > 1:
                    self._send_json(
                        {"error": "pick an index", "indexes": names},
                        status=400)
                    return
                name = names[0]
            elif not isinstance(name, str) or name not in indexes:
                self._send_json(
                    {"error": f"unknown index {name!r}", "indexes": names},
                    status=400)
                return
            self._send_json(ask_payload(indexes[name], question, k=k,
                                        name=name))

    return Handler


def make_server(indexes: dict | dict[str, dict], port: int = 0,
                k: int = 3) -> ThreadingHTTPServer:
    """A server bound to localhost only. Port 0 picks a free port."""
    return ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(indexes, k))


def serve(index_path: str | None = None, index_dir: str | None = None,
          port: int = DEFAULT_PORT, k: int = 3) -> None:
    """Load the index once (or every index in a folder), then serve the
    UI until Ctrl+C."""
    if index_dir is not None:
        indexes, skipped = load_indexes(index_dir)
        if not indexes:
            print(f"no indexes found in {index_dir} — run "
                  "`tinyrag index <folder> --out "
                  f"{index_dir}/<name>.json` first")
            return
        server = make_server(indexes, port, k)
        host, bound_port = server.server_address
        names = sorted(indexes)
        print(f"serving {len(names)} indexes ({', '.join(names)}) at "
              f"http://{host}:{bound_port} — Ctrl+C to stop", flush=True)
        if skipped:
            print(f"skipped {len(skipped)} file(s) that are not indexes: "
                  f"{', '.join(skipped)}", flush=True)
    else:
        index = load_index(index_path)
        server = make_server(index, port, k)
        host, bound_port = server.server_address
        print(f"serving your docs at http://{host}:{bound_port} — Ctrl+C to stop",
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
