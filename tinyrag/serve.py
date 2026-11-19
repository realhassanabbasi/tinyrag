"""Ask your docs in the browser: a tiny local web UI over the index.

One command, zero dependencies — stdlib http.server only. The server loads
the index once and answers every question through the exact pipeline `ask`
uses (tokenize → vectorize → retrieve → answer), so the page can never
disagree with the CLI. It binds localhost only and serves no files from
disk — the page is one built-in string — so there is nothing to traverse
and nothing exposed to the network.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .answerer import answer
from .evidence import evidence_payload
from .retriever import retrieve
from .store import load_index
from .vectorizer import tokenize, vectorize

DEFAULT_PORT = 8000


def ask_payload(index: dict, question: str, k: int = 3) -> dict:
    """The answer to one question, same shape as `ask --json`.

    Runs the exact CLI pipeline, so a browser answer and a terminal
    answer for the same question are always identical.
    """
    q_terms = tokenize(question)
    q_vec = vectorize(q_terms, index["idf"])
    ranked = retrieve(q_vec, index["vectors"], k=k)
    result = answer(set(q_terms), index["idf"], ranked, index["chunks"])
    return {
        "question": question,
        "answer": result["text"],
        "doc": result["doc"],
        "chunk_id": result["chunk_id"],
        "confidence": result["confidence"],
        "sources": evidence_payload(ranked, index["chunks"], n=k),
    }


def render_page(docs: int, chunks: int) -> str:
    """The whole UI: one self-contained page, no external assets.

    Dark, teal accent, one question box. The counts are baked in at
    render time from the loaded index — the page never guesses.
    """
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
  <p class="sub">Ask questions over your own documents — {docs} documents,
  {chunks} chunks indexed. Answers are quoted from your docs, never made up.</p>
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
form.addEventListener("submit", async (e) => {{
  e.preventDefault();
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  box.className = "show";
  box.innerHTML = '<p class="text">Thinking…</p>';
  const res = await fetch("/ask", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ question }}),
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


def make_handler(index: dict, k: int = 3):
    """A request handler bound to one loaded index.

    GET / serves the page, GET /health reports the index, POST /ask
    answers a question. Anything else is a 404 — no file is ever read
    from disk for a request.
    """
    docs = len({c["doc"] for c in index["chunks"]})
    chunks = len(index["chunks"])

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
                self._send_html(render_page(docs, chunks))
            elif self.path == "/health":
                self._send_json({
                    "ok": True,
                    "docs": docs,
                    "chunks": chunks,
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
            self._send_json(ask_payload(index, question, k=k))

    return Handler


def make_server(index: dict, port: int = 0, k: int = 3) -> ThreadingHTTPServer:
    """A server bound to localhost only. Port 0 picks a free port."""
    return ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(index, k))


def serve(index_path: str, port: int = DEFAULT_PORT, k: int = 3) -> None:
    """Load the index once, then serve the UI until Ctrl+C."""
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
