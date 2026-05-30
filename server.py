#!/usr/bin/env python3
"""
SAP Developer Guidelines MCP Server — Qdrant Edition
- Local sentence-transformers embeddings (fully offline)
- Qdrant embedded vector store (persistent HNSW, no Docker required)
- Hybrid search: semantic (Qdrant) + keyword (TF-style)
- Auto-chunking with overlap
- Re-index detection via MD5 checksum
- RAG-augmented ABAP / SAP HANA SQL snippet generation
- All heavy ops run in thread pool — event loop never blocked
"""

import json
import uuid
import re
import hashlib
import shutil
import asyncio
import textwrap
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ─── Optional PDF backend ──────────────────────────────────────────────────────
try:
    import fitz
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_BACKEND = "pdfplumber"
    except ImportError:
        PDF_BACKEND = None

# ─── Lazy-loaded heavy deps ────────────────────────────────────────────────────
_qdrant_client = None
_embed_model   = None

COLLECTION = "sap_guidelines"
VECTOR_DIM  = 384   # all-MiniLM-L6-v2 output dimension

def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        _qdrant_client = QdrantClient(path=str(QDRANT_DIR))
        existing = [c.name for c in _qdrant_client.get_collections().collections]
        if COLLECTION not in existing:
            _qdrant_client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _qdrant_client

def _embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()

def _chunk_uuid(chunk_str_id: str) -> str:
    """Deterministic UUID5 from a string chunk ID so re-indexing upserts in-place."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_str_id))

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DOCS_DIR   = BASE_DIR / "documents"
QDRANT_DIR = BASE_DIR / "qdrant_store"
INDEX_FILE = BASE_DIR / "doc_index.json"
for d in [DOCS_DIR, QDRANT_DIR]:
    d.mkdir(exist_ok=True)

# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk (larger = more context per result)
CHUNK_OVERLAP = 200    # overlap to avoid splitting concepts at boundaries

def chunk_text(text: str, doc_name: str) -> list[dict]:
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(text), step)):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) < 80:
            continue
        chunks.append({
            "id":         f"{doc_name}__chunk_{i}",
            "text":       chunk,
            "doc_name":   doc_name,
            "chunk_idx":  i,
            "start_char": start,
        })
        if end >= len(text):
            break
    return chunks

# ─── Index helpers ─────────────────────────────────────────────────────────────

def load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {}

def save_index(index: dict):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def file_checksum(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def extract_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text(errors="replace")
    if PDF_BACKEND == "pymupdf":
        doc = fitz.open(str(path))
        return "\n\n".join(page.get_text() for page in doc)
    elif PDF_BACKEND == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    return "[PDF extraction unavailable — install: pip install pymupdf]"

def _delete_doc_chunks(doc_name: str):
    """Remove all Qdrant points that belong to doc_name."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
    client = _get_client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))])
        ),
    )

def index_document(dest: Path, description: str, index: dict) -> dict:
    """Extract → chunk → embed → upsert into Qdrant. Blocking; call via to_thread."""
    from qdrant_client.models import PointStruct

    text = extract_text(dest)
    cache = DOCS_DIR / (dest.stem + ".cache.txt")
    cache.write_text(text, encoding="utf-8")
    checksum = file_checksum(dest)

    chunks = chunk_text(text, dest.name)

    _delete_doc_chunks(dest.name)

    client = _get_client()
    BATCH = 64
    for b in range(0, len(chunks), BATCH):
        batch   = chunks[b : b + BATCH]
        vectors = _embed([c["text"] for c in batch])
        points  = [
            PointStruct(
                id=_chunk_uuid(c["id"]),
                vector=vec,
                payload={
                    "doc_name":   c["doc_name"],
                    "chunk_idx":  c["chunk_idx"],
                    "start_char": c["start_char"],
                    "text":       c["text"],
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points)

    entry = {
        "description": description,
        "size":        dest.stat().st_size,
        "chars":       len(text),
        "chunks":      len(chunks),
        "checksum":    checksum,
        "cache":       str(cache),
    }
    index[dest.name] = entry
    save_index(index)
    return entry

def check_stale_documents(index: dict) -> list[str]:
    stale = []
    for name, meta in index.items():
        path = DOCS_DIR / name
        if path.exists() and file_checksum(path) != meta.get("checksum", ""):
            stale.append(name)
    return stale

def _remove_document_sync(name: str, index: dict):
    """Blocking removal from Qdrant + cache + index file. Call via to_thread."""
    _delete_doc_chunks(name)
    cache = Path(index[name].get("cache", ""))
    if cache.exists():
        cache.unlink()
    del index[name]
    save_index(index)

# ─── Hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(query: str, n_results: int = 5, doc_filter: Optional[str] = None) -> list[dict]:
    """Qdrant vector search + BM25 re-ranking. Blocking; call via to_thread."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from rank_bm25 import BM25Okapi

    client = _get_client()
    total  = client.count(collection_name=COLLECTION).count
    if total == 0:
        return []

    qvec       = _embed([query])[0]
    semantic_n = min(n_results * 3, total)

    qfilter = (
        Filter(must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_filter))])
        if doc_filter else None
    )

    results = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=semantic_n,
        with_payload=True,
        query_filter=qfilter,
    )

    if not results:
        return []

    # BM25 over the semantic candidates only (fast — small corpus slice)
    texts      = [r.payload["text"] for r in results]
    tokenized  = [re.findall(r'\w+', t.lower()) for t in texts]
    bm25       = BM25Okapi(tokenized)
    qtokens    = re.findall(r'\w+', query.lower())
    bm25_raw   = bm25.get_scores(qtokens)
    max_bm25   = float(max(bm25_raw)) if max(bm25_raw) > 0 else 1.0

    combined = []
    for i, r in enumerate(results):
        sem_score = float(r.score)                # Qdrant cosine: 1.0 = identical
        kw_score  = bm25_raw[i] / max_bm25        # normalised to [0, 1]
        final     = 0.70 * sem_score + 0.30 * kw_score
        combined.append({
            "id":        str(r.id),
            "text":      texts[i],
            "doc_name":  r.payload["doc_name"],
            "chunk_idx": r.payload["chunk_idx"],
            "sem_score": round(sem_score, 3),
            "kw_score":  round(kw_score, 3),
            "score":     round(final, 3),
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n_results]

# ─── RAG context builder ───────────────────────────────────────────────────────

def build_rag_context(query: str, n: int = 4) -> str:
    """Blocking; always call via to_thread or from within a thread."""
    try:
        hits = hybrid_search(query, n_results=n)
        if not hits:
            return ""
        parts = [
            f"[{h['doc_name']} | chunk {h['chunk_idx']} | score {h['score']}]\n{h['text']}"
            for h in hits
        ]
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""

# ─── ABAP templates ────────────────────────────────────────────────────────────

ABAP_TEMPLATES = {
    "select": textwrap.dedent("""\
        * ABAP Open SQL SELECT — HANA-optimized
        DATA: lt_result TYPE TABLE OF <db_table>.

        SELECT field1, field2, field3
          INTO TABLE @lt_result
          FROM <db_table>
          WHERE <condition>
          ORDER BY field1.

        IF sy-subrc = 0.
          LOOP AT lt_result ASSIGNING FIELD-SYMBOL(<ls_row>).
            " process <ls_row>-field1
          ENDLOOP.
        ENDIF."""),

    "class": textwrap.dedent("""\
        CLASS zcl_my_class DEFINITION
          PUBLIC FINAL CREATE PUBLIC.

          PUBLIC SECTION.
            METHODS:
              constructor,
              execute
                IMPORTING iv_input       TYPE string
                RETURNING VALUE(rv_out)  TYPE string
                RAISING   cx_my_error.

          PRIVATE SECTION.
            DATA: mv_state TYPE string.

        ENDCLASS.

        CLASS zcl_my_class IMPLEMENTATION.
          METHOD constructor.
          ENDMETHOD.

          METHOD execute.
            rv_out = iv_input.
          ENDMETHOD.
        ENDCLASS."""),

    "function_module": textwrap.dedent("""\
        FUNCTION z_my_function.
        *" IMPORTING: VALUE(IV_PARAM) TYPE string
        *" EXPORTING: VALUE(EV_RESULT) TYPE string
        *" EXCEPTIONS: my_error = 1, OTHERS = 2
          ev_result = iv_param.
        ENDFUNCTION."""),

    "badi": textwrap.dedent("""\
        CLASS zcl_badi_impl DEFINITION
          PUBLIC FINAL CREATE PUBLIC.
          PUBLIC SECTION.
            INTERFACES if_badi_name.
        ENDCLASS.

        CLASS zcl_badi_impl IMPLEMENTATION.
          METHOD if_badi_name~my_method.
            " BAdI custom logic
          ENDMETHOD.
        ENDCLASS."""),

    "exception_class": textwrap.dedent("""\
        CLASS cx_my_error DEFINITION
          PUBLIC FINAL
          INHERITING FROM cx_static_check.

          PUBLIC SECTION.
            CONSTANTS:
              gc_msgid TYPE symsgid VALUE 'ZMY_MSG'.
            METHODS constructor
              IMPORTING
                textid   LIKE textid OPTIONAL
                previous LIKE previous OPTIONAL
                iv_detail TYPE string OPTIONAL.

        ENDCLASS.

        CLASS cx_my_error IMPLEMENTATION.
          METHOD constructor.
            super->constructor( textid = textid previous = previous ).
          ENDMETHOD.
        ENDCLASS."""),
}

# ─── HANA SQL templates ────────────────────────────────────────────────────────

HANA_TEMPLATES = {
    "select": textwrap.dedent("""\
        -- SAP HANA SQL: Column-projected SELECT
        SELECT
            t."COL1",
            t."COL2",
            COUNT(*) AS "CNT"
        FROM "SCHEMA"."TABLE" AS t
        WHERE t."STATUS" = 'A'
          AND t."DATE" >= '2024-01-01'
        GROUP BY t."COL1", t."COL2"
        ORDER BY "CNT" DESC;"""),

    "procedure": textwrap.dedent("""\
        CREATE OR REPLACE PROCEDURE "SCHEMA"."P_MY_PROC"(
            IN  iv_key    NVARCHAR(10),
            OUT ev_count  INTEGER,
            OUT et_result TABLE (
                "KEY"   NVARCHAR(10),
                "VALUE" DECIMAL(15,2)
            )
        )
        LANGUAGE SQLSCRIPT SQL SECURITY INVOKER AS
        BEGIN
            et_result = SELECT "KEY", "VALUE"
                        FROM "SCHEMA"."TABLE"
                        WHERE "KEY" = :iv_key;
            SELECT COUNT(*) INTO ev_count FROM :et_result;
        END;"""),

    "window_function": textwrap.dedent("""\
        -- SAP HANA: Analytic / Window Functions
        SELECT
            "EMP_ID",
            "DEPT",
            "SALARY",
            AVG("SALARY")  OVER (PARTITION BY "DEPT")                              AS "DEPT_AVG",
            RANK()         OVER (PARTITION BY "DEPT" ORDER BY "SALARY" DESC)       AS "RANK",
            SUM("SALARY")  OVER (ORDER BY "EMP_ID" ROWS UNBOUNDED PRECEDING)       AS "RUNNING_TOTAL"
        FROM "HR"."EMPLOYEES";"""),

    "calculation_view": textwrap.dedent("""\
        -- Querying a HANA Calculation View
        SELECT
            cv."DIM_KEY",
            cv."MEASURE",
            cv."CATEGORY"
        FROM "_SYS_BIC"."my.package/MY_CALC_VIEW" AS cv
        WHERE cv."FISCAL_YEAR" = '2024'
        WITH HINT(NO_CS_JOIN);"""),

    "full_text_search": textwrap.dedent("""\
        -- SAP HANA Full-Text / Fuzzy Search
        SELECT TOP 10
            "DOC_ID",
            "TITLE",
            SCORE() AS "RELEVANCE"
        FROM "SCHEMA"."DOCUMENTS"
        WHERE CONTAINS("CONTENT", :search_term, FUZZY(0.8))
        ORDER BY SCORE() DESC;"""),

    "graph_workspace": textwrap.dedent("""\
        -- SAP HANA Graph: shortest path example
        SELECT * FROM GRAPH_PROCEDURE_RESULT
        WHERE GRAPH_WORKSPACE = 'MY_GRAPH'
          AND SOURCE_VERTEX_ID = 1
          AND TARGET_VERTEX_ID = 99;"""),
}

# ─── Best-practice checks ──────────────────────────────────────────────────────

def check_abap(code: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = code.upper()
    if "SELECT *" in cu:
        issues.append("SELECT * detected — list only needed fields (performance + stability).")
    if "LOOP AT" in cu and "FIELD-SYMBOL" not in cu and "ASSIGNING" not in cu:
        issues.append("LOOP AT without ASSIGNING FIELD-SYMBOL — use field symbols for in-place modification and better performance.")
    if "INTO TABLE" in cu and "@" not in code:
        issues.append("Missing @ host variable escape in Open SQL (e.g. INTO TABLE @lt_result).")
    if "MOVE " in cu:
        issues.append("MOVE is obsolete — use = assignment.")
    if "PERFORM " in cu:
        tips.append("FORM/PERFORM is procedural — prefer ABAP OO (methods in classes).")
    if re.search(r'\bWAIT\b', cu):
        tips.append("WAIT statement detected — verify this is intentional (blocks work process).")
    if "COMMIT WORK" in cu:
        tips.append("COMMIT WORK in business logic layer — consider delegating to a dedicated update task.")
    if not issues and not tips:
        tips.append("No obvious issues detected — consider also checking naming conventions and modularization.")
    return issues, tips

def check_hana_sql(code: str) -> tuple[list, list]:
    issues, tips = [], []
    cu = code.upper()
    if "SELECT *" in cu:
        issues.append("SELECT * — specify columns; columnar store reads only projected columns.")
    if "CURSOR" in cu:
        issues.append("Row-by-row cursor — use set-based SQL for bulk operations in HANA.")
    if "WHILE" in cu:
        issues.append("WHILE loop in SQLScript — prefer bulk SELECT/INSERT over iterative processing.")
    if '"' not in code and re.search(r'\b(FROM|JOIN)\s+\w+\.\w+', code, re.I):
        tips.append('Quote identifiers with double quotes: "SCHEMA"."TABLE".')
    if "TRUNCATE" in cu:
        tips.append("TRUNCATE is DDL — ensure it is intentional; cannot be rolled back in all contexts.")
    if not issues and not tips:
        tips.append("No obvious issues detected.")
    return issues, tips

# ─── Server ────────────────────────────────────────────────────────────────────

app = Server("sap-dev-guidelines-v2")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="add_document",
             description="Add a PDF or text file to the knowledge base. Pass any absolute path or a filename relative to documents/.",
             inputSchema={"type":"object","properties":{
                 "path":        {"type":"string","description":"Absolute or relative file path"},
                 "description": {"type":"string","description":"Short description of the document"},
             },"required":["path"]}),

        Tool(name="list_documents",
             description="List all indexed documents and their status.",
             inputSchema={"type":"object","properties":{}}),

        Tool(name="remove_document",
             description="Remove a document from the knowledge base by filename.",
             inputSchema={"type":"object","properties":{
                 "name": {"type":"string","description":"Filename to remove"},
             },"required":["name"]}),

        Tool(name="reindex_all",
             description="Scan all documents for checksum changes and re-index any that have been updated.",
             inputSchema={"type":"object","properties":{}}),

        Tool(name="search_documents",
             description="Hybrid semantic+keyword search across all indexed guidelines.",
             inputSchema={"type":"object","properties":{
                 "query":       {"type":"string"},
                 "max_results": {"type":"integer","default":5},
                 "doc_filter":  {"type":"string","description":"Optional: restrict to one document filename"},
             },"required":["query"]}),

        Tool(name="get_document_section",
             description="Read a character range from an indexed document.",
             inputSchema={"type":"object","properties":{
                 "name":       {"type":"string"},
                 "start_char": {"type":"integer","default":0},
                 "length":     {"type":"integer","default":3000},
             },"required":["name"]}),

        Tool(name="abap_snippet",
             description=(
                 "Generate an ABAP code template enriched with relevant guideline context via RAG. "
                 "Types: select, class, function_module, badi, exception_class."
             ),
             inputSchema={"type":"object","properties":{
                 "snippet_type": {"type":"string",
                                  "enum":["select","class","function_module","badi","exception_class"]},
                 "context":      {"type":"string","description":"Describe what the code should do"},
             },"required":["snippet_type"]}),

        Tool(name="hana_sql_snippet",
             description=(
                 "Generate a SAP HANA SQL template enriched with relevant guideline context via RAG. "
                 "Types: select, procedure, window_function, calculation_view, full_text_search, graph_workspace."
             ),
             inputSchema={"type":"object","properties":{
                 "snippet_type": {"type":"string",
                                  "enum":["select","procedure","window_function",
                                          "calculation_view","full_text_search","graph_workspace"]},
                 "context":      {"type":"string","description":"Describe what the SQL should do"},
             },"required":["snippet_type"]}),

        Tool(name="check_guideline",
             description="Review ABAP or HANA SQL code against SAP best practices, enriched with indexed guidelines.",
             inputSchema={"type":"object","properties":{
                 "code":     {"type":"string"},
                 "language": {"type":"string","enum":["abap","hana_sql"]},
             },"required":["code","language"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── add_document ───────────────────────────────────────────────────────────
    if name == "add_document":
        raw  = arguments["path"]
        desc = arguments.get("description", "")
        fp   = Path(raw) if Path(raw).is_absolute() else DOCS_DIR / raw
        if not fp.exists():
            return [TextContent(type="text", text=f"File not found: {fp}")]

        dest = DOCS_DIR / fp.name
        if fp.resolve() != dest.resolve():
            shutil.copy2(fp, dest)

        index = load_index()
        if dest.name in index and index[dest.name].get("checksum") == file_checksum(dest):
            return [TextContent(type="text", text=
                f"Already up-to-date: {dest.name} ({index[dest.name]['chunks']} chunks). "
                f"No re-indexing needed.")]

        entry = await asyncio.to_thread(index_document, dest, desc, index)
        return [TextContent(type="text", text=
            f"Indexed: {dest.name}\n"
            f"  Characters : {entry['chars']:,}\n"
            f"  Chunks     : {entry['chunks']:,} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n"
            f"  Description: {desc or '(none)'}\n"
            f"  PDF backend: {PDF_BACKEND or 'none (text files only)'}")]

    # ── list_documents ─────────────────────────────────────────────────────────
    elif name == "list_documents":
        index = load_index()
        stale = check_stale_documents(index)
        if not index:
            return [TextContent(type="text", text="No documents indexed. Use add_document.")]
        lines = [f"Indexed Documents ({len(index)}):\n"]
        for n, m in index.items():
            flag = " [STALE - needs reindex]" if n in stale else ""
            lines.append(
                f"  {n}{flag}\n"
                f"    {m.get('description','(no description)')}\n"
                f"    {m.get('chars',0):,} chars | {m.get('chunks',0)} chunks"
            )
        if stale:
            lines.append(f"\nRun reindex_all to update {len(stale)} changed document(s).")
        return [TextContent(type="text", text="\n".join(lines))]

    # ── remove_document ────────────────────────────────────────────────────────
    elif name == "remove_document":
        n     = arguments["name"]
        index = load_index()
        if n not in index:
            return [TextContent(type="text", text=f"Not found: {n}")]
        await asyncio.to_thread(_remove_document_sync, n, index)
        return [TextContent(type="text", text=f"Removed: {n}")]

    # ── reindex_all ────────────────────────────────────────────────────────────
    elif name == "reindex_all":
        index = load_index()
        stale = check_stale_documents(index)
        if not stale:
            return [TextContent(type="text", text="All documents are up-to-date. Nothing to reindex.")]
        results = []
        for n in stale:
            path  = DOCS_DIR / n
            desc  = index[n].get("description", "")
            entry = await asyncio.to_thread(index_document, path, desc, index)
            results.append(f"  Re-indexed: {n} → {entry['chunks']} chunks")
        return [TextContent(type="text", text=f"Reindexed {len(stale)} document(s):\n" + "\n".join(results))]

    # ── search_documents ───────────────────────────────────────────────────────
    elif name == "search_documents":
        q          = arguments["query"]
        max_r      = int(arguments.get("max_results", 5))
        doc_filter = arguments.get("doc_filter")
        try:
            hits = await asyncio.to_thread(hybrid_search, q, max_r, doc_filter)
        except Exception as e:
            return [TextContent(type="text", text=f"Search error: {e}\nMake sure documents are indexed.")]
        if not hits:
            return [TextContent(type="text", text=f"No results for: {q}")]
        parts = [f"Hybrid search results for '{q}' ({len(hits)} hits):\n"]
        for h in hits:
            parts.append(
                f"[{h['doc_name']} | chunk {h['chunk_idx']} | "
                f"sem={h['sem_score']} kw={h['kw_score']} score={h['score']}]\n"
                f"{h['text']}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(parts))]

    # ── get_document_section ───────────────────────────────────────────────────
    elif name == "get_document_section":
        n      = arguments["name"]
        start  = int(arguments.get("start_char", 0))
        length = int(arguments.get("length", 3000))
        index  = load_index()
        if n not in index:
            return [TextContent(type="text", text=f"Not found: {n}")]
        cp   = Path(index[n].get("cache", ""))
        text = cp.read_text(encoding="utf-8") if cp.exists() else "(no cache)"
        return [TextContent(type="text", text=
            f"{n} [chars {start}–{start+length}]:\n\n{text[start:start+length]}")]

    # ── abap_snippet ───────────────────────────────────────────────────────────
    elif name == "abap_snippet":
        stype     = arguments["snippet_type"]
        ctx       = arguments.get("context", "")
        tmpl      = ABAP_TEMPLATES.get(stype, "* Unknown snippet type")
        rag_query = f"ABAP {stype} {ctx}".strip()
        rag_ctx   = await asyncio.to_thread(build_rag_context, rag_query, 4)

        out = f"ABAP Template: {stype}\n\n```abap\n{tmpl}\n```"
        if ctx:
            out += f"\n\nContext: {ctx}"
        if rag_ctx:
            out += f"\n\n── Relevant Guideline Excerpts (RAG) ──\n\n{rag_ctx}"
        else:
            out += "\n\n(No guideline documents indexed yet. Add PDFs with add_document for RAG context.)"
        return [TextContent(type="text", text=out)]

    # ── hana_sql_snippet ───────────────────────────────────────────────────────
    elif name == "hana_sql_snippet":
        stype     = arguments["snippet_type"]
        ctx       = arguments.get("context", "")
        tmpl      = HANA_TEMPLATES.get(stype, "-- Unknown snippet type")
        rag_query = f"SAP HANA SQL {stype} {ctx}".strip()
        rag_ctx   = await asyncio.to_thread(build_rag_context, rag_query, 4)

        out = f"SAP HANA SQL Template: {stype}\n\n```sql\n{tmpl}\n```"
        if ctx:
            out += f"\n\nContext: {ctx}"
        if rag_ctx:
            out += f"\n\n── Relevant Guideline Excerpts (RAG) ──\n\n{rag_ctx}"
        else:
            out += "\n\n(No guideline documents indexed yet. Add PDFs with add_document for RAG context.)"
        return [TextContent(type="text", text=out)]

    # ── check_guideline ────────────────────────────────────────────────────────
    elif name == "check_guideline":
        code = arguments["code"]
        lang = arguments["language"]

        if lang == "abap":
            issues, tips = check_abap(code)
            rag_query = "ABAP best practices coding guidelines " + " ".join(
                re.findall(r'\b[A-Z_]{3,}\b', code)[:8])
        else:
            issues, tips = check_hana_sql(code)
            rag_query = "SAP HANA SQL best practices performance " + " ".join(
                re.findall(r'\b[A-Z_]{3,}\b', code)[:8])

        rag_ctx = await asyncio.to_thread(build_rag_context, rag_query, 3)

        parts = [f"Code Review ({lang.upper()})\n"]
        if issues:
            parts.append("Issues:\n" + "\n".join(f"  ⚠  {i}" for i in issues))
        if tips:
            parts.append("Tips:\n"   + "\n".join(f"  💡 {t}" for t in tips))
        if rag_ctx:
            parts.append(f"── Matching Guideline Sections ──\n\n{rag_ctx}")
        return [TextContent(type="text", text="\n\n".join(parts))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def _warmup():
    _get_model()   # loads SentenceTransformer weights (~80 MB) into RAM
    _get_client()  # opens Qdrant store and creates collection if absent

async def main():
    # Load model + open DB before accepting requests — eliminates cold-start.
    await asyncio.to_thread(_warmup)
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
