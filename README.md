# SAP Developer Guidelines MCP Server — ChromaDB Edition

A fully **local, offline** MCP server for Claude Desktop that turns your SAP PDF guidelines into a
semantic code-assistance engine.

## Features

| Feature | Details |
|---|---|
| **Semantic search** | ChromaDB + `all-MiniLM-L6-v2` embeddings — finds *meaning*, not just keywords |
| **Hybrid search** | 70% semantic + 30% keyword score blending |
| **Auto-chunking** | 500-char chunks with 100-char overlap — paragraph-level retrieval |
| **Re-index detection** | MD5 checksum per file — `reindex_all` updates only changed docs |
| **RAG snippets** | ABAP & HANA SQL templates enriched with matching guideline excerpts |
| **Code review** | Best-practice checks + relevant guideline sections pulled via RAG |
| **100% offline** | No cloud calls. Embeddings computed locally via sentence-transformers. |

---

## Quick Start

### 1. Install
```bash
cd sap-mcp-server
chmod +x setup.sh && ./setup.sh
```
The setup script prints your exact `claude_desktop_config.json` block.

First run downloads `all-MiniLM-L6-v2` (~80 MB) once and caches it locally.

### 2. Configure Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sap-dev-guidelines": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

### 3. Restart Claude Desktop → Done.

---

## Available Tools

| Tool | Description |
|---|---|
| `add_document` | Index a PDF or text file (any path) |
| `list_documents` | Show all docs + stale status |
| `remove_document` | Remove by filename |
| `reindex_all` | Re-embed any docs whose file has changed |
| `search_documents` | Hybrid semantic + keyword search |
| `get_document_section` | Read a char range from a doc |
| `abap_snippet` | ABAP template + RAG guideline context |
| `hana_sql_snippet` | HANA SQL template + RAG guideline context |
| `check_guideline` | Code review + matching guideline passages |

### ABAP snippet types
`select` · `class` · `function_module` · `badi` · `exception_class`

### HANA SQL snippet types
`select` · `procedure` · `window_function` · `calculation_view` · `full_text_search` · `graph_workspace`

---

## Example Workflow

```
1. Add your guidelines:
   add_document("/home/roman/docs/SAP_ABAP_Guidelines.pdf", "SAP Clean ABAP Style Guide")
   add_document("/home/roman/docs/HANA_SQL_Reference.pdf",  "SAP HANA SQL & SQLScript Ref")

2. Search semantically:
   search_documents("exception handling best practices")
   → returns relevant chunks even if wording differs

3. Generate a RAG-enriched snippet:
   abap_snippet(snippet_type="class", context="Material master read with error handling")
   → template + matching guideline sections from your PDFs

4. Review your code:
   check_guideline(language="abap", code="SELECT * FROM mara INTO TABLE lt_mat.")
   → flags SELECT *, shows relevant guideline excerpts

5. PDF updated? Just run:
   reindex_all
   → only changed files are re-embedded
```

---

## Directory Structure

```
sap-mcp-server/
├── server.py              ← MCP server (single file)
├── requirements.txt
├── setup.sh               ← One-shot installer (also prints config block)
├── doc_index.json         ← Checksum + metadata per document
├── documents/             ← Drop PDFs here (or pass any abs path)
│   ├── abap_guide.pdf
│   └── abap_guide.cache.txt    ← Plain-text cache for fast retrieval
└── chroma_store/          ← ChromaDB persistent vector index (auto-created)
```

---

## Tuning

Edit the constants at the top of `server.py`:

```python
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks

# Hybrid search blend (in hybrid_search function):
final_score = 0.70 * sem_score + 0.30 * kw_score   # adjust weights
```

To swap the embedding model (e.g. for a larger, more accurate one):
```python
_embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"   # ~420 MB, better for Q&A
)
```
