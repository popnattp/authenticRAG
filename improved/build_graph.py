# build_graph.py
"""
Build a lightweight Knowledge Graph from the same Markdown corpus
used by authenticRAG.py.

The graph is created OFFLINE before retrieval:
Markdown chunks
    -> Qwen extracts entities + relationships
    -> graph.json

This does NOT change the final answer-generation LLM.
The same Qwen model is only used here as an information-extraction
step to build the retrieval graph.

Install:
    pip install openai networkx

Environment:
    OPENROUTER_API_KEY=...

Run:
    python build_graph.py

Then:
    python onlysearchAuthenticRAG_v2.py
"""

import os
import json
import re
from pathlib import Path
from openai import OpenAI

CORPUS_DIR = Path("./corpus_input")
OUTPUT_FILE = Path("./graph.json")

# Same corpus paths as authenticRAG.py.
MD_PATHS = [
    CORPUS_DIR / "1.md",
    CORPUS_DIR / "2.md",
    CORPUS_DIR / "44.md",
    CORPUS_DIR / "5555.md",
]

MODEL = "qwen/qwen-2.5-72b-instruct"


def get_client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )


def load_markdown():
    docs = []

    for path in MD_PATHS:
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        docs.append({
            "doc_id": path.stem,
            "source": str(path),
            "text": text
        })

    print(f"[LOAD] {len(docs)} documents")
    return docs


def split_markdown(text, max_chars=2500, overlap=300):
    """
    Simple structure-aware chunking.

    First tries to keep Markdown sections together.
    Then splits oversized sections with overlap.
    """
    sections = re.split(
        r"(?=^#{1,6}\s+)",
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
            continue

        start = 0

        while start < len(section):
            end = min(
                start + max_chars,
                len(section)
            )

            chunk = section[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(section):
                break

            start = max(
                0,
                end - overlap
            )

    return chunks


def extract_graph_information(client, text):
    """
    Extract entities and directed relationships.

    The model MUST return JSON only.
    """

    prompt = f"""
Extract a small knowledge graph from the following text.

Return JSON only with exactly this structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "description": "short description"
    }}
  ],
  "relations": [
    {{
      "source": "entity name",
      "relationship": "relationship name",
      "target": "entity name",
      "description": "short explanation"
    }}
  ]
}}

Rules:
- Extract only entities explicitly supported by the text.
- Do not invent facts.
- Keep entity names short and canonical.
- Prefer medically meaningful concepts, diseases, symptoms,
  causes, risk groups, prevention methods, treatments,
  complications, vaccines, outcomes, etc.
- A relationship must be supported by the text.
- If there is no clear relationship, return an empty list.
- Return JSON only.

TEXT:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured knowledge from documents. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1200
    )

    content = response.choices[0].message.content.strip()

    # Remove markdown JSON fences if the model adds them.
    content = re.sub(
        r"^```json\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("[WARN] Invalid JSON from extractor")
        return {
            "entities": [],
            "relations": []
        }


def canonical(text):
    return " ".join(str(text).split()).strip()


def build_graph():
    client = get_client()
    documents = load_markdown()

    graph = {
        "nodes": {},
        "edges": []
    }

    total_chunks = 0

    for doc in documents:
        chunks = split_markdown(doc["text"])

        print(
            f"[DOC] {doc['doc_id']}: "
            f"{len(chunks)} chunks"
        )

        for chunk_id, chunk in enumerate(chunks):
            total_chunks += 1

            print(
                f"  Extracting graph "
                f"{chunk_id + 1}/{len(chunks)}"
            )

            data = extract_graph_information(
                client,
                chunk
            )

            # ------------------------------------------
            # Nodes
            # ------------------------------------------
            for entity in data.get("entities", []):
                name = canonical(
                    entity.get("name", "")
                )

                if not name:
                    continue

                description = canonical(
                    entity.get("description", "")
                )

                if name not in graph["nodes"]:
                    graph["nodes"][name] = {
                        "id": name,
                        "description": description,
                        "documents": [],
                        "chunks": []
                    }

                if doc["doc_id"] not in graph["nodes"][name]["documents"]:
                    graph["nodes"][name]["documents"].append(
                        doc["doc_id"]
                    )

                graph["nodes"][name]["chunks"].append({
                    "doc_id": doc["doc_id"],
                    "chunk_id": chunk_id
                })

            # ------------------------------------------
            # Relations
            # ------------------------------------------
            for relation in data.get("relations", []):
                source = canonical(
                    relation.get("source", "")
                )
                target = canonical(
                    relation.get("target", "")
                )
                relationship = canonical(
                    relation.get(
                        "relationship",
                        "related"
                    )
                )

                if not source or not target:
                    continue

                graph["edges"].append({
                    "source": source,
                    "target": target,
                    "relationship": relationship,
                    "description": canonical(
                        relation.get(
                            "description",
                            ""
                        )
                    ),
                    "doc_id": doc["doc_id"],
                    "chunk_id": chunk_id
                })

    # Remove duplicate edges.
    unique_edges = []
    seen = set()

    for edge in graph["edges"]:
        key = (
            edge["source"],
            edge["relationship"],
            edge["target"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_edges.append(edge)

    graph["edges"] = unique_edges
    graph["nodes"] = list(
        graph["nodes"].values()
    )

    graph["metadata"] = {
        "model": MODEL,
        "documents": [
            d["doc_id"]
            for d in documents
        ],
        "total_chunks": total_chunks,
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"])
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            graph,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n================================")
    print("GRAPH CREATED")
    print("================================")
    print(
        "Nodes:",
        len(graph["nodes"])
    )
    print(
        "Edges:",
        len(graph["edges"])
    )
    print(
        "Chunks:",
        total_chunks
    )
    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    build_graph()
