# onlysearchAuthenticRAG_v2.py
"""
AuthenticRAG V2

Baseline:
    BGE-M3 Dense + BM25 + RRF -> Top-k -> Qwen

V2:
    BGE-M3 Dense + BM25
        -> Weighted RRF
        -> Graph Expansion (up to 2 hops)
        -> Graph entity -> BM25 retrieval
        -> Candidate merge
        -> Cross-Encoder reranking
        -> Top-k
        -> SAME Qwen final generator

The final generation model remains:
    qwen/qwen-2.5-72b-instruct
"""

import os
import json
from collections import deque
from pathlib import Path

import networkx as nx
from openai import OpenAI
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder


class AuthenticSearchRAGV2:

    def __init__(
        self,
        opensearch_host="localhost",
        opensearch_port=9200,
        graph_path="./graph.json",
        reranker_model="BAAI/bge-reranker-v2-m3"
    ):

        # =========================================================
        # SAME FINAL LLM AS ORIGINAL
        # =========================================================
        self.api_key = os.environ.get(
            "OPENROUTER_API_KEY"
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # =========================================================
        # SAME EMBEDDING MODEL
        # =========================================================
        self.embed_model = SentenceTransformer(
            "BAAI/bge-m3"
        )

        class Encoder:

            def __init__(self, model):
                self.model = model

            def embed_query(self, text):
                return self.model.encode(
                    text,
                    normalize_embeddings=True
                ).tolist()

        self.encoder = Encoder(
            self.embed_model
        )

        # =========================================================
        # NEW: CROSS ENCODER
        # =========================================================
        self.reranker = CrossEncoder(
            reranker_model
        )

        # =========================================================
        # OPENSEARCH
        # =========================================================
        self.opensearch_client = OpenSearch(
            hosts=[
                {
                    "host": opensearch_host,
                    "port": opensearch_port
                }
            ],
            use_ssl=False
        )

        self.vector_index_name = (
            "anthropic-vector-index"
        )

        self.bm25_index_name = (
            "anthropic-bm25-index"
        )

        # =========================================================
        # PARAMETERS
        # =========================================================
        self.rrf_k = 60

        # Retrieve more candidates before reranking.
        self.candidate_k = 20

        # Final context documents.
        self.final_k = 5

        # Graph expansion.
        self.max_hops = 2
        self.max_graph_seeds = 3
        self.max_graph_nodes = 30

        # Retrieve docs for each graph entity.
        self.graph_docs_per_entity = 3

        # =========================================================
        # GRAPH
        # =========================================================
        self.graph = self.load_graph(
            graph_path
        )

    # =============================================================
    # GRAPH
    # =============================================================

    def load_graph(self, path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"""
graph.json not found.

Run first:
    python build_graph.py

Expected:
    {path.resolve()}
"""
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        graph = nx.DiGraph()

        for node in data.get(
            "nodes",
            []
        ):

            node_id = str(
                node["id"]
            )

            graph.add_node(
                node_id,
                description=node.get(
                    "description",
                    node_id
                ),
                documents=node.get(
                    "documents",
                    []
                ),
                chunks=node.get(
                    "chunks",
                    []
                )
            )

        for edge in data.get(
            "edges",
            []
        ):

            source = str(
                edge["source"]
            )

            target = str(
                edge["target"]
            )

            # Ensure nodes exist even if the
            # extractor created an edge but no
            # separate entity record.
            if not graph.has_node(source):
                graph.add_node(
                    source,
                    description=source
                )

            if not graph.has_node(target):
                graph.add_node(
                    target,
                    description=target
                )

            graph.add_edge(
                source,
                target,
                relationship=edge.get(
                    "relationship",
                    "related"
                ),
                description=edge.get(
                    "description",
                    ""
                )
            )

        print(
            f"[GRAPH] "
            f"{graph.number_of_nodes()} nodes / "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    def find_graph_seeds(
        self,
        hybrid_results
    ):
        """
        Map retrieved OpenSearch documents to graph nodes.

        Since the original OpenSearch schema does not contain
        graph_id, this implementation uses:
            1. exact entity occurrence in content/context
            2. graph node description occurrence
            3. document id matching graph document metadata
        """

        seeds = []

        for doc_id, score, source in hybrid_results:

            text = (
                str(source.get(
                    "content",
                    ""
                ))
                + "\n"
                + str(source.get(
                    "contextualized_content",
                    ""
                ))
            ).lower()

            # Strong match: graph entity appears
            # explicitly in retrieved content.
            matched = []

            for node_id in self.graph.nodes:

                if node_id.lower() in text:
                    matched.append(
                        node_id
                    )

            if matched:
                seeds.extend(
                    matched[:3]
                )
                continue

            # Second: document metadata
            for node_id, attrs in self.graph.nodes(
                data=True
            ):

                docs = attrs.get(
                    "documents",
                    []
                )

                if str(doc_id) in [
                    str(x) for x in docs
                ]:
                    seeds.append(
                        node_id
                    )

        # Deduplicate while preserving order.
        seeds = list(
            dict.fromkeys(seeds)
        )

        return seeds[
            :self.max_graph_seeds
        ]

    def expand_from_graph(
        self,
        seed_ids
    ):
        """
        LightRAG-style bidirectional graph expansion.

        seed -> outgoing + incoming
             -> hop 1
             -> hop 2

        Relationship travels with each expanded node.
        """

        valid_seeds = [
            sid for sid in seed_ids
            if self.graph.has_node(sid)
        ]

        if not valid_seeds:
            return []

        queue = deque(
            (sid, 0)
            for sid in valid_seeds
        )

        visited = set(
            valid_seeds
        )

        expanded = []

        while queue:

            current, hop = (
                queue.popleft()
            )

            if hop >= self.max_hops:
                continue

            neighbors = (
                list(
                    self.graph.successors(
                        current
                    )
                )
                +
                list(
                    self.graph.predecessors(
                        current
                    )
                )
            )

            for nb in neighbors:

                if nb in visited:
                    continue

                visited.add(nb)

                edge = (
                    self.graph.get_edge_data(
                        current,
                        nb
                    )
                    or
                    self.graph.get_edge_data(
                        nb,
                        current
                    )
                    or {}
                )

                relationship = edge.get(
                    "relationship",
                    "related"
                )

                description = (
                    self.graph.nodes[
                        nb
                    ].get(
                        "description",
                        nb
                    )
                )

                expanded.append({
                    "node_id": nb,
                    "hop": hop + 1,
                    "relationship": relationship,
                    "description": description,
                    "text": (
                        f"[hop{hop + 1}]"
                        f"[{relationship}] "
                        f"{nb}: {description}"
                    )
                })

                if len(expanded) >= (
                    self.max_graph_nodes
                ):
                    return expanded

                queue.append(
                    (nb, hop + 1)
                )

        return expanded

    # =============================================================
    # ORIGINAL SEARCH + SMALL IMPROVEMENTS
    # =============================================================

    def sparse_search(
        self,
        query,
        k=10
    ):

        response = (
            self.opensearch_client.search(
                index=self.bm25_index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "content^2.0",
                                "contextualized_content^1.0"
                            ],
                            "type": "best_fields"
                        }
                    },
                    "size": k
                }
            )
        )

        return [
            (
                hit["_id"],
                hit["_score"],
                hit["_source"]
            )
            for hit in response[
                "hits"
            ][
                "hits"
            ]
        ]

    def dense_search(
        self,
        query,
        k=10
    ):

        embedding = (
            self.encoder.embed_query(
                query
            )
        )

        response = (
            self.opensearch_client.search(
                index=self.vector_index_name,
                body={
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": k
                            }
                        }
                    },
                    "size": k
                }
            )
        )

        return [
            (
                hit["_id"],
                hit["_score"],
                hit["_source"]
            )
            for hit in response[
                "hits"
            ][
                "hits"
            ]
        ]

    def rrf_fusion(
        self,
        sparse_results,
        dense_results
    ):

        scores = {}
        sources = {}

        rankings = [
            sparse_results,
            dense_results
        ]

        # New weighting:
        # BM25 = 0.4
        # Dense = 0.6
        weights = [
            0.4,
            0.6
        ]

        for ranker_id, results in enumerate(
            rankings
        ):

            for rank, (
                doc_id,
                score,
                source
            ) in enumerate(results):

                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    sources[doc_id] = source

                scores[doc_id] += (
                    weights[ranker_id]
                    /
                    (
                        self.rrf_k
                        + rank
                        + 1
                    )
                )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (
                doc_id,
                score,
                sources[doc_id]
            )
            for doc_id, score in ranked
        ]

    def hybrid_search(
        self,
        query,
        k=None
    ):

        if k is None:
            k = self.candidate_k

        sparse = self.sparse_search(
            query,
            k=k
        )

        dense = self.dense_search(
            query,
            k=k
        )

        return self.rrf_fusion(
            sparse,
            dense
        )[:k]

    # =============================================================
    # GRAPH RETRIEVAL
    # =============================================================

    def retrieve_by_graph_entities(
        self,
        graph_nodes
    ):
        """
        Search each graph-expanded entity through BM25.

        This turns graph expansion into actual document retrieval.
        """

        candidates = {}

        for node in graph_nodes:

            query = (
                f"{node['node_id']} "
                f"{node['description']}"
            )

            results = self.sparse_search(
                query,
                k=self.graph_docs_per_entity
            )

            for doc_id, score, source in results:

                if doc_id not in candidates:

                    candidates[doc_id] = {
                        "doc_id": doc_id,
                        "source": source,
                        "graph_score": 0.0,
                        "graph_hits": []
                    }

                # Closer hop gets larger evidence.
                hop_weight = (
                    1.0
                    if node["hop"] == 1
                    else 0.5
                )

                candidates[
                    doc_id
                ][
                    "graph_score"
                ] += (
                    float(score)
                    * hop_weight
                )

                candidates[
                    doc_id
                ][
                    "graph_hits"
                ].append(node)

        return list(
            candidates.values()
        )

    # =============================================================
    # MERGE + RERANK
    # =============================================================

    def merge_candidates(
        self,
        hybrid_results,
        graph_results
    ):

        candidates = {}

        # ------------------------------
        # Hybrid candidates
        # ------------------------------
        for rank, (
            doc_id,
            score,
            source
        ) in enumerate(
            hybrid_results,
            1
        ):

            candidates[doc_id] = {
                "doc_id": doc_id,
                "content": source.get(
                    "content",
                    ""
                ),
                "context": source.get(
                    "contextualized_content",
                    ""
                ),
                "hybrid_score": float(
                    score
                ),
                "hybrid_rank": rank,
                "graph_score": 0.0,
                "graph_hits": []
            }

        # ------------------------------
        # Graph candidates
        # ------------------------------
        for item in graph_results:

            doc_id = item["doc_id"]
            source = item["source"]

            if doc_id not in candidates:

                candidates[doc_id] = {
                    "doc_id": doc_id,
                    "content": source.get(
                        "content",
                        ""
                    ),
                    "context": source.get(
                        "contextualized_content",
                        ""
                    ),
                    "hybrid_score": 0.0,
                    "hybrid_rank": None,
                    "graph_score": 0.0,
                    "graph_hits": []
                }

            # Normalize graph evidence to a
            # bounded value.
            graph_bonus = min(
                1.0,
                item["graph_score"] / 10.0
            )

            candidates[
                doc_id
            ][
                "graph_score"
            ] = max(
                candidates[
                    doc_id
                ][
                    "graph_score"
                ],
                graph_bonus
            )

            candidates[
                doc_id
            ][
                "graph_hits"
            ].extend(
                item["graph_hits"]
            )

        return list(
            candidates.values()
        )

    def rerank(
        self,
        question,
        candidates,
        top_k
    ):

        if not candidates:
            return []

        pairs = []

        for item in candidates:

            document = (
                item["content"]
                + "\n"
                + item["context"]
            )

            # Include graph evidence in the
            # reranking representation.
            if item["graph_hits"]:

                graph_text = "\n".join(
                    [
                        g["text"]
                        for g in item[
                            "graph_hits"
                        ][:3]
                    ]
                )

                document += (
                    "\nGRAPH EVIDENCE:\n"
                    + graph_text
                )

            pairs.append(
                [
                    question,
                    document
                ]
            )

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False
        )

        for item, score in zip(
            candidates,
            scores
        ):

            item[
                "reranker_score"
            ] = float(score)

            # Cross-encoder is the main score.
            # Graph only contributes a small bonus.
            item[
                "final_score"
            ] = (
                float(score)
                + 0.10
                * item["graph_score"]
            )

        return sorted(
            candidates,
            key=lambda x: x[
                "final_score"
            ],
            reverse=True
        )[:top_k]

    # =============================================================
    # COMPLETE RETRIEVAL
    # =============================================================

    def retrieve(
        self,
        question,
        final_k=5
    ):

        # 1. Normal hybrid retrieval.
        hybrid = self.hybrid_search(
            question,
            k=self.candidate_k
        )

        print(
            f"[1] Hybrid candidates: "
            f"{len(hybrid)}"
        )

        # 2. Find graph seeds from retrieved docs.
        seeds = self.find_graph_seeds(
            hybrid
        )

        print(
            f"[2] Graph seeds: {seeds}"
        )

        # 3. Expand graph.
        graph_nodes = (
            self.expand_from_graph(
                seeds
            )
        )

        print(
            f"[3] Graph expansion: "
            f"{len(graph_nodes)} nodes"
        )

        for item in graph_nodes[:10]:
            print(
                "   ",
                item["text"]
            )

        # 4. Search graph entities.
        graph_docs = (
            self.retrieve_by_graph_entities(
                graph_nodes
            )
        )

        print(
            f"[4] Graph-derived docs: "
            f"{len(graph_docs)}"
        )

        # 5. Merge.
        candidates = (
            self.merge_candidates(
                hybrid,
                graph_docs
            )
        )

        print(
            f"[5] Candidates before rerank: "
            f"{len(candidates)}"
        )

        # 6. Cross encoder.
        final = self.rerank(
            question,
            candidates,
            top_k=final_k
        )

        print(
            f"[6] Final documents: "
            f"{len(final)}"
        )

        return final, graph_nodes

    # =============================================================
    # CONTEXT + GENERATION
    # =============================================================

    def build_context(
        self,
        results,
        graph_nodes
    ):

        parts = []

        for i, item in enumerate(
            results,
            1
        ):

            parts.append(
                f"""
DOCUMENT {i}
Content:
{item['content']}

Context:
{item['context']}
""".strip()
            )

        if graph_nodes:

            graph_text = "\n".join(
                [
                    x["text"]
                    for x in graph_nodes[:15]
                ]
            )

            parts.append(
                "GRAPH EVIDENCE:\n"
                + graph_text
            )

        return "\n\n".join(
            parts
        )

    def truncate_context(
        self,
        context,
        max_tokens=4000
    ):

        words = context.split()

        if len(words) * 1.3 <= max_tokens:
            return context

        limit = int(
            max_tokens / 1.3
        )

        return (
            " ".join(words[:limit])
            + "\n...(truncated)"
        )

    def call_qwen_api(
        self,
        prompt,
        max_tokens=500,
        temperature=0.2
    ):

        # SAME final model as baseline.
        response = (
            self.client.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
        )

        return (
            response.choices[0]
            .message
            .content
        )

    def generate_response(
        self,
        question,
        context
    ):

        prompt = f"""
You are a question answering assistant.
Answer the question as truthfully and helpfully as possible.
Use the provided context. Do not invent facts.

Context:
{context}

Question:
{question}
"""

        return self.call_qwen_api(
            prompt,
            max_tokens=1000,
            temperature=0.6
        )

    def search_for_question(
        self,
        question,
        k=5
    ):

        results, graph_nodes = (
            self.retrieve(
                question,
                final_k=k
            )
        )

        if not results:
            return {
                "question": question,
                "answer": "ไม่พบข้อมูลที่เกี่ยวข้อง",
                "results": [],
                "graph_expansion": []
            }

        context = self.build_context(
            results,
            graph_nodes
        )

        context = self.truncate_context(
            context
        )

        answer = self.generate_response(
            question,
            context
        )

        output_results = []

        for item in results:

            output_results.append({
                "doc_id": item["doc_id"],
                "hybrid_score": item[
                    "hybrid_score"
                ],
                "graph_score": item[
                    "graph_score"
                ],
                "reranker_score": item[
                    "reranker_score"
                ],
                "final_score": item[
                    "final_score"
                ],
                "content": item[
                    "content"
                ],
                "context": item[
                    "context"
                ],
                "graph_hits": [
                    {
                        "node_id": g[
                            "node_id"
                        ],
                        "hop": g["hop"],
                        "relationship": g[
                            "relationship"
                        ],
                        "description": g[
                            "description"
                        ]
                    }
                    for g in item[
                        "graph_hits"
                    ]
                ]
            })

        return {
            "question": question,
            "answer": answer,
            "results": output_results,
            "graph_expansion": graph_nodes
        }

    def search_multiple_questions(
        self,
        questions,
        k=5
    ):

        results = []

        for i, question in enumerate(
            questions,
            1
        ):

            print(
                f"\n\n"
                f"===== QUESTION "
                f"{i}/{len(questions)} ====="
            )

            print(
                question
            )

            try:

                result = (
                    self.search_for_question(
                        question,
                        k=k
                    )
                )

                results.append(
                    result
                )

                print(
                    "\nANSWER:\n",
                    result["answer"]
                )

            except Exception as e:

                print(
                    f"ERROR: {e}"
                )

                results.append({
                    "question": question,
                    "answer": str(e),
                    "results": [],
                    "graph_expansion": []
                })

        return results

    def export_results_to_json(
        self,
        results,
        output_file
    ):

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2
            )


def main():

    if "OPENROUTER_API_KEY" not in os.environ:
        print(
            "Set OPENROUTER_API_KEY first."
        )
        return

    rag = AuthenticSearchRAGV2(
        opensearch_host="localhost",
        opensearch_port=9200,
        graph_path="./graph.json",
        reranker_model=(
            "BAAI/bge-reranker-v2-m3"
        )
    )

    questions = [
        "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
        "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
        "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
        "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
        "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?"
    ]

    results = (
        rag.search_multiple_questions(
            questions,
            k=5
        )
    )

    rag.export_results_to_json(
        results,
        "authentic_rag_v2_results.json"
    )


if __name__ == "__main__":
    main()
