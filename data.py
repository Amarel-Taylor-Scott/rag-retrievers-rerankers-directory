"""RAG Retrievers & Rerankers Directory — curated registry.

A catalog of the **retrieval + reranking** layer of RAG — sparse/lexical
retrieval, late-interaction retrieval, reranker libraries, open-weight reranker
models, and managed reranking APIs.

This is distinct from RAG *frameworks* (orchestration), *embedding models* (the
vectors), and *vector databases* (storage): these tools *find and re-order* the
most relevant passages. Each entry records: name, org, category, whether it is
open source, its license (or "Proprietary"), a short note, and a canonical link
(GitHub repo / model card for OSS, homepage for hosted services). Uncertain
licenses/URLs are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(Enum):
    sparse_retrieval = "sparse_retrieval"
    late_interaction = "late_interaction"
    reranker_lib = "reranker_lib"
    reranker_model = "reranker_model"
    managed_reranker = "managed_reranker"

    @property
    def label(self) -> str:
        return {
            "sparse_retrieval": "Sparse / Lexical Retrieval",
            "late_interaction": "Late-Interaction Retrieval",
            "reranker_lib": "Reranker Libraries",
            "reranker_model": "Reranker Models",
            "managed_reranker": "Managed Reranker APIs",
        }[self.value]


@dataclass(frozen=True)
class Tool:
    name: str
    org: str
    category: Category
    url: str               # GitHub repo / model card for OSS, homepage for hosted
    open_source: bool
    license: str           # SPDX-ish id, or "Proprietary"
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Registry — curated, public retrieval/reranking tools (links only).
# Licenses verified to best effort; uncertain ones omitted.
# ---------------------------------------------------------------------------
TOOLS: list[Tool] = [
    # --- Sparse / lexical retrieval ---
    Tool("Tantivy", "Tantivy contributors", Category.sparse_retrieval,
         "https://github.com/quickwit-oss/tantivy", True, "MIT",
         note="Rust full-text search library for building inverted indexes and retrieving passages with BM25.",
         tags=("rust", "inverted-index", "bm25", "lexical")),
    Tool("rank_bm25", "dorianbrown", Category.sparse_retrieval,
         "https://github.com/dorianbrown/rank_bm25", True, "Apache-2.0",
         note="Simple, popular BM25 implementations in Python.",
         tags=("bm25", "lexical", "classic")),
    Tool("BM25S", "Xing Han Lù", Category.sparse_retrieval,
         "https://github.com/xhluca/bm25s", True, "MIT",
         note="Fast BM25 in pure NumPy/SciPy with eager sparse scoring.",
         tags=("bm25", "fast")),
    Tool("Pyserini", "Castorini", Category.sparse_retrieval,
         "https://github.com/castorini/pyserini", True, "Apache-2.0",
         note="Reproducible IR with sparse & dense retrieval (Lucene/FAISS).",
         tags=("ir", "lucene", "research")),

    # --- Late-interaction retrieval ---
    Tool("FastPlaid", "LightOn", Category.late_interaction,
         "https://github.com/lightonai/fast-plaid", True, "MIT",
         note="Indexes and searches per-token vectors from ColBERT or ColPali for text and page-image retrieval.",
         tags=("late-interaction", "colbert", "colpali", "multi-vector")),
    Tool("ColBERT", "Stanford FutureData", Category.late_interaction,
         "https://github.com/stanford-futuredata/ColBERT", True, "MIT",
         note="Late-interaction retrieval over token-level embeddings.",
         tags=("late-interaction", "colbert", "popular")),
    Tool("RAGatouille", "Answer.AI", Category.late_interaction,
         "https://github.com/AnswerDotAI/RAGatouille", True, "Apache-2.0",
         note="Easy training & use of ColBERT-style late-interaction models.",
         tags=("colbert", "training")),

    # --- Reranker libraries ---
    Tool("RankLLM", "Castorini", Category.reranker_lib,
         "https://github.com/castorini/rank_llm", True, "Apache-2.0",
         note="Python toolkit for listwise reranking of candidate passages with local models or hosted LLMs.",
         tags=("listwise", "reranking", "llm", "python")),
    Tool("rerankers", "Answer.AI", Category.reranker_lib,
         "https://github.com/AnswerDotAI/rerankers", True, "Apache-2.0",
         note="One unified API over many reranking methods and models.",
         tags=("unified", "api")),
    Tool("FlashRank", "Prithiviraj Damodaran", Category.reranker_lib,
         "https://github.com/PrithivirajDamodaran/FlashRank", True, "Apache-2.0",
         note="Ultra-light, fast reranking with small cross-encoders.",
         tags=("fast", "lightweight")),
    Tool("Sentence-Transformers", "UKPLab / Hugging Face", Category.reranker_lib,
         "https://github.com/UKPLab/sentence-transformers", True, "Apache-2.0",
         note="Bi-encoders & cross-encoders for retrieval and reranking.",
         tags=("cross-encoder", "popular")),

    # --- Reranker models (open weights) ---
    Tool("Qwen3-Reranker-0.6B", "Qwen", Category.reranker_model,
         "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B", True, "Apache-2.0",
         note="Multilingual 0.6B-parameter reranker that scores query-document pairs with task-specific instructions.",
         tags=("multilingual", "instruction-aware", "open-weights")),
    Tool("FlagEmbedding (bge-reranker)", "BAAI", Category.reranker_model,
         "https://github.com/FlagOpen/FlagEmbedding", True, "MIT",
         note="BGE embeddings & rerankers (bge-reranker) from BAAI.",
         tags=("bge", "open-weights")),
    Tool("mxbai-rerank", "Mixedbread", Category.reranker_model,
         "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1", True, "Apache-2.0",
         note="Open-weight cross-encoder reranker models.",
         tags=("open-weights", "cross-encoder")),

    # --- Managed reranker APIs ---
    Tool("Cohere Rerank", "Cohere", Category.managed_reranker,
         "https://cohere.com", False, "Proprietary",
         note="Hosted reranking API (Rerank models).",
         tags=("api", "popular")),
    Tool("Voyage AI", "Voyage AI", Category.managed_reranker,
         "https://www.voyageai.com", False, "Proprietary",
         note="Hosted embedding & reranking APIs.",
         tags=("api", "embeddings")),
    Tool("Jina Reranker", "Jina AI", Category.managed_reranker,
         "https://jina.ai", False, "Proprietary",
         note="Hosted reranker API (with open models available too).",
         tags=("api",)),
]
