"""Hybrid search: FTS5 + vector backend with score fusion."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from typing import Any, Callable

from .types import EmbedFn, Embedder, RerankerFn, SearchResult, embed_for_query

logger = logging.getLogger("sqfox.search")

# Regex for detecting code-like tokens in queries
_CODE_PATTERN = re.compile(
    r"[a-z][A-Z]|_[a-z]|[{}\[\]()]|::|->|=>|\.\w+\("
)
_QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which"}


# ---------------------------------------------------------------------------
# Individual search functions
# ---------------------------------------------------------------------------

def fts_search(
    conn: sqlite3.Connection,
    query_lemmatized: str,
    *,
    limit: int = 20,
) -> list[tuple[int, float]]:
    """Run FTS5 full-text search.

    Args:
        conn:             Reader connection.
        query_lemmatized: Lemmatized query string for FTS5 MATCH.
        limit:            Max results.

    Returns:
        List of (doc_id, relevance_score) tuples.
        Scores are positive (higher = better): negated BM25.
    """
    if not query_lemmatized.strip():
        return []

    # Sanitize FTS5 special characters to prevent syntax errors.
    # Quotes, parentheses, asterisks, carets can crash FTS5 MATCH.
    # Colons trigger column filters, leading hyphens act as NOT prefix.
    safe_query = query_lemmatized
    for ch in '"\'()*^{}[]:+~|':
        safe_query = safe_query.replace(ch, " ")
    # Remove leading hyphens on tokens (FTS5 NOT operator) and standalone dashes
    safe_query = re.sub(r"(?:^|\s)-+", " ", safe_query)
    # Neutralize FTS5 boolean operators — replace standalone AND/OR/NOT/NEAR
    # with lowercase equivalents (FTS5 operators are case-sensitive: uppercase only)
    safe_query = re.sub(r"\bAND\b", "and", safe_query)
    safe_query = re.sub(r"\bOR\b", "or", safe_query)
    safe_query = re.sub(r"\bNOT\b", "not", safe_query)
    safe_query = re.sub(r"\bNEAR\b", "near", safe_query)
    safe_query = safe_query.strip()
    if not safe_query:
        return []

    try:
        rows = conn.execute(
            """
            SELECT d.id, -bm25(documents_fts) as score
            FROM documents_fts f
            JOIN documents d ON d.id = f.rowid
            WHERE documents_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (safe_query, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("FTS5 query failed (bad syntax?): %s", exc)
        return []

    return [(row[0], row[1]) for row in rows]


def vec_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    *,
    limit: int = 20,
    vector_backend=None,
) -> list[tuple[int, float]]:
    """Run vector KNN search.

    If vector_backend is provided, delegates to it.
    Otherwise returns empty (no vector search without a backend).

    Returns:
        List of (doc_id, relevance_score) tuples.
        Scores are converted so higher = better: score = 1 / (1 + distance).
    """
    if vector_backend is None:
        return []

    try:
        raw = vector_backend.search(query_embedding, limit, conn=conn)
    except (TypeError, ValueError):
        raise  # programming errors — don't swallow
    except Exception as exc:
        logger.warning("Vector backend search failed: %s", exc)
        return []
    # Filter out NaN/Inf and negative distances from corrupt embeddings —
    # one bad vector must not poison the entire search pipeline.
    return [
        (key, 1.0 / (1.0 + dist))
        for key, dist in raw
        if dist is not None and math.isfinite(dist) and dist >= 0.0
    ]


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _min_max_normalize(
    results: list[tuple[int, float]],
) -> dict[int, float]:
    """Normalize scores to [0, 1] range using min-max scaling."""
    if not results:
        return {}

    scores = [s for _, s in results]
    min_s = min(scores)
    max_s = max(scores)
    spread = max_s - min_s

    if spread == 0:
        # All scores identical — use 0.5 (neutral) to avoid inflating
        # importance when fused with another source.  Ordering is retained
        # from the original backend.
        return {doc_id: 0.5 for doc_id, _ in results}

    return {
        doc_id: (score - min_s) / spread
        for doc_id, score in results
    }


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def score_fusion(
    fts_results: list[tuple[int, float]],
    vec_results: list[tuple[int, float]],
    *,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Relative Score Fusion: min-max normalize then weighted sum.

    Args:
        fts_results: (doc_id, score) from FTS5.  Higher = better.
        vec_results: (doc_id, score) from vec.  Higher = better.
        alpha:       Weight for vector scores.  (1-alpha) = FTS weight.
                     alpha=0.0 means FTS only, alpha=1.0 means vec only.

    Returns:
        Sorted list of (doc_id, fused_score), highest first.
    """
    fts_norm = _min_max_normalize(fts_results)
    vec_norm = _min_max_normalize(vec_results)

    all_ids = set(fts_norm) | set(vec_norm)

    fused = {}
    for doc_id in all_ids:
        fts_score = fts_norm.get(doc_id, 0.0)
        vec_score = vec_norm.get(doc_id, 0.0)
        fused[doc_id] = (1.0 - alpha) * fts_score + alpha * vec_score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def rrf_fallback(
    fts_results: list[tuple[int, float]],
    vec_results: list[tuple[int, float]],
    *,
    k: int = 60,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion fallback with alpha weighting.

    Used when one result set is too small for meaningful min-max normalization.
    Alpha controls the balance: 0.0 = FTS only, 1.0 = vec only.
    """
    fts_weight = 1.0 - alpha
    vec_weight = alpha
    scores: dict[int, float] = {}

    for rank, (doc_id, _) in enumerate(fts_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + fts_weight / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(vec_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + vec_weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Adaptive alpha
# ---------------------------------------------------------------------------

def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def adaptive_alpha(
    query: str,
    fts_results: list[tuple[int, float]],
    vec_results: list[tuple[int, float]],
    *,
    base_alpha: float = 0.5,
) -> float:
    """Adjust alpha based on query characteristics and score distributions.

    Heuristics:
      1. Code-like tokens in query -> favor FTS (decrease alpha).
      2. Question words at start -> favor vectors (increase alpha).
      3. FTS score variance vs vec variance -> favor the more discriminative.
      4. One result set nearly empty -> favor the other.

    Returns:
        Adjusted alpha clamped to [0.0, 1.0].
    """
    alpha = base_alpha

    # 1. Code-like tokens
    if _CODE_PATTERN.search(query):
        alpha -= 0.15

    # 2. Question words
    first_word = query.strip().split()[0].lower() if query.strip() else ""
    if first_word in _QUESTION_WORDS:
        alpha += 0.10

    # 3. Score distribution comparison
    if len(fts_results) >= 3 and len(vec_results) >= 3:
        fts_scores = [s for _, s in fts_results]
        vec_scores = [s for _, s in vec_results]

        fts_std = _std(fts_scores)
        vec_std = _std(vec_scores)

        if fts_std > 0 and vec_std > 0:
            ratio = fts_std / vec_std
            if ratio > 2.0:
                alpha -= 0.10  # FTS is more discriminative
            elif ratio < 0.5:
                alpha += 0.10  # Vectors are more discriminative

    # 4. Empty result sets
    if len(fts_results) < 2 and len(vec_results) >= 2:
        alpha = max(alpha, 0.8)
    elif len(vec_results) < 2 and len(fts_results) >= 2:
        alpha = min(alpha, 0.2)

    return max(0.0, min(1.0, alpha))


# ---------------------------------------------------------------------------
# Main hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embed_fn: EmbedFn | Embedder,
    *,
    lemmatize_fn: Callable[[str, str | None], str] | None = None,
    limit: int = 10,
    alpha: float | None = None,
    fts_limit: int = 50,
    vec_limit: int = 50,
    min_score: float = 0.0,
    reranker_fn: RerankerFn | None = None,
    rerank_top_n: int | None = None,
    vector_backend=None,
) -> list[SearchResult]:
    """Run hybrid FTS5 + vector search with score fusion.

    Args:
        conn:          Reader connection.
        query:         Raw user query.
        embed_fn:      Embedding function for vector search.
        lemmatize_fn:  Lemmatization function (text, lang) -> lemmatized.
                       If None, uses raw query for FTS.
        limit:         Max final results.
        alpha:         Fusion weight (None = adaptive).
        fts_limit:     How many FTS candidates to retrieve.
        vec_limit:     How many vec candidates to retrieve.
        min_score:     Minimum fused score threshold.
        reranker_fn:   Optional cross-encoder reranker.  Called with
                       (query, candidate_texts) -> scores.
        rerank_top_n:  How many fusion candidates to pass to the
                       reranker.  Defaults to ``limit * 5`` if reranker
                       is provided.

    Returns:
        List of SearchResult, sorted by score descending.
    """
    if limit < 1:
        return []

    # If reranking, we need more candidates from the fusion stage
    fusion_limit = limit
    if reranker_fn is not None:
        fusion_limit = rerank_top_n if rerank_top_n is not None else limit * 5

    # Step 1: FTS search
    fts_results: list[tuple[int, float]] = []
    if lemmatize_fn is not None:
        query_lemmatized = lemmatize_fn(query, None) or query
    else:
        query_lemmatized = query

    try:
        fts_results = fts_search(conn, query_lemmatized, limit=fts_limit)
    except Exception as exc:
        logger.warning("FTS search failed: %s", exc)

    # Step 2: Vector search
    vec_results: list[tuple[int, float]] = []
    try:
        query_embedding = embed_for_query(embed_fn, query)
        vec_results = vec_search(
            conn, query_embedding, limit=vec_limit,
            vector_backend=vector_backend,
        )
    except Exception as exc:
        logger.warning("Vector search failed: %s", exc)

    # Step 3: Fusion
    if not fts_results and not vec_results:
        return []

    if not fts_results:
        norm = _min_max_normalize(vec_results)
        fused = sorted(norm.items(), key=lambda x: x[1], reverse=True)
    elif not vec_results:
        norm = _min_max_normalize(fts_results)
        fused = sorted(norm.items(), key=lambda x: x[1], reverse=True)
    else:
        effective_alpha = alpha
        if effective_alpha is None:
            effective_alpha = adaptive_alpha(query, fts_results, vec_results)

        use_rrf = len(fts_results) < 3 or len(vec_results) < 3
        if use_rrf:
            fused = rrf_fallback(fts_results, vec_results, alpha=effective_alpha)
        else:
            fused = score_fusion(fts_results, vec_results, alpha=effective_alpha)

    # Step 4: Filter and limit to fusion_limit (pre-reranking pool)
    fused = [(doc_id, score) for doc_id, score in fused if score >= min_score]
    fused = fused[:fusion_limit]

    if not fused:
        return []

    # Step 5: Hydrate with document data
    doc_ids = [doc_id for doc_id, _ in fused]

    placeholders = ",".join(["?"] * len(doc_ids))
    rows = conn.execute(
        f"SELECT id, content, metadata, chunk_of FROM documents "
        f"WHERE id IN ({placeholders})",
        doc_ids,
    ).fetchall()

    doc_map = {row[0]: row for row in rows}

    candidates = []
    for doc_id, score in fused:
        if doc_id not in doc_map:
            continue
        row = doc_map[doc_id]
        try:
            meta = json.loads(row[2]) if row[2] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        candidates.append(SearchResult(
            doc_id=row[0],
            score=score,
            text=row[1],
            metadata=meta,
            chunk_id=row[3],
        ))

    # Step 6: Rerank if reranker is provided
    if reranker_fn is not None and candidates:
        texts = [c.text for c in candidates]
        try:
            rerank_scores = reranker_fn(query, texts)
            if len(rerank_scores) == len(candidates):
                # Replace NaN/Inf with -inf so they sort to the bottom
                safe_scores = [
                    rs if math.isfinite(rs) else float("-inf")
                    for rs in rerank_scores
                ]
                candidates = [
                    SearchResult(
                        doc_id=c.doc_id,
                        score=rs,
                        text=c.text,
                        metadata=c.metadata,
                        chunk_id=c.chunk_id,
                    )
                    for c, rs in zip(candidates, safe_scores)
                ]
                candidates.sort(key=lambda r: r.score, reverse=True)
            else:
                logger.warning(
                    "Reranker returned %d scores for %d candidates, skipping",
                    len(rerank_scores), len(candidates),
                )
        except Exception as exc:
            logger.warning("Reranker failed, using fusion scores: %s", exc)

    return candidates[:limit]
