"""
RAG layer: cross-references each input paper's claims against prior work
from two sources:

  1. A persistent local library (Chroma, on disk at config.CHROMA_DIR) of
     every paper this tool has ever processed - grows across runs. This is
     what lets the Skeptic eventually say "you claimed X, but a paper we
     processed three runs ago found the opposite."
  2. Live arXiv search for papers related to each input paper's title -
     catches prior work this tool has never seen before, not just its own
     accumulated library.

Embeddings run fully locally via sentence-transformers (free, no API key,
no rate limit) - only the arXiv *search* calls need network, and those
share ingestion.py's rate-limited client.

Every function here is best-effort: Chroma, sentence-transformers, or arXiv
search failing for any reason (missing deps, no internet on first run before
the embedding model is cached, arXiv rate limit) logs a warning and returns
an empty result rather than raising - RAG is a pure enhancement layer, never
a hard dependency for getting a script out of the pipeline.
"""
import re
import hashlib
import logging
from typing import List, Optional

from .models import PaperData, PaperSummary, RAGMatch
from . import config

logger = logging.getLogger(__name__)

_collection = None  # lazily-initialized Chroma collection, cached at module level
_lazy_ef_class = None


def _lazy_embedding_function():
    """Chroma's SentenceTransformerEmbeddingFunction imports torch and loads
    the model in its constructor, so merely opening the collection (e.g. to
    list it in the dashboard) took ~45s on a cold start.

    This subclass keeps the same name and config, so Chroma treats it as the
    same embedding function the library was created with, but only loads
    the model the first time something needs embeddings (a query or an
    upsert). It is also registered under that name in Chroma's registry,
    because opening a collection rebuilds the embedding function from its
    saved config via that registry - otherwise Chroma would build the
    eager original anyway."""
    global _lazy_ef_class
    if _lazy_ef_class is None:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
            known_embedding_functions,
        )

        class LazySentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
            def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                         normalize_embeddings: bool = False, **kwargs):
                # Deliberately skips the parent __init__, which loads the model.
                self.model_name = model_name
                self.device = device
                self.normalize_embeddings = normalize_embeddings
                self.kwargs = kwargs

            @property
            def _model(self):
                if self.model_name not in self.models:
                    from sentence_transformers import SentenceTransformer
                    self.models[self.model_name] = SentenceTransformer(
                        model_name_or_path=self.model_name, device=self.device, **self.kwargs
                    )
                return self.models[self.model_name]

            @staticmethod
            def build_from_config(cfg):
                return LazySentenceTransformerEmbeddingFunction(
                    model_name=cfg["model_name"], device=cfg["device"],
                    normalize_embeddings=cfg["normalize_embeddings"], **cfg.get("kwargs", {}),
                )

        # Same dict object Chroma reads from, so replace the entry in place.
        known_embedding_functions[SentenceTransformerEmbeddingFunction.name()] = (
            LazySentenceTransformerEmbeddingFunction
        )
        _lazy_ef_class = LazySentenceTransformerEmbeddingFunction

    return _lazy_ef_class(model_name=config.EMBEDDING_MODEL)


def _get_collection():
    """Returns the persistent Chroma collection, creating it on first call.
    Cached afterward. The embedding model itself loads only when a query or
    upsert first needs it - listing or deleting papers never loads it."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    embed_fn = _lazy_embedding_function()
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _collection = client.get_or_create_collection(
        name="paper_library", embedding_function=embed_fn
    )
    _migrate_path_ids(_collection)
    return _collection


def _is_path_id(paper_id: str) -> bool:
    return paper_id.lower().endswith(".pdf") and ("/" in paper_id or "\\" in paper_id)


def _migrate_path_ids(collection) -> None:
    """Older versions used an uploaded PDF's temp-file path as its paper ID.
    Rename those entries to a stable 'pdf-legacy-<hash>' ID, keeping their
    stored text and embeddings (so no model load). Runs on every open but
    does nothing once no path IDs are left. Best-effort."""
    try:
        ids = collection.get(include=[])["ids"]
        old_ids = [i for i in ids if _is_path_id(i)]
        if not old_ids:
            return
        old = collection.get(ids=old_ids, include=["embeddings", "documents", "metadatas"])
        new_ids, metadatas = [], []
        for old_id, meta in zip(old["ids"], old["metadatas"]):
            new_id = "pdf-legacy-" + hashlib.sha256(old_id.encode("utf-8")).hexdigest()[:10]
            new_ids.append(new_id)
            metadatas.append({**(meta or {}), "paper_id": new_id})
        collection.upsert(
            ids=new_ids, embeddings=old["embeddings"],
            documents=old["documents"], metadatas=metadatas,
        )
        collection.delete(ids=old["ids"])
        logger.info(f"Renamed {len(new_ids)} RAG library entries with path IDs")
    except Exception as e:
        logger.warning(f"Could not rename path IDs in the RAG library: {e}")


def add_paper_to_library(paper: PaperData, summary: PaperSummary) -> None:
    """Persist a processed paper into the local library so future runs can
    find it as related work. Safe to call repeatedly - upsert overwrites."""
    try:
        collection = _get_collection()
        doc_text = (
            f"{paper.title}\n\n{summary.summary}\n\n"
            f"Key claims: {'; '.join(summary.key_claims)}"
        )
        collection.upsert(
            ids=[paper.paper_id],
            documents=[doc_text],
            metadatas=[{"title": paper.title, "paper_id": paper.paper_id}],
        )
    except Exception as e:
        logger.warning(f"Failed to add {paper.paper_id} to RAG library: {e}")

def list_library() -> List[dict]:
    """Returns a list of all papers currently in the Chroma library."""
    try:
        collection = _get_collection()
        results = collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []
        
        papers = []
        for meta, doc_id in zip(results["metadatas"], results["ids"]):
            papers.append({
                "id": doc_id,
                "title": meta.get("title", doc_id)
            })
        return papers
    except Exception as e:
        logger.warning(f"Failed to list RAG library: {e}")
        return []

def delete_from_library(paper_id: str) -> bool:
    """Deletes a specific paper from the local library."""
    try:
        collection = _get_collection()
        collection.delete(ids=[paper_id])
        return True
    except Exception as e:
        logger.warning(f"Failed to delete {paper_id} from RAG library: {e}")
        return False

def clear_library() -> bool:
    """Deletes all papers from the local library."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        client.delete_collection("paper_library")
        global _collection
        _collection = None
        return True
    except Exception as e:
        logger.warning(f"Failed to clear RAG library: {e}")
        return False


def query_library(query_text: str, for_paper_id: str, exclude_ids: List[str], top_k: int = None) -> List[RAGMatch]:
    """Semantic search over the local library, excluding the papers in the
    current run (we want *prior* work, not the input set matching itself)."""
    top_k = top_k or config.RAG_TOP_K
    try:
        collection = _get_collection()
        # over-fetch a bit since we filter out excluded ids after the fact
        results = collection.query(query_texts=[query_text], n_results=top_k + len(exclude_ids))
    except Exception as e:
        logger.warning(f"RAG library query failed: {e}")
        return []

    matches = []
    ids = results.get("ids", [[]])[0]
    if not ids:
        return matches
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    for doc_id, meta, doc, dist in zip(ids, metadatas, documents, distances):
        if doc_id in exclude_ids:
            continue
        matches.append(
            RAGMatch(
                paper_id=doc_id,
                title=meta.get("title", doc_id),
                snippet=doc[:300],
                source="library",
                for_paper_id=for_paper_id,
                similarity=round(max(0.0, 1 - dist), 3),  # Chroma default = cosine distance
            )
        )
        if len(matches) >= top_k:
            break
    return matches


def search_related_arxiv(paper: PaperData, top_k: int = None) -> List[RAGMatch]:
    """Live arXiv search for papers related to this one. Best-effort: any
    failure (network, rate limit) returns [] rather than raising, so a
    flaky arXiv search never breaks the pipeline."""
    top_k = top_k or config.RAG_TOP_K
    try:
        import arxiv
        from .ingestion import ARXIV_CLIENT

        search = arxiv.Search(
            query=paper.title, max_results=top_k + 1, sort_by=arxiv.SortCriterion.Relevance
        )
        matches = []
        this_id = re.sub(r"v\d+$", "", paper.paper_id)
        for result in ARXIV_CLIENT.results(search):
            if re.sub(r"v\d+$", "", result.get_short_id()) == this_id:
                continue  # skip the paper itself
            matches.append(
                RAGMatch(
                    paper_id=result.get_short_id(),
                    title=result.title,
                    snippet=result.summary[:300].replace("\n", " "),
                    source="arxiv_search",
                    for_paper_id=paper.paper_id,
                    similarity=None,
                )
            )
            if len(matches) >= top_k:
                break
        return matches
    except Exception as e:
        logger.warning(f"Related-paper arXiv search failed for {paper.paper_id}: {e}")
        return []
