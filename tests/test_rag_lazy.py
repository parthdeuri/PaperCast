"""Listing the RAG library must not load the embedding model.

Runs in a fresh Python process, because the check is "was sentence_transformers
ever imported" and other tests in this process could have imported it.
"""
import os
import subprocess
import sys
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_listing_library_does_not_load_embedding_model(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        import src.config as config
        config.CHROMA_DIR = {str(tmp_path)!r}
        from src import rag

        rag._get_collection()   # creates the collection and saves its config
        rag._collection = None  # force a reopen, which rebuilds the embedding
                                # function from that saved config
        assert rag.list_library() == []
        assert "sentence_transformers" not in sys.modules, "listing loaded the embedding model"
        assert "torch" not in sys.modules, "listing imported torch"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": PROJECT_ROOT}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_path_ids_are_renamed_keeping_embeddings(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        import src.config as config
        config.CHROMA_DIR = {str(tmp_path)!r}
        from src import rag

        col = rag._get_collection()
        path_id = "C:/Temp/old upload.pdf"
        col.add(ids=[path_id, "2306.07691"], embeddings=[[0.1] * 384, [0.2] * 384],
                documents=["old doc", "arxiv doc"],
                metadatas=[{{"title": "Old", "paper_id": path_id}}, {{"title": "Arxiv", "paper_id": "2306.07691"}}])

        rag._collection = None          # reopen: the migration runs here
        ids = sorted(p["id"] for p in rag.list_library())
        assert "2306.07691" in ids and path_id not in ids, ids
        [new_id] = [i for i in ids if i.startswith("pdf-legacy-")]
        got = rag._get_collection().get(ids=[new_id], include=["embeddings", "documents", "metadatas"])
        assert got["documents"] == ["old doc"]
        assert got["metadatas"][0]["paper_id"] == new_id and got["metadatas"][0]["title"] == "Old"
        assert abs(float(got["embeddings"][0][0]) - 0.1) < 1e-6
        assert "sentence_transformers" not in sys.modules
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": PROJECT_ROOT}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout
