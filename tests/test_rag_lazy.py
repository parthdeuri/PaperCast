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
