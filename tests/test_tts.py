"""Audio rendering: the per-line progress callback, and stopping mid-render.
Uses a fake Kokoro model, so no model files or audio libraries are exercised."""
import sys
import types

import numpy as np
import pytest

from src.models import DebateScript, DialogueLine
from src.tts import render_script_to_audio


class FakeKokoro:
    def __init__(self, model_path, voices_path):
        pass

    def create(self, text, voice, speed, lang, trim):
        return np.ones(240, dtype=np.float32), 24000


@pytest.fixture
def kokoro_files(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    model, voices = tmp_path / "model.onnx", tmp_path / "voices.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return str(model), str(voices)


def script(n):
    return DebateScript(lines=[DialogueLine(speaker="Host", text=f"Line {i}.") for i in range(n)])


def test_on_line_reports_each_line(kokoro_files, tmp_path):
    model, voices = kokoro_files
    calls = []
    out = tmp_path / "ep.wav"
    render_script_to_audio(script(3), str(out), model_path=model, voices_path=voices,
                           on_line=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert out.exists()


def test_stopping_mid_render_writes_no_file(kokoro_files, tmp_path):
    # Streamlit stops a script with a BaseException raised from the next st.*
    # call - here, inside the progress callback.
    class Stop(BaseException):
        pass

    def on_line(done, total):
        if done == 2:
            raise Stop()

    model, voices = kokoro_files
    out = tmp_path / "ep.wav"
    with pytest.raises(Stop):
        render_script_to_audio(script(4), str(out), model_path=model, voices_path=voices, on_line=on_line)
    assert not out.exists()
