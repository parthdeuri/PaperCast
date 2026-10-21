"""
Text-to-speech rendering: converts a DebateScript into a single concatenated
audio file using Kokoro TTS via the `kokoro-onnx` package (open-source,
~82M params, runs locally - no API key, no per-request cost, no torch
dependency, works on Python 3.10-3.13).

Unlike the original `kokoro` PyPI package (torch-based, capped at Python
<3.13 and auto-downloads weights from Hugging Face), `kokoro-onnx` needs you
to manually download two files once from GitHub releases - see
MODEL_SETUP_INSTRUCTIONS below.

Output is .wav, not .mp3 - soundfile/libsndfile can't encode mp3, and .wav
needs no extra system dependencies to produce or play. See
`convert_wav_to_mp3()` below if you want an actual .mp3 (requires ffmpeg).
"""
import os
import numpy as np

from .models import DebateScript

SAMPLE_RATE = 24000
SILENCE_BETWEEN_LINES_S = 0.4

# One Kokoro voice per persona, so each speaker is audibly distinct.
# Full voice list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
VOICE_MAP_KOKORO = {
    "Host": "af_heart",       # warm, neutral - moderator
    "Academic": "am_michael",  # energetic male voice
    "Skeptic": "bm_george",    # deeper British male voice - contrast with Academic
}

# Edge TTS voices
VOICE_MAP_EDGE = {
    "Host": "en-US-AriaNeural",
    "Academic": "en-US-GuyNeural",
    "Skeptic": "en-GB-RyanNeural",
}

MODEL_SETUP_INSTRUCTIONS = """
Before using audio generation, download these two files (once) from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
and place them in your project root (or pass their paths explicitly):

  kokoro-v1.0.int8.onnx   (~88MB, quantized - recommended, good quality/size tradeoff)
  voices-v1.0.bin         (~27MB)

Or via command line:
  curl -L -o kokoro-v1.0.int8.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
  curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
"""


def render_script_to_audio(
    script: DebateScript,
    output_path: str,
    model_path: str = "kokoro-v1.0.int8.onnx",
    voices_path: str = "voices-v1.0.bin",
    tts_engine: str = "kokoro",
    on_line=None,
) -> str:
    """
    Renders every line in `script` to speech and concatenates into one file
    at `output_path` (.wav or .mp3 depending on engine). Returns the output path.

    `on_line(done, total)`, if given, is called after each line - the
    dashboard uses it for a progress bar, which is also where a Stop click
    can interrupt the render. The output file is only written at the end,
    so a stopped render never leaves a partial episode behind.
    """
    total = len(script.lines)
    if not script.lines:
        raise ValueError("script.lines is empty - nothing to render.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if tts_engine == "edge_tts":
        import asyncio
        import edge_tts
        from pydub import AudioSegment
        
        async def render_edge():
            combined = AudioSegment.empty()
            silence = AudioSegment.silent(duration=int(SILENCE_BETWEEN_LINES_S * 1000))
            for i, line in enumerate(script.lines):
                voice = VOICE_MAP_EDGE.get(line.speaker, "en-US-AriaNeural")
                communicate = edge_tts.Communicate(line.text, voice)
                temp_mp3 = f"{output_path}_temp_{i}.mp3"
                try:
                    await communicate.save(temp_mp3)
                    segment = AudioSegment.from_mp3(temp_mp3)
                finally:
                    if os.path.exists(temp_mp3):
                        os.remove(temp_mp3)
                combined += segment + silence
                if on_line:
                    on_line(i + 1, total)
            
            # Export as wav since dashboard expects wav
            combined.export(output_path, format="wav")
            
        asyncio.run(render_edge())
        return output_path

    # Kokoro fallback
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        raise FileNotFoundError(
            f"Model files not found ({model_path}, {voices_path}).\n{MODEL_SETUP_INSTRUCTIONS}"
        )

    # Imported lazily, and only after validating input/files exist.
    from kokoro_onnx import Kokoro
    import soundfile as sf

    kokoro = Kokoro(model_path, voices_path)
    silence = np.zeros(int(SAMPLE_RATE * SILENCE_BETWEEN_LINES_S), dtype=np.float32)

    segments = []
    for i, line in enumerate(script.lines):
        voice = VOICE_MAP_KOKORO.get(line.speaker, "af_heart")
        try:
            samples, sr = kokoro.create(line.text, voice=voice, speed=1.0, lang="en-us", trim=False)
        except Exception as e:
            print(f"[tts] WARNING: failed to render line {i} ({line.speaker}): {e}")
            samples = []

        if len(samples) == 0:
            print(f"[tts] WARNING: 0 samples produced for line {i} ({line.speaker}): {line.text[:50]!r}")
        else:
            segments.append(samples)
            segments.append(silence)
        if on_line:
            on_line(i + 1, total)

    if not segments:
        raise ValueError("No audio was generated for any line - check the model files are valid.")

    full_audio = np.concatenate(segments)
    sf.write(output_path, full_audio, SAMPLE_RATE)
    return output_path


def convert_wav_to_mp3(wav_path: str, mp3_path: str) -> str:
    """Optional: convert the .wav to .mp3. Requires ffmpeg installed and on
    PATH (on Windows: download from ffmpeg.org or `winget install ffmpeg`).
    Not called automatically - .wav plays fine as-is in any media player."""
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate="192k")
    return mp3_path
