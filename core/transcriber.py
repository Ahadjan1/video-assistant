import os
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str, task: str = "transcribe", language: str = None) -> str:
    """
    task="transcribe" -> text in the spoken language
    task="translate"  -> auto-translated to English
    language          -> force the spoken language instead of auto-detecting
                         (auto-detect is often wrong for Urdu)
    """
    model = load_model()
    result = model.transcribe(
        chunk_path,
        task=task,
        language=language,
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=False,
    )
    print(f"Chunk done ({len(result['text'])} characters)")
    return result["text"]


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    - english -> transcribe as-is
    - urdu    -> force language="ur", translate directly to English in one pass
    """
    if language.lower() == "urdu":
        return transcribe_chunk_whisper(chunk_path, task="translate", language="ur")
    return transcribe_chunk_whisper(chunk_path, task="transcribe", language="en")


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    print(f"Using Whisper ({WHISPER_MODEL}) for {language} transcription.")
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "
    print("Transcription complete.")
    return full_transcript.strip()