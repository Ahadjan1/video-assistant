# 🎙️ Waveform: AI Meeting Assistant

Turn any meeting recording (a YouTube link or a local file) into a clean transcript, a summary, action items, key decisions, and a chat interface you can ask follow-up questions in.

Transcription runs locally (Whisper). Summarization and extraction use Google Gemini through LangChain. A RAG pipeline backed by ChromaDB lets you chat with the transcript.

---

## ✨ Features

- **Flexible input:** paste a YouTube URL or upload a local audio/video file
- **Local transcription:** OpenAI Whisper runs on your machine, so the audio itself is never sent to an API (only the transcript text goes to the LLM)
- **English & Urdu support:** Urdu audio is transcribed and translated to English in one pass
- **LLM-powered analysis** (LangChain + Gemini):
  - Auto-generated meeting title
  - Map-reduce summary (handles long transcripts without hitting token limits)
  - Action items with owner and deadline
  - Key decisions
  - Open / unresolved questions
- **Chat with your meeting:** a RAG pipeline (ChromaDB + HuggingFace embeddings) answers questions grounded only in what was actually said
- **Export:** download the transcript or a full report as `.txt`

---

## 🏗️ Architecture

```
YouTube URL / Local File
          │
          ▼
   utils/audio_processing.py     → downloads / converts / chunks audio into WAV pieces
          │
          ▼
   core/transcriber.py           → Whisper transcribes each chunk (English or Urdu→English)
          │
          ▼
     Full transcript (text)
          │
   ┌──────┼───────────────┬──────────────────┐
   ▼      ▼               ▼                  ▼
summarizer.py  extractor.py      vector_store.py + rag_engine.py
(title,        (action items,    (chunks transcript → embeddings →
summary)       decisions,        ChromaDB → retrieval-augmented
               questions)        chat over the transcript)
```

Every stage is a separate module with a single responsibility. The LLM provider lives in one file (`core/llm.py`), so it can be swapped without touching the rest of the pipeline.

---

## 🧰 Tech Stack

| Layer             | Tool                               |
| ----------------- | ---------------------------------- |
| Audio ingestion   | `yt-dlp`, `pydub`, FFmpeg          |
| Transcription     | `openai-whisper` (local)           |
| LLM orchestration | `LangChain` (LCEL)                 |
| LLM provider      | Google Gemini                      |
| Embeddings        | `HuggingFace` (`all-MiniLM-L6-v2`) |
| Vector store      | `ChromaDB`                         |
| UI                | `Streamlit`                        |

---

## 📁 Project Structure

```
.
├── app.py                    # Streamlit web UI (main entry point)
├── main.py                   # CLI entry point
├── core/
│   ├── llm.py                # Gemini LLM setup (rate limit + retries)
│   ├── transcriber.py        # Whisper-based transcription (English + Urdu)
│   ├── summarizer.py         # Title + map-reduce summary
│   ├── extractor.py          # Action items / decisions / open questions
│   ├── vector_store.py       # ChromaDB embedding + retrieval setup
│   └── rag_engine.py         # RAG chain for chatting with the transcript
├── utils/
│   └── audio_processing.py   # YouTube download, format conversion, chunking
├── packages.txt              # System packages for deployment (ffmpeg)
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/Ahadjan1/video-assistant.git
cd video-assistant

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
WHISPER_MODEL=small
```

- `GEMINI_MODEL` is optional. Check Google AI Studio for the model names currently available on the free tier.
- `WHISPER_MODEL` can be `tiny`, `base`, `small`, `medium`, or `large-v3`. `small` is a good default; use `base` on low-RAM machines.

---

## ▶️ Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a URL or upload a file, pick a language, and hit **Process**. Then browse the Summary /
