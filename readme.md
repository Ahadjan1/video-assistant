# 🎙️ AI Meeting Assistant

Turn any meeting recording — a YouTube link or a local file — into a clean transcript, a summary, action items, key decisions, and a chat interface you can ask follow-up questions in.

Runs transcription locally (Whisper), summarization/extraction through Mistral (via LangChain), and lets you "chat" with the transcript using a RAG pipeline backed by ChromaDB.

---

## ✨ Features

- **Flexible input** — paste a YouTube URL or upload a local audio/video file
- **Local transcription** — OpenAI Whisper runs on your machine, no audio leaves your system
- **English & Urdu support** — Urdu audio is transcribed and translated to English in one pass
- **LLM-powered analysis** (via LangChain + Mistral):
  - Auto-generated meeting title
  - Map-reduce summary (handles long transcripts without hitting token limits)
  - Action items with owner + deadline
  - Key decisions
  - Open / unresolved questions
- **Chat with your meeting** — a RAG pipeline (ChromaDB + HuggingFace embeddings) lets you ask natural-language questions about the transcript and get answers grounded only in what was actually said
- **Export** — download the transcript or a full report as `.txt`
- **Two interfaces** — a CLI (`main.py`) for quick runs, and a full Streamlit web UI (`app.py`)

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

Every stage is a separate module with a single responsibility — swap the transcription engine, the LLM provider, or the vector store without touching the rest of the pipeline.

---

## 🧰 Tech Stack

| Layer             | Tool                               |
| ----------------- | ---------------------------------- |
| Audio ingestion   | `yt-dlp`, `pydub`, FFmpeg          |
| Transcription     | `openai-whisper` (local)           |
| LLM orchestration | `LangChain` (LCEL)                 |
| LLM provider      | `Mistral` (`mistral-small-latest`) |
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
│   ├── transcriber.py        # Whisper-based transcription (English + Urdu)
│   ├── summarizer.py         # Title + map-reduce summary
│   ├── extractor.py          # Action items / decisions / open questions
│   ├── vector_store.py       # ChromaDB embedding + retrieval setup
│   └── rag_engine.py         # RAG chain for chatting with the transcript
├── utils/
│   └── audio_processing.py   # YouTube download, format conversion, chunking
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11 (Whisper's dependencies don't support 3.12+/3.14 yet)
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your PATH
- A free [Mistral API key](https://console.mistral.ai/)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

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
MISTRAL_API_KEY=your_mistral_api_key_here
WHISPER_MODEL=small          # tiny / base / small / medium / large-v3
```

> `small` is recommended on machines with limited RAM. Use `medium` or `large-v3` for better accuracy if you have the resources.

---

## ▶️ Usage

### Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a URL or upload a file, pick a language, and hit **Process**. Once done, browse the Summary / Action Items / Decisions / Questions / Transcript tabs, download a report, or chat with the meeting directly.

### CLI

```bash
python main.py
```

You'll be prompted for a source (URL or file path) and a language (`english` / `urdu`). After processing, results print to the terminal and you can chat with the transcript right there.

---

## 🗺️ Roadmap / Known Limitations

- Long meetings (2+ hours) can take a while to transcribe on CPU — a GPU speeds this up significantly
- Action item / decision / question extraction currently processes the full transcript in one call; a map-reduce pass (like the summarizer) would help on very long meetings
- No authentication / multi-user support — built for local/single-user use

---

## 📄 License

[Add your chosen license here, e.g. MIT]

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for local transcription
- [LangChain](https://www.langchain.com/) for chain orchestration
- [Mistral AI](https://mistral.ai/) for the LLM
- [ChromaDB](https://www.trychroma.com/) for the vector store
