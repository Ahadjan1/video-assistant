import streamlit as st
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError

from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Waveform — AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0D0F16;
        --surface: #171A24;
        --surface-2: #1F2330;
        --surface-3: #262B3A;
        --border: #2C3142;
        --border-soft: #21252F;
        --text: #ECEEF3;
        --muted: #8B93A7;
        --amber: #F2A65A;
        --amber-soft: rgba(242,166,90,0.14);
        --cyan: #5FA8D3;
        --cyan-soft: rgba(95,168,211,0.14);
        --green: #6EBF8B;
        --green-soft: rgba(110,191,139,0.14);
        --violet: #A78BFA;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% -10%, rgba(242,166,90,0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 0%, rgba(95,168,211,0.05) 0%, transparent 40%),
            var(--bg);
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ---- Hero ---- */
    .wf-hero {
        position: relative; overflow: hidden;
        display: flex; align-items: center; gap: 22px;
        padding: 30px 34px; border-radius: 18px;
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--border);
        margin-bottom: 26px;
    }
    .wf-hero::after {
        content: ""; position: absolute; top: -40%; right: -10%;
        width: 260px; height: 260px; border-radius: 50%;
        background: radial-gradient(circle, rgba(242,166,90,0.18) 0%, transparent 70%);
    }
    .wf-hero-text h1 {
        margin: 0; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #ECEEF3 0%, var(--amber) 120%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .wf-hero-text p { margin: 6px 0 0 0; color: var(--muted); font-size: 0.96rem; }
    .wf-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        color: var(--amber); letter-spacing: 0.14em; text-transform: uppercase;
        margin-bottom: 6px; display: block;
    }
    .wf-bars { display: flex; align-items: flex-end; gap: 3px; height: 44px; flex-shrink: 0; z-index: 1; }
    .wf-bar { width: 4px; background: linear-gradient(180deg, var(--amber), var(--cyan)); border-radius: 2px; opacity: 0.9; }

    /* ---- Sidebar pipeline (compact, current-step-only) ---- */
    .wf-pipeline-box {
        background: var(--surface); border: 1px solid var(--cyan);
        border-radius: 12px; padding: 14px 16px; margin-bottom: 4px;
        box-shadow: 0 0 0 1px rgba(95,168,211,0.08), 0 0 18px rgba(95,168,211,0.12);
    }
    .wf-pipeline-box .step-count {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        color: var(--cyan); letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
    }
    .wf-pipeline-box .step-count .pulse {
        width: 7px; height: 7px; border-radius: 50%; background: var(--cyan);
        box-shadow: 0 0 8px var(--cyan); animation: wf-pulse 1.4s ease-in-out infinite;
    }
    @keyframes wf-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .wf-pipeline-box .step-label { font-size: 0.88rem; color: var(--text); font-weight: 500; }
    .wf-pipeline-box.complete { border-color: var(--green); box-shadow: 0 0 0 1px rgba(110,191,139,0.08), 0 0 18px rgba(110,191,139,0.12); }
    .wf-pipeline-box.complete .step-count { color: var(--green); }
    .wf-pipeline-box.complete .step-count .pulse { background: var(--green); box-shadow: 0 0 8px var(--green); animation: none; }

    /* ---- Pipeline step cards (legacy full-list, kept for reference) ---- */
    .wf-step {
        display: flex; align-items: center; gap: 12px;
        background: var(--surface); border: 1px solid var(--border-soft);
        border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
        font-size: 0.85rem; transition: all 0.25s ease;
    }
    .wf-step.active { border-color: var(--cyan); background: var(--cyan-soft); }
    .wf-step.done { border-color: var(--green); }
    .wf-step .dot {
        width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        background: var(--border);
    }
    .wf-step.active .dot { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
    .wf-step.done .dot { background: var(--green); }
    .wf-step .label { flex: 1; color: var(--text); }
    .wf-step .tag {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        color: var(--muted); text-transform: uppercase;
    }
    .wf-step.done .tag { color: var(--green); }
    .wf-step.active .tag { color: var(--cyan); }

    /* ---- Metric cards ---- */
    .wf-metric {
        background: linear-gradient(160deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--border); border-radius: 14px;
        padding: 18px 20px; text-align: left; position: relative; overflow: hidden;
    }
    .wf-metric .icon { font-size: 1.1rem; margin-bottom: 6px; opacity: 0.9; }
    .wf-metric .val { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text); }
    .wf-metric .label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

    /* ---- Section cards ---- */
    .wf-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 22px 26px; line-height: 1.7;
    }

    /* ---- Sidebar history ---- */
    .wf-history-item {
        padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border);
        background: var(--surface); margin-bottom: 8px; font-size: 0.85rem;
    }
    .wf-history-item.active { border-color: var(--amber); background: var(--amber-soft); }
    .wf-history-item .title { color: var(--text); font-weight: 600; }
    .wf-history-item .meta { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; margin-top: 2px; }

    /* ---- Buttons ---- */
    .stButton>button {
        border-radius: 10px; border: 1px solid var(--border);
        font-family: 'Inter', sans-serif; font-weight: 500;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, var(--amber), #f5b878); color: #1a1400; border: none; font-weight: 700;
    }
    .stButton>button[kind="primary"]:hover { filter: brightness(1.06); }
    div[data-testid="stHorizontalBlock"] .stButton>button {
        background: var(--surface-2); font-size: 0.8rem; padding: 4px 12px;
    }

    .wf-divider { border-top: 1px solid var(--border); margin: 30px 0; }

    /* ---- Tabs polish ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
        color: var(--muted); padding: 8px 4px;
    }
    .stTabs [aria-selected="true"] { color: var(--amber) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def waveform_bars(n=16, seed=None):
    import random
    rnd = random.Random(seed or 42)
    heights = [rnd.randint(10, 40) for _ in range(n)]
    bars = "".join(f'<div class="wf-bar" style="height:{h}px;"></div>' for h in heights)
    return f'<div class="wf-bars">{bars}</div>'


PIPELINE_STEPS = [
    {"key": "ingest", "label": "Ingesting audio (download / convert / chunk)"},
    {"key": "transcribe", "label": "Transcribing with Whisper"},
    {"key": "title", "label": "Generating title"},
    {"key": "summary", "label": "Summarizing (map-reduce)"},
    {"key": "actions", "label": "Extracting action items"},
    {"key": "decisions", "label": "Extracting key decisions"},
    {"key": "questions", "label": "Extracting open questions"},
    {"key": "rag", "label": "Building retrieval index (RAG)"},
]


def render_sidebar_pipeline(container, current_idx, total, label, complete=False):
    with container.container():
        if complete:
            st.markdown(
                f"""<div class="wf-pipeline-box complete">
                        <div class="step-count"><div class="pulse"></div>DONE · {total}/{total}</div>
                        <div class="step-label">Pipeline complete</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="wf-pipeline-box">
                        <div class="step-count"><div class="pulse"></div>STEP {current_idx}/{total}</div>
                        <div class="step-label">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "active_idx" not in st.session_state:
    st.session_state.active_idx = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="wf-eyebrow">Settings</span>', unsafe_allow_html=True)
    retrieval_k = st.slider("Chat context depth (chunks retrieved)", 2, 8, 4)
    st.caption("Higher = more transcript context per answer, but slower.")

    st.markdown('<div class="wf-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="wf-eyebrow">Pipeline</span>', unsafe_allow_html=True)
    sidebar_pipeline_slot = st.empty()
    sidebar_pipeline_slot.caption("Idle — nothing running.")

    st.markdown('<div class="wf-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="wf-eyebrow">Meeting history</span>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.caption("No meetings processed yet this session.")
    else:
        for i, item in enumerate(st.session_state.history):
            is_active = i == st.session_state.active_idx
            css = "active" if is_active else ""
            st.markdown(
                f"""<div class="wf-history-item {css}">
                        <div class="title">{item['title']}</div>
                        <div class="meta">{item['chunk_count']} chunk(s) · {item['language']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"open_{i}", use_container_width=True):
                st.session_state.active_idx = i
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="wf-hero">
        {waveform_bars(18)}
        <div class="wf-hero-text">
         <span class="wf-eyebrow">Local Whisper · Gemini · RAG</span>
            <h1>AI Meeting Assistant</h1>
            <p>Turn a recording into a transcript, summary, action items — and a conversation.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# INPUT ROW
# ──────────────────────────────────────────────────────────────────────────
input_col, lang_col, btn_col = st.columns([3, 1, 1])
with input_col:
    source = st.text_input(
        "Source", placeholder="Paste a YouTube URL, e.g. https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with lang_col:
    language = st.selectbox("Language", ["english", "urdu"], label_visibility="collapsed")
with btn_col:
    process_clicked = st.button("Process ▶", type="primary", use_container_width=True)

with st.expander("Or upload a local file instead"):
    uploaded_file = st.file_uploader(
        "Audio or video file", type=["mp3", "wav", "mp4", "m4a"], label_visibility="collapsed"
    )

# ──────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION — live step cards
# ──────────────────────────────────────────────────────────────────────────
if process_clicked:
    final_source = None
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        final_source = temp_path
    elif source.strip():
        final_source = source.strip()

    if not final_source:
        st.warning("Add a URL, path, or upload a file first.")
    else:
        statuses = {s["key"]: "waiting" for s in PIPELINE_STEPS}
        total_steps = len(PIPELINE_STEPS)

        def step(key):
            idx = next(i for i, s in enumerate(PIPELINE_STEPS) if s["key"] == key)
            label = PIPELINE_STEPS[idx]["label"]
            render_sidebar_pipeline(sidebar_pipeline_slot, idx + 1, total_steps, label)

        def done(key):
            pass  # sidebar advances to the next step() call automatically

              step("ingest")
        try:
            chunks = process_input(final_source)
        except DownloadError:
            st.error(
                "YouTube blocks downloads from cloud servers, so this link can't be "
                "processed here. Please upload an audio/video file instead "
                "(use 'Or upload a local file instead' above)."
            )
            st.stop()

        step("transcribe")
        transcript = transcribe_all(chunks, language=language)

        step("title")
        title = generate_title(transcript)

        step("summary")
        summary = summarize(transcript)

        step("actions")
        action_items = extract_action_items(transcript)

        step("decisions")
        decisions = extract_key_decisions(transcript)

        step("questions")
        questions = extract_questions(transcript)

        step("rag")
        rag_chain = build_rag_chain(transcript, k=retrieval_k)

        render_sidebar_pipeline(sidebar_pipeline_slot, total_steps, total_steps, "", complete=True)

        result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "language": language,
            "chunk_count": len(chunks),
            "word_count": len(transcript.split()),
        }
        st.session_state.history.append(result)
        new_idx = len(st.session_state.history) - 1
        st.session_state.active_idx = new_idx
        st.session_state.chat_history[new_idx] = []
        st.balloons()
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ──────────────────────────────────────────────────────────────────────────
if st.session_state.active_idx is not None and st.session_state.history:
    result = st.session_state.history[st.session_state.active_idx]
    idx = st.session_state.active_idx

    st.markdown('<div class="wf-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### {result['title']}")

    m1, m2, m3, m4 = st.columns(4)
    est_minutes = result["chunk_count"] * 10
    for col, icon, val, label in [
        (m1, "⏱️", f"~{est_minutes} min", "Est. Duration"),
        (m2, "📝", result["word_count"], "Words Transcribed"),
        (m3, "🧩", result["chunk_count"], "Audio Chunks"),
        (m4, "🌐", result["language"].capitalize(), "Language"),
    ]:
        col.markdown(
            f'<div class="wf-metric"><div class="icon">{icon}</div>'
            f'<div class="val">{val}</div><div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript"]
    )
    with tab1:
        st.markdown(f'<div class="wf-card">{result["summary"]}</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown(f'<div class="wf-card">{result["action_items"]}</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown(f'<div class="wf-card">{result["key_decisions"]}</div>', unsafe_allow_html=True)
    with tab4:
        st.markdown(f'<div class="wf-card">{result["open_questions"]}</div>', unsafe_allow_html=True)
    with tab5:
        st.text_area("Full transcript", result["transcript"], height=350, label_visibility="collapsed")

    st.write("")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇ Transcript (.txt)", data=result["transcript"],
            file_name=f"{result['title']}_transcript.txt", mime="text/plain",
            use_container_width=True,
        )
    with d2:
        report = (
            f"Title: {result['title']}\n\n"
            f"SUMMARY:\n{result['summary']}\n\n"
            f"ACTION ITEMS:\n{result['action_items']}\n\n"
            f"KEY DECISIONS:\n{result['key_decisions']}\n\n"
            f"OPEN QUESTIONS:\n{result['open_questions']}\n"
        )
        st.download_button(
            "⬇ Full Report (.txt)", data=report,
            file_name=f"{result['title']}_report.txt", mime="text/plain",
            use_container_width=True,
        )

    # ── Chat (RAG) ──────────────────────────────────────────────────────
    st.markdown('<div class="wf-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="wf-eyebrow">Chat with this meeting</span>', unsafe_allow_html=True)

    suggestions = ["What were the action items?", "Summarize the key decisions.", "Any unresolved questions?"]
    chip_cols = st.columns(len(suggestions))
    chip_clicked = None
    for c, s in zip(chip_cols, suggestions):
        if c.button(s, key=f"chip_{s}_{idx}"):
            chip_clicked = s

    for role, msg in st.session_state.chat_history.get(idx, []):
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask something about the meeting...") or chip_clicked
    if question:
        st.session_state.chat_history[idx].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], question)
                st.markdown(answer)
        st.session_state.chat_history[idx].append(("assistant", answer))
else:
    st.info("Process a meeting above to see results here.")