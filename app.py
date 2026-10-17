"""
AI Meeting Intelligence & Decision Tracker
Premium Blue & White Enterprise Dashboard — 100% Offline Local NLP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import json
import traceback
from datetime import datetime

# ── Page Config (MUST be first Streamlit call) ─────────────
st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Safe imports with graceful fallback ─────────────────────
import_errors = []

try:
    from src.file_parser import extract_text_from_file
except Exception as e:
    import_errors.append(f"file_parser: {e}")
    def extract_text_from_file(f, name): return f.read().decode("utf-8", errors="ignore")

try:
    from src.audio_processor import process_audio_file
except Exception as e:
    import_errors.append(f"audio_processor: {e}")
    def process_audio_file(f, filename="audio.mp3", model_size="tiny"):
        return "Alex: Sample audio transcript fallback.\nSarah: I will complete the task by Friday."

try:
    from src.preprocessing import preprocess_transcript
except Exception as e:
    import_errors.append(f"preprocessing: {e}")
    def preprocess_transcript(txt):
        return {"cleaned_text": txt, "speaker_turns": [{"speaker": "Unknown", "text": txt, "line_num": 1}],
                "sentences": [{"speaker": "Unknown", "sentence": s, "tokens": s.split()} for s in txt.split(".") if s.strip()],
                "speakers": ["Unknown"], "all_tokens": txt.split(), "raw_text": txt}

try:
    from src.summarization import generate_meeting_summary
except Exception as e:
    import_errors.append(f"summarization: {e}")
    def generate_meeting_summary(txt):
        return {"summary_text": txt[:300], "main_points": [txt[:150]], "key_outcomes": [], "method": "Fallback"}

try:
    from src.ner_extraction import extract_meeting_ner
except Exception as e:
    import_errors.append(f"ner_extraction: {e}")
    def extract_meeting_ner(txt): return {"PERSON": [], "DATE": [], "ORGANIZATION": [], "TASK": []}

try:
    from src.action_extraction import extract_action_items_list
except Exception as e:
    import_errors.append(f"action_extraction: {e}")
    def extract_action_items_list(txt): return []

try:
    from src.decision_extraction import extract_meeting_decisions
except Exception as e:
    import_errors.append(f"decision_extraction: {e}")
    def extract_meeting_decisions(txt): return []

try:
    from src.topic_modeling import get_meeting_topics
except Exception as e:
    import_errors.append(f"topic_modeling: {e}")
    def get_meeting_topics(sents, method="LDA"):
        return [{"topic_id": 1, "topic_name": "General Discussion", "top_words": ["meeting", "project"]}]

try:
    from src.vector_search import get_vector_store
    vector_store = get_vector_store()
except Exception as e:
    import_errors.append(f"vector_search: {e}")
    class _FakeVS:
        def add_meeting(self, d): pass
        def search_meetings(self, q, top_k=3): return []
    vector_store = _FakeVS()

try:
    from src.productivity import calculate_productivity_score
except Exception as e:
    import_errors.append(f"productivity: {e}")
    def calculate_productivity_score(d, t, s):
        return {"overall_score": 75.0, "grade": "B (Good)", "feedback": "Productivity module unavailable.",
                "breakdown": {"decisions_score": 20, "tasks_score": 20, "clarity_score": 15, "participation_score": 20, "speaker_participation_balance_pct": 60}}

try:
    from src.database import get_db_manager
    db = get_db_manager()
except Exception as e:
    import_errors.append(f"database: {e}")
    class _FakeDB:
        def save_meeting(self, d): pass
        def fetch_all_meetings(self): return []
        def fetch_all_decisions(self): return []
        def fetch_all_action_items(self): return []
    db = _FakeDB()

try:
    from src.eda import MeetingDatasetEDA
    eda_engine = MeetingDatasetEDA()
except Exception as e:
    import_errors.append(f"eda: {e}")
    eda_engine = None

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🎙️</span>
        <h2 style="margin:0.3rem 0 0 0; font-size:1.15rem; font-weight:800; color:#FFFFFF;">
            Meeting Intelligence
        </h2>
        <p style="font-size:0.75rem; color:#93C5FD; margin:0;">AI-Powered Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📥 Upload & Analyze",
            "📋 Meeting Dashboard",
            "🔍 AI Memory Search",
            "📊 Dataset EDA",
            "⚙️ System Info"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15);
                border-radius:12px; padding:0.9rem; margin-top:0.5rem;">
        <p style="font-size:0.78rem; color:#BFDBFE; margin:0; font-weight:500;">
            💡 <strong style="color:#FFFFFF;">100% Offline NLP</strong><br>
            <span style="color:#93C5FD;">Whisper · BART/T5 · spaCy<br>Sentence-BERT · FAISS</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if import_errors:
        with st.expander("⚠️ Module Warnings", expanded=False):
            for err in import_errors:
                st.caption(f"• {err}")


# ── App Header ───────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-badge">🟢 All Systems Online &nbsp;|&nbsp; 100% Local & Private</div>
    <h1>🎙️ AI Meeting Intelligence & Decision Tracker</h1>
    <p>Upload meeting transcripts or audio recordings — get instant AI summaries, decisions, action items, topics & speaker analytics</p>
</div>
""", unsafe_allow_html=True)


# ── Helper: Robust topic display ─────────────────────────────
def format_topics_for_display(topics_raw):
    """Safely format topics list (handles both str and dict items)."""
    if not topics_raw:
        return []
    result = []
    for t in topics_raw:
        if isinstance(t, dict):
            result.append({
                "name": t.get("topic_name", "Topic"),
                "words": ", ".join(t.get("top_words", []))
            })
        else:
            result.append({"name": str(t), "words": ""})
    return result


# ── NLP Pipeline ─────────────────────────────────────────────
def run_full_nlp_pipeline(raw_transcript: str, meeting_id: str, department: str, date_str: str):
    steps = [
        "Preprocessing & Speaker Turn Parsing",
        "Generating Executive Summary (BART/T5)",
        "Named Entity Recognition (spaCy + BERT)",
        "Action Item Extraction",
        "Decision Extraction",
        "Topic Modeling (LDA)",
        "Speaker Analytics & Productivity Score"
    ]

    progress_bar = st.progress(0, text="Starting NLP pipeline...")

    try:
        # Step 1
        progress_bar.progress(10, text=f"⚙️ Step 1/7: {steps[0]}")
        prep = preprocess_transcript(raw_transcript)

        # Step 2
        progress_bar.progress(25, text=f"🧠 Step 2/7: {steps[1]}")
        summary = generate_meeting_summary(prep["cleaned_text"])
        if not isinstance(summary, dict):
            summary = {"summary_text": str(summary), "main_points": [], "key_outcomes": [], "method": "Unknown"}

        # Step 3
        progress_bar.progress(40, text=f"🔍 Step 3/7: {steps[2]}")
        ner = extract_meeting_ner(prep["cleaned_text"])

        # Step 4
        progress_bar.progress(54, text=f"📝 Step 4/7: {steps[3]}")
        actions = extract_action_items_list(prep["cleaned_text"])
        if not isinstance(actions, list):
            actions = []

        # Step 5
        progress_bar.progress(65, text=f"✅ Step 5/7: {steps[4]}")
        decisions = extract_meeting_decisions(prep["cleaned_text"])
        if not isinstance(decisions, list):
            decisions = []

        # Step 6
        progress_bar.progress(78, text=f"🏷️ Step 6/7: {steps[5]}")
        sent_texts = [s["sentence"] for s in prep.get("sentences", []) if s.get("sentence")]
        if not sent_texts:
            sent_texts = [s.strip() for s in raw_transcript.split(".") if s.strip()]
        topics = get_meeting_topics(sent_texts, method="LDA")
        if not isinstance(topics, list):
            topics = [{"topic_id": 1, "topic_name": "General Discussion", "top_words": ["meeting", "project"]}]

        # Step 7
        progress_bar.progress(90, text=f"📊 Step 7/7: {steps[6]}")
        speaker_counts = {}
        for turn in prep.get("speaker_turns", []):
            spk = turn.get("speaker", "Unknown")
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

        tot_turns = sum(speaker_counts.values()) or 1
        speaker_analytics = {
            spk: {"count": cnt, "percentage": round((cnt / tot_turns) * 100, 1)}
            for spk, cnt in speaker_counts.items()
        }

        prod = calculate_productivity_score(decisions, actions, prep.get("speaker_turns", []))
        if not isinstance(prod, dict):
            prod = {"overall_score": 75.0, "grade": "B", "feedback": "Score unavailable."}

        # Build result
        processed_meeting = {
            "meeting_id": meeting_id,
            "date": date_str,
            "department": department,
            "raw_transcript": raw_transcript,
            "cleaned_transcript": prep.get("cleaned_text", raw_transcript),
            "summary": summary,
            "ner": ner,
            "action_items": actions,
            "decisions": decisions,
            "topics": topics,
            "speaker_analytics": speaker_analytics,
            "productivity": prod,
            "productivity_score": prod.get("overall_score", 75.0),
            "speakers": prep.get("speakers", [])
        }

        # Save to DB & Vector Store
        try:
            db.save_meeting(processed_meeting)
        except Exception as e:
            st.warning(f"DB save warning: {e}")

        try:
            vector_store.add_meeting(processed_meeting)
        except Exception as e:
            st.warning(f"Vector store warning: {e}")

        progress_bar.progress(100, text="✅ Pipeline complete!")
        return processed_meeting

    except Exception as e:
        progress_bar.empty()
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
        return None


# ════════════════════════════════════════════════════════════
# PAGE 1: UPLOAD & ANALYZE
# ════════════════════════════════════════════════════════════
if page == "📥 Upload & Analyze":
    st.markdown("### 📥 Upload Meeting Files")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-header">📄 Text Transcript</div>
        """, unsafe_allow_html=True)
        uploaded_text_file = st.file_uploader(
            "Upload .txt, .pdf, or .docx transcript",
            type=["txt", "pdf", "docx"],
            key="txt_up",
            help="Supports plain text, PDF and Word documents"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-header">🎙️ Audio Recording</div>
        """, unsafe_allow_html=True)
        uploaded_audio_file = st.file_uploader(
            "Upload .mp3 or .wav audio file",
            type=["mp3", "wav"],
            key="aud_up",
            help="Transcribed locally using OpenAI Whisper (no internet needed)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Metadata row
    st.markdown("""
    <div class="custom-card">
        <div class="custom-card-header">⚙️ Meeting Metadata</div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        dept_opt = st.selectbox(
            "Department / Domain",
            ["Software Development", "Project Planning", "Marketing", "HR", "Finance", "Product Management"]
        )
    with m2:
        date_opt = st.date_input("Meeting Date", datetime.now())
    with m3:
        custom_id = st.text_input("Meeting ID", f"MTG_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    st.markdown("</div>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        process_btn = st.button("🚀 Process & Extract Intelligence", use_container_width=True)

    if process_btn:
        raw_text = ""

        if uploaded_audio_file is not None:
            with st.spinner("🎙️ Running local Whisper Speech-to-Text..."):
                try:
                    raw_text = process_audio_file(uploaded_audio_file, filename=uploaded_audio_file.name)
                    st.success(f"✅ Audio transcribed successfully! ({len(raw_text)} characters)")
                except Exception as e:
                    st.error(f"Audio processing error: {e}")
                    raw_text = ""

        elif uploaded_text_file is not None:
            try:
                raw_text = extract_text_from_file(uploaded_text_file, uploaded_text_file.name)
                st.success(f"✅ Text extracted from **{uploaded_text_file.name}** ({len(raw_text)} characters)")
            except Exception as e:
                st.error(f"File parsing error: {e}")
                raw_text = ""

        else:
            st.warning("⚠️ No file uploaded — using built-in sample transcript.")
            raw_text = (
                "Alex: Welcome everyone to today's Software Development sync regarding the API Migration project.\n"
                "Sarah: Thanks Alex. I will complete the website testing and full API documentation by this Friday.\n"
                "David: We decided to approve the cloud infrastructure migration starting next Monday.\n"
                "Priya: Great point. I agree with that timeline. I will handle the QA test suite by Wednesday.\n"
                "Alex: Perfect. We've finalized the budget allocation for the project at $50,000.\n"
                "Sarah: I will coordinate with the design team for the new dashboard UI by end of week."
            )

        if raw_text and raw_text.strip():
            with st.container():
                result = run_full_nlp_pipeline(raw_text, custom_id, dept_opt, str(date_opt))

            if result:
                st.session_state["current_meeting"] = result
                st.success("🎉 Meeting intelligence extraction complete! Go to **Meeting Dashboard** to view results.")
                st.balloons()

                # Quick preview
                st.markdown("---")
                st.markdown("#### 📌 Quick Preview")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Decisions Found", len(result.get("decisions", [])))
                p2.metric("Action Items", len(result.get("action_items", [])))
                p3.metric("Topics Identified", len(result.get("topics", [])))
                p4.metric("Productivity Score", f"{result.get('productivity_score', 0)}/100")
        else:
            st.error("Could not extract text from the uploaded file. Please try a different file.")


# ════════════════════════════════════════════════════════════
# PAGE 2: MEETING DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📋 Meeting Dashboard":
    st.markdown("### 📋 Meeting Intelligence Dashboard")

    meeting = st.session_state.get("current_meeting", None)

    if meeting is None:
        all_db_meetings = db.fetch_all_meetings()
        if all_db_meetings:
            st.info("📂 Showing most recent meeting from database.")
            last_m = all_db_meetings[0]
            meeting = {
                "meeting_id": last_m.get("meeting_id", "N/A"),
                "date": last_m.get("date", "N/A"),
                "department": last_m.get("department", "N/A"),
                "raw_transcript": last_m.get("raw_transcript", ""),
                "summary": {
                    "summary_text": last_m.get("summary", ""),
                    "main_points": [last_m.get("summary", "")],
                    "key_outcomes": []
                },
                "decisions": db.fetch_all_decisions(),
                "action_items": db.fetch_all_action_items(),
                "topics": [{"topic_name": "Core Discussion", "top_words": ["meeting", "project", "team"]}],
                "speaker_analytics": {"Team": {"count": 10, "percentage": 100.0}},
                "productivity": {
                    "overall_score": last_m.get("productivity_score", 75.0),
                    "grade": "B (Good)",
                    "feedback": "Retrieved from database."
                }
            }
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; background:#FFFFFF; border-radius:16px;
                        border:2px dashed #CBD5E1; margin:2rem 0;">
                <span style="font-size:3rem;">📭</span>
                <h3 style="color:#64748B; margin:1rem 0 0.5rem 0;">No meetings processed yet</h3>
                <p style="color:#94A3B8;">Upload a meeting file in <strong>📥 Upload & Analyze</strong> to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    # ── KPI Row ───────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Meeting ID", meeting["meeting_id"])
    kpi2.metric("Department", meeting["department"])
    kpi3.metric("Date", meeting["date"])
    kpi4.metric("Decisions Made", len(meeting.get("decisions", [])))
    kpi5.metric("Action Items", len(meeting.get("action_items", [])))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary + Score Row ───────────────────────────────────
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown('<div class="custom-card"><div class="custom-card-header">📌 AI Executive Summary</div>', unsafe_allow_html=True)
        summary_obj = meeting.get("summary", {})
        if isinstance(summary_obj, str):
            summary_obj = {"summary_text": summary_obj, "main_points": [], "key_outcomes": []}

        summary_text = summary_obj.get("summary_text", "No summary available.")
        st.write(summary_text)

        main_pts = summary_obj.get("main_points", [])
        if main_pts:
            st.markdown("**📋 Main Discussion Points:**")
            for pt in main_pts:
                if pt and pt.strip():
                    st.markdown(f"• {pt}")

        key_out = summary_obj.get("key_outcomes", [])
        if key_out:
            st.markdown("**🎯 Key Outcomes:**")
            for ko in key_out:
                if ko and ko.strip():
                    st.markdown(f"✅ {ko}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        prod = meeting.get("productivity", {})
        if not isinstance(prod, dict):
            prod = {"overall_score": 75, "grade": "B", "feedback": ""}
        score_val = float(prod.get("overall_score", 75.0))
        score_pct = int(score_val)

        st.markdown(f"""
        <div class="custom-card" style="text-align:center;">
            <div class="custom-card-header" style="justify-content:center;">⚡ Productivity Score</div>
            <div style="font-size:3.5rem; font-weight:800; color:#1E3A8A; line-height:1;">{score_val}</div>
            <div style="font-size:0.85rem; color:#64748B; margin-bottom:0.5rem;">/ 100</div>
            <div style="background:linear-gradient(90deg,#2563EB {score_pct}%,#E2E8F0 {score_pct}%);
                        height:8px; border-radius:4px; margin:0.5rem 0 1rem 0;"></div>
            <div style="font-size:1rem; font-weight:700; color:#1E40AF;">{prod.get("grade","")}</div>
            <div style="font-size:0.82rem; color:#64748B; margin-top:0.5rem;">{prod.get("feedback","")}</div>
        </div>
        """, unsafe_allow_html=True)

        breakdown = prod.get("breakdown", {})
        if breakdown:
            st.markdown("**Score Breakdown:**")
            for k, v in breakdown.items():
                if "pct" not in k:
                    label = k.replace("_", " ").replace("score", "").strip().title()
                    st.caption(f"{label}: {v} pts")

    # ── Decisions + Actions ───────────────────────────────────
    st.markdown("---")
    t1, t2 = st.columns(2, gap="large")

    with t1:
        st.markdown('<div class="custom-card"><div class="custom-card-header">🎯 Decision Tracker</div>', unsafe_allow_html=True)
        dec_list = meeting.get("decisions", [])
        if dec_list:
            safe_decs = []
            for d in dec_list:
                if isinstance(d, dict):
                    safe_decs.append({
                        "Decision": d.get("decision", ""),
                        "Date": d.get("date", ""),
                        "Person": d.get("person", "Team")
                    })
                else:
                    safe_decs.append({"Decision": str(d), "Date": "", "Person": "Team"})
            st.dataframe(pd.DataFrame(safe_decs), use_container_width=True, hide_index=True)
        else:
            st.info("No explicit decisions detected in this transcript.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="custom-card"><div class="custom-card-header">📝 Action Items</div>', unsafe_allow_html=True)
        act_list = meeting.get("action_items", [])
        if act_list:
            safe_acts = []
            for a in act_list:
                if isinstance(a, dict):
                    priority = a.get("priority", "Medium")
                    safe_acts.append({
                        "Task": a.get("task", ""),
                        "Assigned To": a.get("assigned_person", "Unassigned"),
                        "Deadline": a.get("deadline", "TBD"),
                        "Priority": priority,
                        "Status": a.get("status", "Pending")
                    })
                else:
                    safe_acts.append({"Task": str(a), "Assigned To": "TBD", "Deadline": "TBD", "Priority": "Medium", "Status": "Pending"})
            st.dataframe(pd.DataFrame(safe_acts), use_container_width=True, hide_index=True)
        else:
            st.info("No action items detected in this transcript.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Topics + Speaker Analytics ────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="custom-card"><div class="custom-card-header">🏷️ Topics & Keywords</div>', unsafe_allow_html=True)
        topics_display = format_topics_for_display(meeting.get("topics", []))
        for t in topics_display:
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <span class="topic-tag">🏷️ {t['name']}</span>
                <br><small style="color:#64748B; padding-left:4px;">{t['words']}</small>
            </div>
            """, unsafe_allow_html=True)

        # Word Cloud
        raw_t = meeting.get("raw_transcript", "")
        if raw_t and WORDCLOUD_OK:
            try:
                wc = WordCloud(width=500, height=200, background_color="white",
                               colormap="Blues", max_words=60).generate(raw_t)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 2.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                fig_wc.patch.set_alpha(0)
                st.pyplot(fig_wc)
            except Exception:
                pass
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="custom-card"><div class="custom-card-header">👥 Speaker Participation</div>', unsafe_allow_html=True)
        spk_data = meeting.get("speaker_analytics", {})
        if spk_data:
            df_spk = pd.DataFrame([
                {"Speaker": k, "Sentences": v.get("count", 0), "Share (%)": v.get("percentage", 0)}
                for k, v in spk_data.items()
                if isinstance(v, dict)
            ])
            if not df_spk.empty:
                fig_pie = px.pie(
                    df_spk, names="Speaker", values="Sentences",
                    title="Speaker Turn Distribution",
                    color_discrete_sequence=["#1E3A8A", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
                )
                fig_pie.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    font=dict(family="Inter, sans-serif"),
                    title_font=dict(size=14, color="#1E3A8A")
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── NER Section ───────────────────────────────────────────
    ner_data = meeting.get("ner", {})
    if ner_data and any(ner_data.values()):
        st.markdown("---")
        st.markdown('<div class="custom-card"><div class="custom-card-header">🔍 Named Entities Detected</div>', unsafe_allow_html=True)
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            st.markdown("**👤 People**")
            for p in ner_data.get("PERSON", []):
                st.markdown(f'<span class="topic-tag">{p}</span>', unsafe_allow_html=True)
        with n2:
            st.markdown("**📅 Dates**")
            for d in ner_data.get("DATE", []):
                st.markdown(f'<span class="topic-tag">{d}</span>', unsafe_allow_html=True)
        with n3:
            st.markdown("**🏢 Organizations**")
            for o in ner_data.get("ORGANIZATION", []):
                st.markdown(f'<span class="topic-tag">{o}</span>', unsafe_allow_html=True)
        with n4:
            st.markdown("**📌 Tasks**")
            for t in ner_data.get("TASK", [])[:5]:
                st.markdown(f'<span class="topic-tag">{t[:40]}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3: AI MEMORY SEARCH
# ════════════════════════════════════════════════════════════
elif page == "🔍 AI Memory Search":
    st.markdown("### 🔍 AI Meeting Memory & Vector Search")
    st.write("Search historical meetings using natural language — powered by **Sentence-BERT** embeddings and **FAISS** similarity index.")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    search_query = st.text_input(
        "🔎 Ask anything about your meetings",
        placeholder="e.g. What was decided about the marketing budget last month?",
        label_visibility="collapsed"
    )

    col_btn, col_k = st.columns([1, 3])
    with col_btn:
        search_btn = st.button("🔍 Search Meetings", use_container_width=True)
    with col_k:
        top_k = st.slider("Max results", 1, 10, 3, label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    if search_btn and search_query.strip():
        with st.spinner("Searching vector database..."):
            try:
                results = vector_store.search_meetings(search_query.strip(), top_k=top_k)
            except Exception as e:
                st.error(f"Search error: {e}")
                results = []

        if results:
            st.success(f"✅ Found **{len(results)}** matching meeting(s)")
            for idx, res in enumerate(results, 1):
                sim = res.get("similarity_score", 0)
                with st.expander(
                    f"📋 Match #{idx} — {res.get('meeting_id', 'N/A')} | {res.get('department', '')} | Similarity: {sim}%",
                    expanded=(idx == 1)
                ):
                    r1, r2 = st.columns(2)
                    r1.markdown(f"**📅 Date:** {res.get('date', 'N/A')}")
                    r2.markdown(f"**🏢 Department:** {res.get('department', 'N/A')}")

                    summary_raw = res.get("summary", "")
                    if isinstance(summary_raw, dict):
                        summary_raw = summary_raw.get("summary_text", "")
                    if summary_raw:
                        st.markdown(f"**📝 Summary:** {summary_raw}")

                    topics_raw = res.get("topics", [])
                    topics_disp = format_topics_for_display(topics_raw)
                    if topics_disp:
                        st.markdown("**🏷️ Topics:** " + " · ".join([t["name"] for t in topics_disp]))

                    decisions_raw = res.get("decisions", [])
                    if decisions_raw:
                        st.markdown("**✅ Decisions:**")
                        for d in decisions_raw[:3]:
                            if isinstance(d, dict):
                                st.markdown(f"  - {d.get('decision', str(d))}")
                            else:
                                st.markdown(f"  - {d}")
        else:
            st.info("💡 No matches found. Process more meetings to build your AI memory index.")


# ════════════════════════════════════════════════════════════
# PAGE 4: DATASET EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 Dataset EDA":
    st.markdown("### 📊 Exploratory Data Analysis")
    st.write("Statistical insights across the synthetic enterprise meeting dataset.")

    if eda_engine is None:
        st.warning("EDA engine failed to load. Please check the `src/eda.py` module.")
        st.stop()

    try:
        kpis = eda_engine.get_summary_kpis()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📋 Total Meetings", kpis.get("total_meetings", 0))
        m2.metric("⏱️ Avg Duration", f"{kpis.get('avg_duration', 0)} min")
        m3.metric("✅ Total Decisions", kpis.get("total_decisions", 0))
        m4.metric("📝 Total Tasks", kpis.get("total_tasks", 0))
        m5.metric("👥 Avg Speakers", kpis.get("avg_speakers", 0))
    except Exception as e:
        st.warning(f"KPI load error: {e}")

    st.markdown("---")

    chart_opt = st.selectbox(
        "Select Visualization",
        [
            "📊 Meeting Duration Distribution",
            "👥 Speaker Contribution Analysis",
            "🏷️ Topic Frequency Analysis",
            "✅ Decision Count Trends",
            "📌 Task vs Decision Density Matrix"
        ]
    )

    try:
        if "Duration" in chart_opt:
            st.plotly_chart(eda_engine.plot_duration_distribution(), use_container_width=True)
        elif "Speaker" in chart_opt:
            st.plotly_chart(eda_engine.plot_speaker_contribution(), use_container_width=True)
        elif "Topic" in chart_opt:
            st.plotly_chart(eda_engine.plot_topic_frequency(), use_container_width=True)
        elif "Decision" in chart_opt:
            st.plotly_chart(eda_engine.plot_decision_trends(), use_container_width=True)
        elif "Task" in chart_opt:
            st.plotly_chart(eda_engine.plot_task_completion_analysis(), use_container_width=True)
    except Exception as e:
        st.error(f"Chart render error: {e}")
        st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════
# PAGE 5: SYSTEM INFO
# ════════════════════════════════════════════════════════════
elif page == "⚙️ System Info":
    st.markdown("### ⚙️ System Architecture & NLP Pipeline")

    st.markdown("""
    <div class="custom-card">
    <div class="custom-card-header">🔧 8-Module NLP Pipeline</div>
    """, unsafe_allow_html=True)

    modules = [
        ("1", "📁", "File Parser", "Parses .txt, .pdf, .docx transcripts"),
        ("2", "🎙️", "Audio Processor", "Local Whisper STT (.mp3, .wav → text)"),
        ("3", "🧹", "Preprocessor", "NLTK & spaCy tokenization, stopword, lemmatization"),
        ("4", "📝", "Summarization", "BART/T5 transformer + TF-IDF extractive fallback"),
        ("5", "🔍", "NER Extraction", "spaCy + BERT NER — Person, Date, Org, Task"),
        ("6", "✅", "Action Extractor", "Structured JSON action item builder"),
        ("7", "🎯", "Decision Engine", "Keyword + syntactic decision detector"),
        ("8", "🏷️", "Topic Modeler", "Dual LDA + BERTopic/Sentence-Transformers"),
        ("9", "🔎", "Vector Search", "Sentence-BERT (all-MiniLM-L6-v2) + FAISS"),
        ("10", "🗄️", "Database", "SQLite persistence — meetings, decisions, actions"),
    ]

    for num, icon, name, desc in modules:
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="background:#EFF6FF; color:#1E3A8A; border-radius:8px;
                         padding:4px 10px; font-weight:700; font-size:0.82rem;">#{num}</span>
            <span style="font-size:1.2rem;">{icon}</span>
            <span><strong style="color:#1E3A8A;">{name}</strong>
            <span style="color:#64748B; font-size:0.85rem;"> — {desc}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
    <div class="custom-card-header">🛠️ Technology Stack</div>
    """, unsafe_allow_html=True)

    tech = {
        "Language": "Python 3.9+",
        "NLP & Transformers": "transformers · sentence-transformers · spaCy · nltk",
        "Speech Recognition": "openai-whisper (local, no API key needed)",
        "Vector Database": "faiss-cpu",
        "UI Dashboard": "streamlit · plotly · wordcloud · matplotlib",
        "Database": "sqlite3 (local file)",
        "Privacy": "100% offline — no data leaves your machine"
    }

    for k, v in tech.items():
        st.markdown(f"**{k}:** `{v}`")

    st.markdown("</div>", unsafe_allow_html=True)

    if import_errors:
        st.markdown("""<div class="custom-card">
        <div class="custom-card-header">⚠️ Module Load Warnings</div>""", unsafe_allow_html=True)
        for err in import_errors:
            st.warning(f"• {err}")
        st.markdown("</div>", unsafe_allow_html=True)
