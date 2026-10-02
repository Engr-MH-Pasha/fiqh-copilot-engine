import os
import uuid
import json
import urllib.request
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# PDF Engine Imports
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Page Configuration
st.set_page_config(
    page_title="Fiqh Co-Pilot | Juristic AI Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Multilingual Polish
st.markdown("""
<style>
    .card-arabic {
        background-color: #FFFFFF;
        padding: 16px 20px;
        border-right: 6px solid #1A365D;
        border-radius: 8px;
        direction: rtl;
        text-align: right;
        font-family: 'Amiri', Tahoma, serif;
        font-size: 1.25rem;
        line-height: 2.1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .card-translation {
        background-color: #F8FAFC;
        padding: 14px 20px;
        border-right: 6px solid #0D9488;
        border-radius: 8px;
        direction: rtl;
        text-align: right;
        font-family: Arial, Tahoma, sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #0F172A;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    .agent-pill {
        background-color: #EEF2F6;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1E293B;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. INITIALIZE CLOUD VECTOR DATABASE & EMBEDDINGS
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Qdrant Cloud & Vector Models...")
def initialize_system():
    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    qdrant_url = st.secrets.get("QDRANT_URL", None)
    qdrant_key = st.secrets.get("QDRANT_API_KEY", None)
    
    if qdrant_url and qdrant_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    else:
        client = QdrantClient(":memory:")

    collection_name = "fiqh_canonical_collection"

    # Amiri Font Setup for PDF Engine
    font_path = "Amiri-Regular.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf", 
            font_path
        )
    pdfmetrics.registerFont(TTFont("Amiri", font_path))

    return encoder, client, collection_name

encoder, qdrant_client, COLLECTION_NAME = initialize_system()

# Helper function to detect available Groq model
def get_active_groq_model(groq_client: Groq) -> str:
    """Finds the best active model available on the user's Groq tier."""
    try:
        available_ids = [m.id for m in groq_client.models.list().data]
        priority_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
        for p in priority_models:
            if p in available_ids:
                return p
        return available_ids[0] if available_ids else "llama-3.1-8b-instant"
    except Exception:
        return "llama-3.1-8b-instant"

# ------------------------------------------------------------------------------
# 2. TRI-AGENT JURISTIC PIPELINE
# ------------------------------------------------------------------------------
class State(BaseModel):
    query: str
    target_lang: str
    classical_arabic_query: str = ""
    citations: List[Dict[str, Any]] = []
    juristic_synthesis: str = ""
    translations: Dict[int, Dict[str, str]] = {}

def run_agentic_workflow(user_query: str, selected_lang: str, limit: int) -> State:
    state = State(query=user_query, target_lang=selected_lang)
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    
    # Language Detection
    q_low = user_query.lower()
    if selected_lang == "Auto Detect":
        if any(c in q_low for c in ['ı', 'ş', 'ğ', 'ç']):
            state.target_lang = "Turkish"
        elif any(c in user_query for c in ['ہے', 'کیا', 'کے', 'سے', 'نیت', 'نماز', 'حکم', 'پر', 'میں']):
            state.target_lang = "Urdu"
        elif any(c in user_query for c in ['است', 'شدن', 'کردن']):
            state.target_lang = "Persian"
        elif any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in user_query):
            state.target_lang = "Arabic"
        else:
            state.target_lang = "English"

    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
    active_model = get_active_groq_model(groq_client) if groq_client else "llama-3.1-8b-instant"

    # --------------------------------------------------------------------------
    # AGENT 1: Query Transformation into Classical Hanafi Arabic Terminology
    # --------------------------------------------------------------------------
    if groq_client:
        try:
            expansion_prompt = """
            You are a Classical Hanafi Juristic Lexicographer (محرر المذهب الحنفي).
            Your task is to convert any user inquiry (Urdu, English, Turkish, Persian) into the exact Classical Arabic terminology used in canonical Hanafi treatises (رد المحتار، الفتاوى الهندية، الهداية، بدائع الصنائع).
            
            Instructions:
            - Output ONLY 2-3 precise Arabic legal search sentences with exact fiqh terminology (Kitab, Bab, and Masa'il keywords).
            - Do not include explanations, notes, or introductions. Only output the classical Arabic text.
            """
            
            expansion_res = groq_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": expansion_prompt},
                    {"role": "user", "content": f"User Inquiry: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=150
            )
            state.classical_arabic_query = expansion_res.choices[0].message.content.strip()
        except Exception:
            state.classical_arabic_query = user_query
    else:
        state.classical_arabic_query = user_query

    # --------------------------------------------------------------------------
    # AGENT 2: Semantic Retrieval via Qdrant Cloud using Classical Arabic
    # --------------------------------------------------------------------------
    search_payload = f"{state.classical_arabic_query} {user_query}"
    q_vec = encoder.encode(search_payload).tolist()
    
    try:
        hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, 
            query=q_vec, 
            limit=limit
        )
        state.citations = [{"score": round(h.score, 4), "data": h.payload} for h in hits.points]
    except Exception:
        state.citations = []

    # --------------------------------------------------------------------------
    # AGENT 3: Synthesis & Dual Translation Engine into User's Language
    # --------------------------------------------------------------------------
    if groq_client and state.citations:
        try:
            numbered_contexts = []
            for idx, c in enumerate(state.citations, 1):
                numbered_contexts.append(
                    f"[Citation {idx}] {c['data']['book_title_ar']} (Vol {c['data']['volume']}, Page {c['data']['page']}):\n{c['data']['text']}"
                )
            context_payload = "\n\n".join(numbered_contexts)

            synthesis_prompt = f"""
            You are an elite Mufti and Juristic Researcher assisting Dar-ul-Ifta.
            You must return your output strictly in valid JSON format.
            The user asked the question in {state.target_lang}.
            
            CRITICAL REQUIREMENTS:
            1. All synthesis, legal analysis, and text translations MUST be written strictly in fluent, formal, and authoritative {state.target_lang}.
            2. The "synthesis" must provide:
               - The definitive legal ruling (حکمِ شرعی) directly answering the user.
               - Analysis of the texts (نصوص کی روشنی میں دل کی نیت، زبان کی نیت، وقت، اور شرائط کا حکم).
               - Guidance for Muftis (حاصلِ بحث برائے مفتی).
            3. In "translations", provide a comprehensive translation for EVERY citation, along with its specific legal takeaway (حاصلِ نکتہ).
            
            JSON Structure:
            {{
              "synthesis": "مکمل شرعی خلاصہ و تحقیق (In {state.target_lang})",
              "translations": [
                {{
                  "id": 1,
                  "translation": "عربی عبارت کا سلیس ترجمہ (In {state.target_lang})",
                  "legal_point": "اس عبارت کا فقہی خلاصہ و نکتہ (In {state.target_lang})"
                }}
              ]
            }}
            """

            user_msg = f"User Question: {user_query}\nCanonical Arabic Findings:\n{context_payload}"

            completion = groq_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": synthesis_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=2200
            )

            res_json = json.loads(completion.choices[0].message.content)
            state.juristic_synthesis = res_json.get("synthesis", "")
            
            for item in res_json.get("translations", []):
                state.translations[item["id"]] = {
                    "text": item.get("translation", ""),
                    "point": item.get("legal_point", "")
                }
        except Exception as e:
            state.juristic_synthesis = f"نصوص حاصل ہو گئے ہیں، لیکن ماڈل تجزیہ میں رکاوٹ آئی: {str(e)}"
    else:
        state.juristic_synthesis = "فقہ حنفی کے معتمد ذخیرے سے نصوص برآمد ہو گئے ہیں۔ برائے مہربانی Streamlit Secrets میں GROQ_API_KEY کی تصدیق فرمائیں۔"

    return state

# ------------------------------------------------------------------------------
# 3. BUSINESS PROCESS AUTOMATION (PDF DOSSIER ENGINE)
# ------------------------------------------------------------------------------
def create_pdf_dossier(state: State) -> str:
    pdf_path = "Fiqh_Research_Dossier.pdf"
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=A4, 
        rightMargin=36, 
        leftMargin=36, 
        topMargin=36, 
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    def reshape(txt):
        try:
            return get_display(arabic_reshaper.reshape(txt))
        except:
            return txt

    story = [
        Paragraph("DAR-UL-IFTA JURISTIC RESEARCH DOSSIER", ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=15, alignment=1, textColor=colors.HexColor("#1A365D"))),
        Paragraph("Canonical Retrieval & Multilingual Juristic Verification Record", ParagraphStyle('H2', fontName='Helvetica', fontSize=8, alignment=1, textColor=colors.HexColor("#4A5568"))),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A365D"), spaceAfter=10)
    ]
    
    meta_rows = [
        [Paragraph("<b>Query:</b>", styles['Normal']), Paragraph(reshape(state.query), ParagraphStyle('ArQ', fontName='Amiri', fontSize=10, alignment=2))],
        [Paragraph("<b>Juristic Arabic Query:</b>", styles['Normal']), Paragraph(reshape(state.classical_arabic_query), ParagraphStyle('ArExp', fontName='Amiri', fontSize=9, alignment=2))],
        [Paragraph("<b>Target Language:</b>", styles['Normal']), Paragraph(state.target_lang, styles['Normal'])],
        [Paragraph("<b>Verified Citations:</b>", styles['Normal']), Paragraph(f"{len(state.citations)} Canonical References", styles['Normal'])],
        [Paragraph("<b>Date:</b>", styles['Normal']), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M UTC"), styles['Normal'])]
    ]
    t = Table(meta_rows, colWidths=[130, 390])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#EDF2F7")),
        ('PADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    for idx, c in enumerate(state.citations, 1):
        d = c["data"]
        story.append(Paragraph(
            f"<b>Citation #{idx}: {d['book_title_ar']}</b> (Vol: {d['volume']}, Page: {d['page']})", 
            ParagraphStyle('C1', fontName='Amiri', fontSize=10, textColor=colors.HexColor("#2C5282"), alignment=2)
        ))
        
        quote = Table([[Paragraph(f"« {reshape(d['text'])} »", ParagraphStyle('ArTxt', fontName='Amiri', fontSize=10, leading=14, alignment=2))]], colWidths=[520])
        quote.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFAF0")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#DD6B20")),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        story.append(quote)
        
        tr = state.translations.get(idx)
        if tr:
            tr_box = Table([[Paragraph(f"<b>مفہوم و ترجمہ:</b> {reshape(tr['text'])}", ParagraphStyle('TrTxt', fontName='Amiri', fontSize=9, leading=13, alignment=2))]], colWidths=[520])
            tr_box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#10B981")),
                ('PADDING', (0,0), (-1,-1), 5)
            ]))
            story.append(tr_box)
            
        story.append(Spacer(1, 8))

    doc.build(story)
    return pdf_path

# ------------------------------------------------------------------------------
# 4. STREAMLIT UI DASHBOARD
# ------------------------------------------------------------------------------
st.title("⚖️ Fiqh Co-Pilot | Juristic AI Engine")
st.caption("Autonomous Multi-Agent Legal System for Muftis & Islamic Academics | Hanafi Canonical Treatises")

with st.sidebar:
    st.header("⚙️ ترتیباتِ تحقیق (Settings)")
    lang_choice = st.selectbox(
        "جواب اور ترجمہ کی زبان (Language):",
        ["Auto Detect", "Urdu", "Arabic", "English", "Turkish", "Persian"],
        index=0
    )
    top_k = st.slider("مطلوبہ مراجع کی تعداد (Citations):", min_value=3, max_value=10, value=7)
    st.markdown("---")
    st.markdown("### 📚 معتمد کتبِ خمسہ")
    st.markdown("""
    1. **رد المحتار على الدر المختار** (ابن عابدين الشامي)
    2. **الفتاوى الهندية** (البلخي وجماعة العلماء)
    3. **الهداية شرح بداية المبتدي** (المرغيناني)
    4. **بدائع الصنائع في ترتيب الشرائع** (الكاساني)
    5. **البحر الرائق شرح كنز الدقائق** (ابن نجيم)
    """)

user_prompt = st.text_input(
    "اپنا فقہی سوال یہاں درج فرمائیں (مثلاً: نماز میں نیت کا کیا حکم ہے؟):",
    placeholder="سوال اردو، عربی، انگلش یا ترکی میں لکھیں..."
)

if user_prompt:
    with st.spinner("ملٹی ایجنٹ پائپ لائن فعال ہے (عربی اصطلاحات کی تشکیل ⮞ کلاؤڈ تلاش ⮞ سلیس ترجمہ و خلاصہ)..."):
        result = run_agentic_workflow(user_prompt, lang_choice, top_k)
    
    st.success(f"منتخب شدہ زبان: **{result.target_lang}** | مصدقہ مراجع: **{len(result.citations)} کتب**")
    
    # Inspection of Agent 1 Transformation
    with st.expander("🔍 ایجنٹ 1: سوال کا کتبِ فقہ کی اصطلاحی عربی میں تجزیہ (Juristic Arabic Mapping)"):
        st.markdown(f"<div class='agent-pill'><b>تلاش کے لیے تیار کردہ عربی عبارات:</b><br>{result.classical_arabic_query}</div>", unsafe_allow_html=True)

    # Agent 3 Juristic Synthesis
    st.markdown("### 📋 شرعی حکم و مفتی خلاصہ (Juristic Synthesis)")
    st.info(result.juristic_synthesis)
    
    # PDF Action
    pdf_filename = create_pdf_dossier(result)
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="📄 باقاعدہ فتاویٰ ڈوزیئر ڈاؤنلوڈ کریں (Download Official PDF)",
            data=f,
            file_name="Fiqh_Research_Dossier.pdf",
            mime="application/pdf"
        )
        
    st.markdown("---")
    st.subheader(f"📖 فقہی نصوص مع سلیس ترجمہ و تخریج ({len(result.citations)} مراجع)")
    
    # Citation Cards
    for i, item in enumerate(result.citations, 1):
        meta = item["data"]
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### حوالہ #{i}: {meta['book_title_ar']}")
                st.caption(f"**مصنف:** {meta['author_ar']} | **باب:** {meta['kitab']} ⮞ {meta['bab']}")
            with col2:
                score_val = item['score']
                score_badge = "🟢 قوی ترین" if score_val > 0.50 else "🟡 اوسط"
                st.metric("مطابقت (Score)", f"{score_val}", delta=score_badge)
                st.write(f"📖 **جلد:** {meta['volume']} | **صفحہ:** {meta['page']}")
            
            # Original Arabic Matn
            st.markdown(
                f"<div class='card-arabic'><b>العبارة الأصلية (متن):</b><br>{meta['text']}</div>", 
                unsafe_allow_html=True
            )
            
            # Target Language Translation
            tr_data = result.translations.get(i)
            if tr_data and tr_data.get("text"):
                st.markdown(
                    f"<div class='card-translation'><b>سلیس ترجمہ و مفہوم ({result.target_lang}):</b><br>"
                    f"{tr_data['text']}<br><br>"
                    f"<b>📌 حاصل نکتہ:</b> {tr_data.get('point', '')}</div>", 
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
