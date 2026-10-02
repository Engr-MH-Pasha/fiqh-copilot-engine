import os
import uuid
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

# Custom Styling for Academic & RTL Polish
st.markdown("""
<style>
    .card-arabic {
        background-color: #FFFFFF;
        padding: 18px;
        border-right: 6px solid #1A365D;
        border-radius: 8px;
        direction: rtl;
        text-align: right;
        font-family: 'Amiri', Tahoma, serif;
        font-size: 1.25rem;
        line-height: 2.0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. INITIALIZE PERSISTENT CLOUD VECTOR DATABASE & EMBEDDINGS
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Qdrant Cloud & AI Models...")
def initialize_system():
    # Dense Multilingual Embedding Model
    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Retrieve cloud credentials from Streamlit Secrets
    qdrant_url = st.secrets.get("QDRANT_URL", None)
    qdrant_key = st.secrets.get("QDRANT_API_KEY", None)
    
    if qdrant_url and qdrant_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    else:
        client = QdrantClient(":memory:")

    collection_name = "fiqh_canonical_collection"

    # Font setup for PDF Dossier (Arabic/Urdu Typography)
    font_path = "Amiri-Regular.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf", 
            font_path
        )
    pdfmetrics.registerFont(TTFont("Amiri", font_path))

    return encoder, client, collection_name

encoder, qdrant_client, COLLECTION_NAME = initialize_system()

# ------------------------------------------------------------------------------
# 2. MULTI-AGENT JURISTIC REASONING PIPELINE
# ------------------------------------------------------------------------------
class State(BaseModel):
    query: str
    target_lang: str
    expanded_arabic: str = ""
    citations: List[Dict[str, Any]] = []
    juristic_synthesis: str = ""

def run_agentic_workflow(user_query: str, selected_lang: str, limit: int) -> State:
    state = State(query=user_query, target_lang=selected_lang)
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    
    # Agent 1: Language Detection & Classical Fiqh Mapping
    q_low = user_query.lower()
    if selected_lang == "Auto Detect":
        if any(c in q_low for c in ['ı', 'ş', 'ğ', 'ç']):
            state.target_lang = "Turkish"
        elif any(c in user_query for c in ['ہے', 'کیا', 'کے', 'سے', 'نیت', 'نماز', 'حکم']):
            state.target_lang = "Urdu"
        elif any(c in user_query for c in ['است', 'شدن', 'کردن']):
            state.target_lang = "Persian"
        elif any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in user_query):
            state.target_lang = "Arabic"
        else:
            state.target_lang = "English"

    vocab = {
        "نماز": "شروط الصلاة وأركانها وباب النية والتحريمة",
        "نیت": "نية الصلاة وتعيين الفرض وإخلاص القلب",
        "وضو": "فرائض الوضوء وسننه ونواقضه",
        "طہارت": "شروط الطهارة وأحكام المياه",
        "سود": "باب الربا والمعاملات المصرفية المحرمة",
        "prayer": "شروط الصلاة وباب النية",
        "wudu": "فرائض الوضوء وأركانه"
    }
    matched = [v for k, v in vocab.items() if k in q_low or k in user_query]
    state.expanded_arabic = " ".join(matched) if matched else user_query

    # Agent 2: Dense Semantic Retrieval from Qdrant Cloud
    search_str = f"{state.expanded_arabic} {user_query}"
    q_vec = encoder.encode(search_str).tolist()
    
    try:
        hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, 
            query=q_vec, 
            limit=limit
        )
        state.citations = [{"score": round(h.score, 4), "data": h.payload} for h in hits.points]
    except Exception:
        state.citations = []

    # Agent 3: Multilingual Juristic Synthesis Engine (Strictly in Target Language)
    if groq_api_key and state.citations:
        try:
            client = Groq(api_key=groq_api_key)
            context_blocks = "\n".join([
                f"- {c['data']['book_title_ar']} (جلد: {c['data']['volume']}، صفحہ: {c['data']['page']}): {c['data']['text']}" 
                for c in state.citations
            ])
            
            lang_instruction = {
                "Urdu": "آپ کو اپنا مکمل تجزیہ، شرعی حکم، فقہی موازنہ اور خلاصہ خالص، سلیس اور مستند اردو زبان میں لکھنا ہے۔ عربی صرف اصل حوالہ کے طور پر استعمال ہو۔",
                "English": "Write your entire juristic evaluation and synthesis strictly in professional scholarly English.",
                "Arabic": "اكتب التقرير الفقهي والتحقيق المذهبي باللغة العربية الفصحى الرصينة.",
                "Turkish": "Tüm fıkhi tahlili ve özeti akademik Türkçe ile kaleme alınız.",
                "Persian": "گزارش و جمع‌بندی فقهی را به زبان فارسی روان و دقیق بنویسید."
            }.get(state.target_lang, "جواب سوال کی زبان میں دیں۔")

            system_prompt = f"""
            آپ مفتیانِ کرام اور فقہی محققین کے لیے ایک معاون Fiqh Co-Pilot ہیں۔
            نیچے دی گئی فقہ حنفی کی معتمد کتب (رد المحتار، الہندیہ، الہدایہ، بدائع، بحر الرائق) کے نصوص کی روشنی میں سائل کے سوال کا تفصیلی، جامع اور مدلل جواب دیں۔
            
            اہم شرائط:
            1. زبان کی پابندی: {lang_instruction}
            2. جواب کا اسٹرکچر:
               - **حکمِ شرعی (خلاصہ):** سوال کا براہِ راست فقہی حکم۔
               - **نصوص سے استدلال و تفصیل:** نصوص کی بنیاد پر دلائل، دل اور زبان کی نیت کا فرق، وقت اور شرائط۔
               - **حاصلِ بحث برائے مفتی:** فتویٰ لکھنے والے کے لیے اہم تنبیہات۔
            3. ذاتی رائے مت دیں؛ نصوص میں جو منقول ہے اسی کی ترجمانی کریں۔
            """
            
            user_msg = f"سوال:\n{user_query}\n\nمعتمد فقہی نصوص (Evidence):\n{context_blocks}"
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            state.juristic_synthesis = completion.choices[0].message.content
        except Exception:
            state.juristic_synthesis = "کلاؤڈ سے متعلقہ کتب کے نصوص کامیابی سے حاصل کر لیے گئے ہیں۔ نصوص ذیل میں ملاحظہ فرمائیں۔"
    else:
        state.juristic_synthesis = "فقہ حنفی کے معتمد ذخیرے سے نصوص برآمد ہو گئے ہیں۔ درست خلاصے کے لیے Streamlit Secrets میں GROQ_API_KEY درج فرمائیں۔"

    return state

# ------------------------------------------------------------------------------
# 3. BUSINESS PROCESS AUTOMATION (BPA PDF ENGINE)
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
        Paragraph("Grounded Multi-Agent Retrieval & Scholarly Citation Report", ParagraphStyle('H2', fontName='Helvetica', fontSize=8, alignment=1, textColor=colors.HexColor("#4A5568"))),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A365D"), spaceAfter=10)
    ]
    
    meta_rows = [
        [Paragraph("<b>Query:</b>", styles['Normal']), Paragraph(reshape(state.query), ParagraphStyle('ArQ', fontName='Amiri', fontSize=10, alignment=2))],
        [Paragraph("<b>Language:</b>", styles['Normal']), Paragraph(state.target_lang, styles['Normal'])],
        [Paragraph("<b>Citations:</b>", styles['Normal']), Paragraph(f"{len(state.citations)} Canonical References", styles['Normal'])],
        [Paragraph("<b>Date:</b>", styles['Normal']), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M UTC"), styles['Normal'])]
    ]
    t = Table(meta_rows, colWidths=[120, 400])
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
        story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path

# ------------------------------------------------------------------------------
# 4. STREAMLIT UI DASHBOARD
# ------------------------------------------------------------------------------
st.title("⚖️ Fiqh Co-Pilot | Juristic AI Engine")
st.caption("AI-Powered Research System for Muftis & Islamic Scholars | Hanafi Canonical Treatises")

with st.sidebar:
    st.header("⚙️ ترتیباتِ تحقیق (Settings)")
    lang_choice = st.selectbox(
        "سوال کی زبان منتخب کریں (Language):",
        ["Auto Detect", "Urdu", "Arabic", "English", "Turkish", "Persian"],
        index=0
    )
    top_k = st.slider("مطلوبہ مراجع و فتاویٰ کی تعداد (Citations):", min_value=3, max_value=10, value=7)
    st.markdown("---")
    st.markdown("### 📚 معتمد کتبِ خمسہ")
    st.markdown("""
    1. **رد المحتار على الدر المختار** (علامہ ابن عابدین الشامی)
    2. **الفتاوى الهندية** (شیخ نظام الدین بلخی و علماء ہند)
    3. **الهداية شرح بداية المبتدي** (امام برہان الدین مرغینانی)
    4. **بدائع الصنائع في ترتيب الشرائع** (امام علاء الدین کاسانی)
    5. **البحر الرائق شرح كنز الدقائق** (علامہ ابن نجیم مصری)
    """)

user_prompt = st.text_input(
    "اپنا فقہی سوال یہاں درج فرمائیں (مثلاً: نماز میں نیت کا کیا حکم ہے؟):",
    placeholder="اردو، عربی، انگلش، ترکی یا فارسی میں سوال لکھیں..."
)

if user_prompt:
    with st.spinner("کلاؤڈ ویکٹر انڈیکس اور فقہی ایجنٹ سے معلومات اخذ کی جا رہی ہیں..."):
        result = run_agentic_workflow(user_prompt, lang_choice, top_k)
    
    st.success(f"شناخت شدہ زبان: **{result.target_lang}** | بازیاب شدہ مراجع: **{len(result.citations)} کتب**")
    
    # 1. Juristic Synthesis in User Language
    st.markdown("### 📋 فقہی جائزہ و خلاصہ برائے مفتی (Juristic Synthesis)")
    st.info(result.juristic_synthesis)
    
    # 2. PDF Download Action
    pdf_filename = create_pdf_dossier(result)
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="📄 باقاعدہ فتاویٰ ڈوزیئر ڈاؤنلوڈ کریں (Download Official PDF)",
            data=f,
            file_name="Fiqh_Research_Dossier.pdf",
            mime="application/pdf"
        )
        
    st.markdown("---")
    st.subheader(f"📖 کتبِ فقہ کی اصل عبارات مع جلد و صفحہ ({len(result.citations)} مراجع)")
    
    # 3. Individual Source Cards
    for i, item in enumerate(result.citations, 1):
        meta = item["data"]
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### حوالہ #{i}: {meta['book_title_ar']}")
                st.caption(f"**مصنف:** {meta['author_ar']} | **مقام:** {meta['kitab']} -> {meta['bab']}")
            with col2:
                score_val = item['score']
                score_badge = "🟢 قوی ترین" if score_val > 0.50 else "🟡 اوسط"
                st.metric(f"مطابقت (Score)", f"{score_val}", delta=score_badge)
                st.write(f"📖 **جلد:** {meta['volume']} | **صفحہ:** {meta['page']}")
            
            # Arabic Matn in Beautiful Box
            st.markdown(f"<div class='card-arabic'>{meta['text']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
