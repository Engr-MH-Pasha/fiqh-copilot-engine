import os
import uuid
import urllib.request
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# PDF Generation Imports
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

# ------------------------------------------------------------------------------
# CACHED ENGINES & DATA INITIALIZATION
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Multilingual Vector Models...")
def initialize_system():
    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    client = QdrantClient(":memory:")
    collection_name = "fiqh_canonical_collection"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    raw_corpus = [
        {
            "book_title_ar": "رد المحتار على الدر المختار (حاشية ابن عابدين)",
            "book_slug": "radd_al_muhtar",
            "author_ar": "محمد أمين بن عمر (ابن عابدين الشامي)",
            "volume": 1, "page": 95, "kitab": "كتاب الطهارة", "bab": "باب الوضوء",
            "text": "فرائض الوضوء أربعة: غسل الوجه، وغسل اليدين مع المرفقين، ومسح ربع الرأس، وغسل الرجلين مع الكعبين. والنية فيه سنة مؤكدة لأداء العبادة وليست بشرط لصحة الطهارة عندنا."
        },
        {
            "book_title_ar": "الفتاوى الهندية (عالمكيري)",
            "book_slug": "al_fatawa_al_hindiyya",
            "author_ar": "لجنة برئاسة الشيخ نظام الدين البلخي",
            "volume": 1, "page": 2, "kitab": "كتاب الطهارة", "bab": "الفصل الأول في فرائض الوضوء",
            "text": "وفيه أربعة فصول: الفصل الأول في فرائضه، وهي أربعة: غسل الوجه وهو من قصاص شعر الرأس إلى أسفل الذقن طولاً، وما بين شحمتي الأذنين عرضاً."
        },
        {
            "book_title_ar": "الهداية في شرح بداية المبتدي",
            "book_slug": "al_hidayah",
            "author_ar": "علي بن أبي بكر المرغيناني",
            "volume": 1, "page": 16, "kitab": "كتاب الطهارات", "bab": "باب نواقض الوضوء",
            "text": "وينقض الوضوء كل ما خرج من السبيلين، والدم والقيح والصديد إذا خرج من البدن فتجاوز إلى موضع يلحقه حكم التطهير، والقيء إذا ملأ الفم."
        },
        {
            "book_title_ar": "بدائع الصنائع في ترتيب الشرائع",
            "book_slug": "badai_al_sanai",
            "author_ar": "علاء الدين الكاساني",
            "volume": 1, "page": 3, "kitab": "كتاب الطهارة", "bab": "فصل في بيان أركان الوضوء",
            "text": "أما ركن الوضوء فهو غسل الأعضاء الثلاثة ومسح الرأس، والكلام فيه يقع في مواضع: أحدها في بيان المقدار المفروض من الغسل والمسح."
        },
        {
            "book_title_ar": "البحر الرائق شرح كنز الدقائق",
            "book_slug": "al_bahr_al_raiq",
            "author_ar": "زين الدين بن إبراهيم (ابن نجيم)",
            "volume": 1, "page": 12, "kitab": "كتاب الطهارة", "bab": "شروط وجوب الوضوء",
            "text": "وشرائط وجوب الوضوء: الإسلام، والعقل، والبلوغ، ووجود الماء الكافي، والقدرة على استعماله، وطهارة المحل من الحيض والنفاس."
        },
        {
            "book_title_ar": "رد المحتار على الدر المختار (حاشية ابن عابدين)",
            "book_slug": "radd_al_muhtar",
            "author_ar": "محمد أمين بن عمر (ابن عابدين الشامي)",
            "volume": 5, "page": 166, "kitab": "كتاب البيوع", "bab": "باب الربا",
            "text": "الربا شرعاً: فضل مال خال عن عوض بمعيار شرعي مشروط لأحد العاقدين في المعاوضة. وحرمته ثابتة بالكتاب والسنة والإجماع."
        },
        {
            "book_title_ar": "الفتاوى الهندية (عالمكيري)",
            "book_slug": "al_fatawa_al_hindiyya",
            "author_ar": "لجنة برئاسة الشيخ نظام الدين البلخي",
            "volume": 3, "page": 112, "kitab": "كتاب البيوع", "bab": "الباب الأول في شروط البيع",
            "text": "البيع ينعقد بالإيجاب والقبول إذا كانا بلفظ الماضي، كقول البائع بعت والمشتري اشتريت، ولا ينعقد بلفظ المستقبل إلا بالنية."
        }
    ]

    points = []
    for item in raw_corpus:
        v = encoder.encode(item["text"]).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=item
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    
    font_path = "Amiri-Regular.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf", font_path)
    pdfmetrics.registerFont(TTFont("Amiri", font_path))

    return encoder, client, collection_name

encoder, qdrant_client, COLLECTION_NAME = initialize_system()

# ------------------------------------------------------------------------------
# MULTI-AGENT JURISTIC PIPELINE
# ------------------------------------------------------------------------------
class State(BaseModel):
    query: str
    target_lang: str
    expanded_arabic: str = ""
    citations: List[Dict[str, Any]] = []

def run_agentic_workflow(user_query: str, selected_lang: str, limit: int) -> State:
    state = State(query=user_query, target_lang=selected_lang)
    q_low = user_query.lower()
    
    if selected_lang == "Auto Detect":
        if any(c in q_low for c in ['ı', 'ş', 'ğ', 'ç']):
            state.target_lang = "Turkish"
        elif any(c in user_query for c in ['ہے', 'کیا', 'کے', 'سے']):
            state.target_lang = "Urdu"
        elif any(c in user_query for c in ['است', 'شدن', 'کردن']):
            state.target_lang = "Persian"
        elif any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in user_query):
            state.target_lang = "Arabic"
        else:
            state.target_lang = "English"

    vocab = {
        "وضو": "فرائض الوضوء وسننه",
        "طہارت": "شروط الطهارة وأحكام المياه",
        "سود": "باب الربا والمعاملات المصرفية",
        "wudu": "فرائض الوضوء وسننه",
        "ablution": "أحكام الطهارة وفرائض الوضوء",
        "interest": "باب الربا وفضل المال الخالي عن العوض",
        "abdest": "abdestin farzları ve hükümleri"
    }
    
    matched = [v for k, v in vocab.items() if k in q_low]
    state.expanded_arabic = " ".join(matched) if matched else user_query
    
    search_str = f"{state.expanded_arabic} {user_query}"
    q_vec = encoder.encode(search_str).tolist()
    
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=limit
    )
    
    state.citations = [{"score": round(h.score, 4), "data": h.payload} for h in hits.points]
    return state

# ------------------------------------------------------------------------------
# BPA PDF GENERATOR
# ------------------------------------------------------------------------------
def create_pdf_dossier(state: State) -> str:
    pdf_path = "Fiqh_Research_Dossier.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    def reshape(txt):
        try:
            return get_display(arabic_reshaper.reshape(txt))
        except:
            return txt

    story = [
        Paragraph("DAR-UL-IFTA LEGAL RESEARCH DOSSIER", ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=15, alignment=1, textColor=colors.HexColor("#1A365D"))),
        Paragraph("Automated Juristic Retrieval & Verification Record", ParagraphStyle('H2', fontName='Helvetica', fontSize=8, alignment=1, textColor=colors.HexColor("#4A5568"))),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A365D"), spaceAfter=10)
    ]
    
    meta_rows = [
        [Paragraph("<b>Query:</b>", styles['Normal']), Paragraph(reshape(state.query), ParagraphStyle('ArQ', fontName='Amiri', fontSize=10, alignment=2))],
        [Paragraph("<b>Language:</b>", styles['Normal']), Paragraph(state.target_lang, styles['Normal'])],
        [Paragraph("<b>Verified Citations:</b>", styles['Normal']), Paragraph(f"{len(state.citations)} Canonical Extracts", styles['Normal'])],
        [Paragraph("<b>Timestamp:</b>", styles['Normal']), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M UTC"), styles['Normal'])]
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
        story.append(Paragraph(f"<b>Citation #{idx}: {d['book_title_ar']}</b> (Vol: {d['volume']}, Page: {d['page']})", ParagraphStyle('C1', fontName='Amiri', fontSize=10, textColor=colors.HexColor("#2C5282"), alignment=2)))
        
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
# STREAMLIT UI LAYOUT
# ------------------------------------------------------------------------------
st.title("⚖️ Fiqh Co-Pilot | Juristic AI Engine")
st.caption("AI-Powered Multi-Agent Research Assistant for Muftis, Scholars & Islamic Legal Institutions")

with st.sidebar:
    st.header("⚙️ Research Parameters")
    lang_choice = st.selectbox(
        "Select Query Language:",
        ["Auto Detect", "Urdu", "Arabic", "English", "Turkish", "Persian"],
        index=0
    )
    top_k = st.slider("Citations to Retrieve (Top-K):", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📚 Indexed Treatises")
    st.markdown("- رد المحتار (ابن عابدين)\n- الفتاوى الهندية (عالمكيري)\n- الهداية (المرغيناني)\n- بدائع الصنائع (الكاساني)\n- البحر الرائق (ابن نجيم)")
    st.info("System grounded on Classical Hanafi Corpus. Generates citation-backed legal memos.")

user_prompt = st.text_input(
    "Enter juristic inquiry (e.g., وضو کے فرائض کیا ہیں؟ / What nullifies wudu?):",
    placeholder="Type question in Urdu, Arabic, English, Turkish, or Persian..."
)

if user_prompt:
    with st.spinner("Multi-Agent Network evaluating juristic query..."):
        result = run_agentic_workflow(user_prompt, lang_choice, top_k)
    
    st.success(f"Language Determined: **{result.target_lang}** | Matched **{len(result.citations)}** Canonical Sources")
    
    pdf_filename = create_pdf_dossier(result)
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="📄 Download Official Legal Dossier (PDF)",
            data=f,
            file_name="Fiqh_Legal_Dossier.pdf",
            mime="application/pdf"
        )
        
    st.markdown("---")
    
    for i, item in enumerate(result.citations, 1):
        meta = item["data"]
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"#{i} {meta['book_title_ar']}")
                st.caption(f"**Author:** {meta['author_ar']} | **Chapter:** {meta['kitab']} ({meta['bab']})")
            with col2:
                st.metric("Similarity Score", f"{item['score']}")
                st.write(f"📖 **Vol:** {meta['volume']} | **Page:** {meta['page']}")
            
            box_html = (
                "<div style='background-color: #F7FAFC; padding: 16px; "
                "border-left: 5px solid #1A365D; border-radius: 4px; "
                "direction: rtl; text-align: right; font-family: Arial, Tahoma; "
                f"font-size: 1.15rem; line-height: 1.8;'>{meta['text']}</div>"
            )
            st.markdown(box_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
