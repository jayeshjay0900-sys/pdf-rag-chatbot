import os
import re
import json
import time
import hashlib
import secrets
import urllib.request
import urllib.error
from pathlib import Path

import streamlit as st
import numpy as np

try:
    import fitz
except ImportError:
    fitz = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# APP CONFIGURATION
# ============================================================

APP_VERSION = "Step 21 Local"
DB_VERSION = "v13"

DEFAULT_MODEL = "llama3.2:3b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_UPLOAD_MB = 25

SEMANTIC_CANDIDATES = 10

DEFAULT_FINAL_DOCS = 4
DEFAULT_MAX_CONTEXT = 7000

PDF_ROOT = Path("uploaded_pdfs")
DB_ROOT = Path("pdf_database")
CHAT_FILE = Path("chat_history.json")
SETTINGS_FILE = Path("user_settings.json")

OLLAMA_URL = "http://127.0.0.1:11434"


# ============================================================
# CREATE LOCAL STORAGE
# ============================================================

PDF_ROOT.mkdir(parents=True, exist_ok=True)
DB_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path, default):

    try:

        if path.exists():

            return json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

    except Exception:

        pass

    return default


def save_json(path, data):

    path.write_text(

        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_SETTINGS = {

    "model": DEFAULT_MODEL,

    "temperature": 0.2,

    "final_docs":
        DEFAULT_FINAL_DOCS,

    "max_context":
        DEFAULT_MAX_CONTEXT,

    "theme":
        "Dark"

}


settings = load_json(

    SETTINGS_FILE,

    DEFAULT_SETTINGS.copy()

)


MODEL = settings.get(

    "model",

    DEFAULT_MODEL

)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="PDF RAG Chatbot",

    page_icon="📚",

    layout="wide"

)


# ============================================================
# THEME
# ============================================================

theme = settings.get(

    "theme",

    "Dark"

)


# ============================================================
# THEME SELECTOR
# ============================================================

with st.sidebar:

    chosen_theme = st.selectbox(

        "🎨 Theme",

        ["Dark", "Light"],

        index=(
            0
            if theme == "Dark"
            else 1
        )

    )


if chosen_theme != theme:

    settings["theme"] = chosen_theme

    save_json(
        SETTINGS_FILE,
        settings
    )

    st.rerun()


# ============================================================
# THEME COLORS
# ============================================================

if chosen_theme == "Dark":

    BG = "#0f1117"
    FG = "#f5f7fa"

    SIDEBAR_BG = "#151821"

    CARD = "#1b1f29"

    INPUT_BG = "#20242e"

    INPUT_BORDER = "#3a4150"

    MUTED = "#aeb6c4"

    BUTTON_BG = "#20242e"

    BUTTON_HOVER = "#2b3140"

    CHAT_USER = "#303542"

    CHAT_ASSISTANT = "#1b1f29"

    EXPANDER_BG = "#181c24"

    CODE_BG = "#11151c"

    BORDER = "#343b48"

else:

    BG = "#ffffff"
    FG = "#111827"

    SIDEBAR_BG = "#f8fafc"

    CARD = "#ffffff"

    INPUT_BG = "#ffffff"

    INPUT_BORDER = "#cbd5e1"

    MUTED = "#64748b"

    BUTTON_BG = "#ffffff"

    BUTTON_HOVER = "#f1f5f9"

    CHAT_USER = "#f1f5f9"

    CHAT_ASSISTANT = "#ffffff"

    EXPANDER_BG = "#f8fafc"

    CODE_BG = "#f1f5f9"

    BORDER = "#d1d5db"


# ============================================================
# COMPLETE THEME CSS
# ============================================================

st.markdown(

    f"""
    <style>

    /* =====================================================
       GLOBAL APP
       ===================================================== */

    .stApp {{
        background-color: {BG} !important;
        color: {FG} !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {BG} !important;
        color: {FG} !important;
    }}

    [data-testid="stMain"] {{
        background-color: {BG} !important;
        color: {FG} !important;
    }}

    .main {{
        background-color: {BG} !important;
        color: {FG} !important;
    }}

    section.main {{
        background-color: {BG} !important;
    }}


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        color: {FG} !important;
    }}

    [data-testid="stSidebar"] > div {{
        background-color: {SIDEBAR_BG} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {FG} !important;
    }}


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1, h2, h3, h4, h5, h6 {{
        color: {FG} !important;
    }}

    p {{
        color: {FG};
    }}

    label {{
        color: {FG} !important;
    }}

    .stCaption,
    [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}


    /* =====================================================
       TEXT INPUTS
       ===================================================== */

    input {{
        background-color: {INPUT_BG} !important;
        color: {FG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    input::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}

    textarea {{
        background-color: {INPUT_BG} !important;
        color: {FG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}


    /* =====================================================
       STREAMLIT BASE INPUT CONTAINERS
       ===================================================== */

    [data-baseweb="input"] {{
        background-color: {INPUT_BG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    [data-baseweb="input"] > div {{
        background-color: {INPUT_BG} !important;
    }}

    [data-baseweb="textarea"] {{
        background-color: {INPUT_BG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    [data-baseweb="textarea"] > div {{
        background-color: {INPUT_BG} !important;
    }}


    /* =====================================================
       SELECTBOX
       ===================================================== */

    [data-baseweb="select"] {{
        background-color: {INPUT_BG} !important;
    }}

    [data-baseweb="select"] > div {{
        background-color: {INPUT_BG} !important;
        color: {FG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    [data-baseweb="select"] * {{
        color: {FG} !important;
    }}

    [role="option"] {{
        background-color: {CARD} !important;
        color: {FG} !important;
    }}

    [role="option"]:hover {{
        background-color: {BUTTON_HOVER} !important;
    }}


    /* =====================================================
       SLIDERS
       ===================================================== */

    [data-testid="stSlider"] {{
        color: {FG} !important;
    }}

    [data-testid="stSlider"] * {{
        color: {FG} !important;
    }}


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button,
    .stDownloadButton > button {{
        background-color: {BUTTON_BG} !important;
        color: {FG} !important;

        border: 1px solid {INPUT_BORDER} !important;

        border-radius: 10px;

        min-height: 40px;

        transition: all 0.15s ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background-color: {BUTTON_HOVER} !important;
        color: {FG} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    .stButton > button p,
    .stDownloadButton > button p {{
        color: {FG} !important;
    }}


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {{
        background-color: {CARD} !important;
        color: {FG} !important;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {CARD} !important;
        border-color: {INPUT_BORDER} !important;
    }}

    [data-testid="stFileUploader"] * {{
        color: {FG} !important;
    }}


    /* =====================================================
       CHAT MESSAGES
       ===================================================== */

    [data-testid="stChatMessage"] {{
        color: {FG} !important;
    }}

    [data-testid="stChatMessage"] p {{
        color: {FG} !important;
    }}

    [data-testid="stChatMessage"] li {{
        color: {FG} !important;
    }}

    [data-testid="stChatMessage"] strong {{
        color: {FG} !important;
    }}

    [data-testid="stChatMessage"] code {{
        background-color: {CODE_BG} !important;
        color: {FG} !important;
    }}


    /* =====================================================
       CHAT INPUT
       ===================================================== */

    [data-testid="stChatInput"] {{
        background-color: {BG} !important;
    }}

    [data-testid="stChatInput"] > div {{
        background-color: {INPUT_BG} !important;

        border: 1px solid {INPUT_BORDER} !important;

        border-radius: 14px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background-color: {INPUT_BG} !important;

        color: {FG} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {MUTED} !important;
    }}

    [data-testid="stChatInput"] button {{
        color: {FG} !important;
    }}


    /* =====================================================
       EXPANDERS
       ===================================================== */

    [data-testid="stExpander"] {{
        background-color: {EXPANDER_BG} !important;

        border: 1px solid {BORDER} !important;

        border-radius: 10px !important;
    }}

    [data-testid="stExpander"] summary {{
        color: {FG} !important;
    }}

    [data-testid="stExpander"] summary span {{
        color: {FG} !important;
    }}

    [data-testid="stExpander"] * {{
        color: {FG} !important;
    }}


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {{
        color: {FG} !important;
    }}

    [data-testid="stAlert"] p {{
        color: {FG} !important;
    }}


    /* =====================================================
       CODE BLOCKS
       ===================================================== */

    code {{
        background-color: {CODE_BG} !important;
        color: {FG} !important;
    }}

    pre {{
        background-color: {CODE_BG} !important;
        color: {FG} !important;
        border-radius: 10px;
    }}


    /* =====================================================
       MARKDOWN LINKS
       ===================================================== */

    a {{
        color: #2563eb !important;
    }}


    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {{
        border-color: {BORDER} !important;
    }}


    /* =====================================================
       CUSTOM RAG CARD
       ===================================================== */

    .rag-card {{
        padding: 12px 16px;

        border-radius: 12px;

        background-color: {CARD};

        border: 1px solid {BORDER};

        margin: 8px 0;

        color: {FG};
    }}


    /* =====================================================
       SCROLLBAR
       ===================================================== */

    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {BG};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {BORDER};
        border-radius: 10px;
    }}

    </style>
    """,

    unsafe_allow_html=True

)


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(

        model_name=EMBED_MODEL

    )


# ============================================================
# CACHED OLLAMA MODEL
# ============================================================

@st.cache_resource
def get_llm(

    model,

    temperature

):

    return OllamaLLM(

        model=model,

        temperature=temperature

    )


embeddings = get_embeddings()


llm = get_llm(

    MODEL,

    float(

        settings.get(

            "temperature",

            0.2

        )

    )

)


# ============================================================
# OLLAMA CONNECTION CHECK
# ============================================================

def check_ollama():

    try:

        with urllib.request.urlopen(

            f"{OLLAMA_URL}/api/tags",

            timeout=2

        ) as response:

            if response.status != 200:

                return False, []


            raw = response.read().decode(

                "utf-8"

            )


            data = json.loads(raw)


            models = [

                m.get(
                    "name",
                    ""
                )

                for m in data.get(
                    "models",
                    []
                )

            ]


            return True, models


    except Exception:

        return False, []


# ============================================================
# SAFE OLLAMA INVOCATION
# ============================================================

def safe_llm_invoke(prompt):

    try:

        result = llm.invoke(prompt)


        if result is None:

            return (

                None,

                "🔴 Ollama returned "
                "an empty response."

            )


        result = str(result).strip()


        if not result:

            return (

                None,

                "🔴 Ollama returned "
                "an empty response."

            )


        return result, None


    except Exception as e:

        error_text = str(e).lower()


        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        if (

            "winerror 10061"
            in error_text

            or "connection refused"
            in error_text

            or "connecterror"
            in error_text

            or "actively refused"
            in error_text

            or "failed to connect"
            in error_text

        ):

            return (

                None,

                "🔴 **Ollama is not running.**\n\n"

                "Open another VS Code terminal "
                "and run:\n\n"

                "```powershell\n"
                "ollama run llama3.2:3b\n"
                "```\n\n"

                "If necessary, start the Ollama "
                "server with:\n\n"

                "```powershell\n"
                "ollama serve\n"
                "```\n\n"

                f"Selected model: `{MODEL}`"

            )


        # ----------------------------------------------------
        # MODEL NOT FOUND
        # ----------------------------------------------------

        if (

            "model" in error_text

            and (

                "not found"
                in error_text

                or "pull"
                in error_text

            )

        ):

            return (

                None,

                f"🔴 **Ollama model not found.**\n\n"

                f"Selected model: `{MODEL}`\n\n"

                "Run:\n\n"

                "```powershell\n"

                f"ollama pull {MODEL}\n"

                "```"

            )


        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        return (

            None,

            "🔴 **Ollama error occurred.**\n\n"

            f"```text\n{str(e)}\n```"

        )


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():

    data = load_json(

        CHAT_FILE,

        {}

    )


    if isinstance(data, dict):

        return data


    return {}


def save_chats(chats):

    save_json(

        CHAT_FILE,

        chats

    )


def new_chat_id():

    return (

        time.strftime(
            "%Y%m%d_%H%M%S"
        )

        + "_"

        + secrets.token_hex(3)

    )


if "chats" not in st.session_state:

    st.session_state.chats = load_chats()


if not st.session_state.chats:

    cid = new_chat_id()

    st.session_state.chats[cid] = []

    st.session_state.current_chat = cid

    save_chats(

        st.session_state.chats

    )


if "current_chat" not in st.session_state:

    st.session_state.current_chat = next(

        iter(
            st.session_state.chats
        )

    )


# ============================================================
# VECTOR DATABASE
# ============================================================

def load_db():

    try:

        return FAISS.load_local(

            str(DB_ROOT),

            embeddings,

            allow_dangerous_deserialization=True

        )

    except Exception:

        return None


def build_db():

    pdfs = list(

        PDF_ROOT.glob("*.pdf")

    )


    if not pdfs:

        if DB_ROOT.exists():

            for x in DB_ROOT.iterdir():

                if x.is_file():

                    try:

                        x.unlink()

                    except Exception:

                        pass


        return None


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150,

        separators=[

            "\n\n",

            "\n",

            ". ",

            "? ",

            "! ",

            ", ",

            " ",

            ""

        ]

    )


    docs = []


    for pdf in pdfs:

        try:

            loaded = PyPDFLoader(

                str(pdf)

            ).load()


            chunks = splitter.split_documents(

                loaded

            )


            for i, d in enumerate(chunks):

                d.metadata.update({

                    "document_name":
                        pdf.name,

                    "source":
                        pdf.name,

                    "page_number":
                        int(

                            d.metadata.get(

                                "page",

                                0

                            )

                        ) + 1,

                    "chunk_id":
                        i,

                    "content_type":
                        "pdf"

                })


                docs.append(d)


        except Exception as e:

            st.warning(

                f"Could not read "
                f"{pdf.name}: {e}"

            )


    if not docs:

        return None


    db = FAISS.from_documents(

        docs,

        embeddings

    )


    db.save_local(

        str(DB_ROOT)

    )


    return db


if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = load_db()


vectorstore = st.session_state.vectorstore


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

def tokens(s):

    return set(

        re.findall(

            r"[A-Za-z0-9_]+",

            s.lower()

        )

    )


def key(d):

    return (

        d.metadata.get(

            "document_name",

            ""

        ),

        d.metadata.get(

            "page_number",

            0

        ),

        d.metadata.get(

            "chunk_id",

            0

        )

    )


def phrase_score(q, t):

    q = " ".join(

        re.findall(

            r"\w+",

            q.lower()

        )

    )


    t = t.lower()


    if len(q) > 8 and q in t:

        return 1.0


    words = q.split()


    if len(words) < 2:

        return 0


    pairs = [

        " ".join(

            words[i:i + 2]

        )

        for i in range(

            len(words) - 1

        )

    ]


    return (

        sum(

            x in t

            for x in pairs

        )

        / len(pairs)

    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(

    query,

    final_n

):

    current_vectorstore = (

        st.session_state.get(

            "vectorstore"

        )

    )


    if not current_vectorstore:

        return []


    total_documents = len(

        current_vectorstore.index_to_docstore_id

    )


    semantic = (

        current_vectorstore

        .similarity_search_with_score(

            query,

            k=min(

                SEMANTIC_CANDIDATES,

                max(

                    1,

                    total_documents

                )

            )

        )

    )


    docs = []

    seen = set()


    for d, dist in semantic:

        if key(d) not in seen:

            docs.append(d)

            seen.add(

                key(d)

            )


    texts = [

        d.page_content

        for d in docs

    ]


    if texts:

        try:

            vec = TfidfVectorizer(

                stop_words="english",

                ngram_range=(1, 2),

                sublinear_tf=True

            )


            vec.fit(texts)


            sims = cosine_similarity(

                vec.transform([query]),

                vec.transform(texts)

            )[0]


        except Exception:

            sims = np.zeros(

                len(texts)

            )

    else:

        sims = []


    qwords = tokens(query)

    scored = []


    for i, d in enumerate(docs):

        stoks = tokens(

            d.page_content

        )


        cov = (

            len(

                qwords & stoks

            )

            /

            max(

                1,

                len(qwords)

            )

        )


        ph = phrase_score(

            query,

            d.page_content

        )


        sem = (

            1

            /

            (

                1

                +

                max(

                    float(

                        semantic[i][1]

                    ),

                    0

                )

            )

            if i < len(semantic)

            else 0

        )


        score = (

            0.50 * sem

            +

            0.25 * float(

                sims[i]

                if len(sims)

                else 0

            )

            +

            0.15 * cov

            +

            0.10 * ph

        )


        scored.append(

            (

                score,

                d

            )

        )


    scored.sort(

        key=lambda x: x[0],

        reverse=True

    )


    out = []

    pages = {}

    docs_seen = {}


    for score, d in scored:

        pg = (

            d.metadata.get(

                "document_name"

            ),

            d.metadata.get(

                "page_number"

            )

        )


        dn = d.metadata.get(

            "document_name"

        )


        if pages.get(

            pg,

            0

        ) >= 2:

            continue


        if docs_seen.get(

            dn,

            0

        ) >= 3:

            continue


        out.append(

            (

                score,

                d

            )

        )


        pages[pg] = (

            pages.get(

                pg,

                0

            )

            + 1

        )


        docs_seen[dn] = (

            docs_seen.get(

                dn,

                0

            )

            + 1

        )


        if len(out) >= final_n:

            break


    return out


# ============================================================
# FOLLOW-UP QUERY DETECTION
# ============================================================

def is_follow_up(question):

    return bool(

        re.match(

            r"^(what about|why|how|give|explain|and |then |that|this|it|example)",

            question.lower()

        )

    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(

    question,

    history

):

    follow = (

        bool(history)

        and is_follow_up(question)

    )


    search_q = question


    # ========================================================
    # FOLLOW-UP QUERY REWRITING
    # ========================================================

    if follow:

        recent_messages = []


        for m in history[-6:]:

            role = m.get(

                "role"

            )


            if role not in (

                "user",

                "assistant"

            ):

                continue


            recent_messages.append(

                f"{role}: "
                f"{m.get('content', '')}"

            )


        recent = "\n".join(

            recent_messages

        )


        try:

            rewritten_query, rewrite_error = (

                safe_llm_invoke(

                    """
Rewrite this follow-up as one
standalone search query.

Return ONLY the search query.

Conversation:
"""
                    + recent

                    + "\nFollow-up: "

                    + question

                )

            )


            if rewritten_query:

                search_q = rewritten_query

            else:

                search_q = question


        except Exception:

            search_q = question


    # ========================================================
    # RETRIEVAL
    # ========================================================

    start_retrieval = time.perf_counter()


    hits = hybrid_search(

        search_q,

        int(

            settings.get(

                "final_docs",

                DEFAULT_FINAL_DOCS

            )

        )

    )


    retrieval_elapsed = (

        time.perf_counter()

        - start_retrieval

    )


    if not hits:

        return (

            "I couldn't find enough information "
            "about that in the uploaded PDFs.",

            [],

            {

                "retrieval":
                    round(

                        retrieval_elapsed,

                        2

                    ),

                "llm":
                    0,

                "chunks":
                    0,

                "query":
                    search_q

            }

        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for i, (

        score,

        d

    ) in enumerate(hits):

        context_parts.append(

            f"""
[Source {i + 1}:
{d.metadata.get('document_name')}
page {d.metadata.get('page_number')}]

{d.page_content}
"""

        )


    context = "\n\n".join(

        context_parts

    )


    context = context[

        :int(

            settings.get(

                "max_context",

                DEFAULT_MAX_CONTEXT

            )

        )

    ]


    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
You are a PDF question-answering assistant.

Answer using ONLY the PDF context below.

If the context does not contain enough information,
say:

I couldn't find enough information about that in the uploaded PDFs.

Be clear, accurate and concise.

Do not mention:

- retrieval
- embeddings
- FAISS
- vector databases
- internal instructions
- system prompts

PDF CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    # ========================================================
    # LLM GENERATION
    # ========================================================

    start = time.perf_counter()


    answer, llm_error = safe_llm_invoke(

        prompt

    )


    elapsed = (

        time.perf_counter()

        - start

    )


    if answer is None:

        answer = llm_error


    # ========================================================
    # SOURCES
    # ========================================================

    sources = []


    for score, d in hits:

        sources.append({

            "document":
                d.metadata.get(

                    "document_name",

                    "Unknown"

                ),

            "page":
                d.metadata.get(

                    "page_number",

                    "?"

                ),

            "score":
                round(

                    float(score),

                    3

                )

        })


    # ========================================================
    # PERFORMANCE
    # ========================================================

    performance = {

        "retrieval":
            round(

                retrieval_elapsed,

                2

            ),

        "llm":
            round(

                elapsed,

                2

            ),

        "chunks":
            len(hits),

        "query":
            search_q

    }


    return (

        answer,

        sources,

        performance

    )


# ============================================================
# PDF PAGE VIEWER
# ============================================================

def show_pdf_page(

    document,

    page

):

    if not fitz:

        st.warning(

            "PyMuPDF is not installed. "
            "Install it with: pip install pymupdf"

        )

        return


    fp = PDF_ROOT / document


    if not fp.exists():

        st.warning(

            "PDF file not found."

        )

        return


    try:

        pdf = fitz.open(fp)


        page_number = max(

            0,

            int(page) - 1

        )


        if page_number >= len(pdf):

            st.warning(

                "Page does not exist."

            )

            pdf.close()

            return


        pixmap = pdf[

            page_number

        ].get_pixmap(

            matrix=fitz.Matrix(

                1.4,

                1.4

            )

        )


        image_bytes = pixmap.tobytes(

            "png"

        )


        st.image(

            image_bytes,

            caption=(

                f"{document} "
                f"— Page {page}"

            )

        )


        pdf.close()


    except Exception as e:

        st.error(

            f"Could not display page: {e}"

        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 PDF RAG")


    st.caption(

        f"{APP_VERSION} • "
        f"{DB_VERSION} • "
        f"Local Ollama"

    )


    # ========================================================
    # OLLAMA STATUS
    # ========================================================

    ollama_ok, ollama_models = check_ollama()


    if ollama_ok:

        if MODEL in ollama_models:

            st.success(

                f"🟢 Ollama connected\n\n"
                f"🤖 {MODEL}"

            )

        else:

            st.warning(

                f"🟡 Ollama connected, "
                f"but model `{MODEL}` "
                f"was not found."

            )


            st.caption(

                "Available models: "

                +

                (

                    ", ".join(

                        ollama_models

                    )

                    if ollama_models

                    else "None"

                )

            )

    else:

        st.error(

            "🔴 Ollama is not connected"

        )


        st.caption(

            "Run `ollama run llama3.2:3b` "
            "in another terminal."

        )


    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(

        "➕ New Chat",

        use_container_width=True

    ):

        cid = new_chat_id()


        st.session_state.chats[cid] = []


        st.session_state.current_chat = cid


        save_chats(

            st.session_state.chats

        )


        st.rerun()


    # ========================================================
    # SEARCH CONVERSATIONS
    # ========================================================

    search = st.text_input(

        "🔎 Search conversations"

    )


    for cid, msgs in (

        st.session_state.chats.items()

    ):

        title = "New Chat"


        # ----------------------------------------------------
        # MANUALLY RENAMED CHAT
        # ----------------------------------------------------

        for m in msgs:

            if (

                m.get("role")

                == "system"

                and m.get("chat_title")

            ):

                title = m.get(

                    "chat_title"

                )

                break


        # ----------------------------------------------------
        # FIRST USER MESSAGE
        # ----------------------------------------------------

        if title == "New Chat":

            for m in msgs:

                if m.get(

                    "role"

                ) == "user":

                    title = m.get(

                        "content",

                        ""

                    )[:40]

                    break


        if search:

            if (

                search.lower()

                not in title.lower()

            ):

                continue


        if st.button(

            title or "New Chat",

            key="chat_" + cid,

            use_container_width=True

        ):

            st.session_state.current_chat = cid

            st.rerun()


    st.divider()


    # ========================================================
    # PDF LIBRARY
    # ========================================================

    st.subheader(

        "📄 PDF Library"

    )


    pdfs = list(

        PDF_ROOT.glob("*.pdf")

    )


    st.write(

        f"{len(pdfs)} PDF(s)"

    )


    upload = st.file_uploader(

        "Add PDF",

        type=["pdf"],

        key="pdf_upload"

    )


    if upload:

        if upload.size > (

            MAX_UPLOAD_MB

            * 1024

            * 1024

        ):

            st.error(

                f"PDF is larger than "
                f"{MAX_UPLOAD_MB} MB."

            )

        elif st.button(

            "⬆️ Save PDF & Rebuild",

            use_container_width=True

        ):

            name = Path(

                upload.name

            ).name


            file_path = (

                PDF_ROOT / name

            )


            file_path.write_bytes(

                upload.getbuffer()

            )


            st.session_state.vectorstore = (

                build_db()

            )


            st.success(

                f"Added {name}"

            )


            st.rerun()


    # ========================================================
    # LIST PDFS
    # ========================================================

    for pdf in pdfs:

        c1, c2 = st.columns(

            [4, 1]

        )


        c1.caption(

            pdf.name

        )


        delete_key = (

            "del_"

            +

            hashlib.md5(

                pdf.name.encode()

            ).hexdigest()

        )


        if c2.button(

            "🗑️",

            key=delete_key

        ):

            try:

                pdf.unlink()

            except Exception as e:

                st.error(

                    f"Could not delete PDF: {e}"

                )

                continue


            st.session_state.vectorstore = (

                build_db()

            )


            st.rerun()


    # ========================================================
    # REBUILD DATABASE
    # ========================================================

    if st.button(

        "🔄 Rebuild Database",

        use_container_width=True

    ):

        with st.spinner(

            "Rebuilding FAISS database..."

        ):

            st.session_state.vectorstore = (

                build_db()

            )


        st.success(

            "Database rebuilt."

        )


        st.rerun()


    st.divider()


    # ========================================================
    # SETTINGS
    # ========================================================

    st.subheader(

        "⚙️ Settings"

    )


    new_model = st.text_input(

        "Ollama model",

        value=MODEL

    )


    temp = st.slider(

        "Temperature",

        0.0,

        1.0,

        float(

            settings.get(

                "temperature",

                0.2

            )

        ),

        0.05

    )


    final_docs = st.slider(

        "Retrieved chunks",

        1,

        8,

        int(

            settings.get(

                "final_docs",

                DEFAULT_FINAL_DOCS

            )

        )

    )


    max_context = st.slider(

        "Max context",

        2000,

        12000,

        int(

            settings.get(

                "max_context",

                DEFAULT_MAX_CONTEXT

            )

        ),

        500

    )


    if st.button(

        "💾 Save Settings",

        use_container_width=True

    ):

        settings.update({

            "model":
                new_model,

            "temperature":
                temp,

            "final_docs":
                final_docs,

            "max_context":
                max_context,

            "theme":
                chosen_theme

        })


        save_json(

            SETTINGS_FILE,

            settings

        )


        st.cache_resource.clear()


        st.success(

            "Settings saved. Reloading..."

        )


        time.sleep(

            0.5

        )


        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title(

    "📚 PDF RAG Chatbot"

)


st.caption(

    f"{APP_VERSION} • "
    f"Advanced RAG • "
    f"Local Ollama • "
    f"{MODEL}"

)


# ============================================================
# DATABASE STATUS
# ============================================================

vectorstore = st.session_state.get(

    "vectorstore"

)


if vectorstore is None:

    st.info(

        "📄 Upload a PDF from the sidebar "
        "and rebuild the database to start."

    )

else:

    st.success(

        "✅ PDF database ready"

    )


# ============================================================
# CURRENT CHAT
# ============================================================

msgs = st.session_state.chats[

    st.session_state.current_chat

]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for idx, m in enumerate(msgs):

    role = m.get(

        "role",

        "assistant"

    )


    # --------------------------------------------------------
    # HIDE INTERNAL SYSTEM METADATA
    # --------------------------------------------------------

    if role == "system":

        continue


    with st.chat_message(

        role

    ):

        st.markdown(

            m.get(

                "content",

                ""

            )

        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if m.get("sources"):

            with st.expander(

                "📚 Sources"

            ):

                for si, s in enumerate(

                    m["sources"]

                ):

                    document = s.get(

                        "document",

                        "Unknown"

                    )


                    page = s.get(

                        "page",

                        "?"

                    )


                    score = s.get(

                        "score"

                    )


                    source_text = (

                        f"📄 {document} "
                        f"— Page {page}"

                    )


                    if score is not None:

                        source_text += (

                            f" — Relevance: "
                            f"{score}"

                        )


                    st.write(

                        source_text

                    )


                    if st.button(

                        f"👁️ View Page {page}",

                        key=(

                            f"hist_"

                            f"{st.session_state.current_chat}_"

                            f"{idx}_"

                            f"{si}"

                        )

                    ):

                        show_pdf_page(

                            document,

                            page

                        )


        # ----------------------------------------------------
        # DOWNLOAD ANSWER
        # ----------------------------------------------------

        if (

            role == "assistant"

            and idx == len(msgs) - 1

        ):

            st.download_button(

                "📥 Download answer",

                m.get(

                    "content",

                    ""

                ).encode(),

                file_name="answer.txt",

                key=(

                    f"dl_"

                    f"{st.session_state.current_chat}_"

                    f"{idx}"

                )

            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(

    "Ask something about your PDFs..."

)


if question:

    if vectorstore is None:

        st.warning(

            "Please upload a PDF first."

        )

        st.stop()


    # ========================================================
    # USER MESSAGE
    # ========================================================

    msgs.append({

        "role":
            "user",

        "content":
            question

    })


    save_chats(

        st.session_state.chats

    )


    with st.chat_message(

        "user"

    ):

        st.markdown(

            question

        )


    # ========================================================
    # AI ANSWER
    # ========================================================

    with st.chat_message(

        "assistant"

    ):

        with st.spinner(

            "🔎 Searching PDFs "
            "and generating answer..."

        ):

            ans, sources, perf = (

                answer_question(

                    question,

                    msgs[:-1]

                )

            )


        st.markdown(

            ans

        )


        # ====================================================
        # SOURCES
        # ====================================================

        if sources:

            with st.expander(

                "📚 Sources",

                expanded=False

            ):

                for si, s in enumerate(

                    sources

                ):

                    st.write(

                        f"📄 "
                        f"{s['document']} "
                        f"— Page "
                        f"{s['page']} "
                        f"— Relevance: "
                        f"{s['score']}"

                    )


                    if st.button(

                        f"👁️ View Page {s['page']}",

                        key=(

                            f"now_"

                            f"{st.session_state.current_chat}_"

                            f"{len(msgs)}_"

                            f"{si}"

                        )

                    ):

                        show_pdf_page(

                            s["document"],

                            s["page"]

                        )


            st.caption(

                f"⚡ Retrieval: "
                f"{perf.get('retrieval', 0):.2f}s "
                f"• LLM: "
                f"{perf.get('llm', 0):.2f}s "
                f"• Chunks: "
                f"{perf.get('chunks', 0)}"

            )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    msgs.append({

        "role":
            "assistant",

        "content":
            ans,

        "sources":
            sources,

        "performance":
            perf

    })


    save_chats(

        st.session_state.chats

    )


    st.rerun()


# ============================================================
# CHAT CONTROLS
# ============================================================

st.divider()


c1, c2, c3 = st.columns(

    3

)


# ============================================================
# RENAME CHAT
# ============================================================

with c1:

    if st.button(

        "✏️ Rename Chat",

        use_container_width=True

    ):

        st.session_state.rename_mode = True


# ============================================================
# CLEAR CHAT
# ============================================================

with c2:

    if st.button(

        "🧹 Clear Chat",

        use_container_width=True

    ):

        st.session_state.chats[

            st.session_state.current_chat

        ] = []


        save_chats(

            st.session_state.chats

        )


        st.rerun()


# ============================================================
# EXPORT CHAT
# ============================================================

with c3:

    export = json.dumps(

        msgs,

        indent=2,

        ensure_ascii=False

    )


    st.download_button(

        "📥 Export Chat",

        export,

        file_name=(

            f"chat_"

            f"{st.session_state.current_chat}"

            f".json"

        ),

        use_container_width=True

    )


# ============================================================
# RENAME INTERFACE
# ============================================================

if st.session_state.get(

    "rename_mode"

):

    current_title = "New Chat"


    for m in msgs:

        if (

            m.get("role")

            == "system"

            and m.get("chat_title")

        ):

            current_title = m.get(

                "chat_title"

            )

            break


    new_title = st.text_input(

        "New chat name",

        value=current_title,

        key="rename_input"

    )


    if st.button(

        "💾 Save Name"

    ):

        # ----------------------------------------------------
        # REMOVE OLD TITLE
        # ----------------------------------------------------

        msgs[:] = [

            m for m in msgs

            if not (

                m.get("role")

                == "system"

                and m.get("chat_title")

            )

        ]


        # ----------------------------------------------------
        # ADD NEW TITLE
        # ----------------------------------------------------

        msgs.insert(

            0,

            {

                "role":
                    "system",

                "chat_title":
                    new_title[:80]

            }

        )


        save_chats(

            st.session_state.chats

        )


        st.session_state.rename_mode = False


        st.rerun()