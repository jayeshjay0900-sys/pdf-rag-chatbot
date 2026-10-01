import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "pdf_database"

MODEL_NAME = "llama3.2:3b"

TOP_K = 4


# ============================================================
# TEST QUESTIONS
# ============================================================

tests = [

    {
        "question": "What is machine learning?",
        "keywords": [
            "machine learning",
            "data"
        ]
    },

    {
        "question": "What are the main types of machine learning?",
        "keywords": [
            "supervised",
            "unsupervised"
        ]
    },

    {
        "question": "What is supervised learning?",
        "keywords": [
            "supervised",
            "labeled",
            "data"
        ]
    },

    {
        "question": "What is unsupervised learning?",
        "keywords": [
            "unsupervised",
            "unlabeled",
            "data"
        ]
    },

    {
        "question": "What is deep learning?",
        "keywords": [
            "deep learning"
        ]
    }
]


# ============================================================
# LOAD MODELS
# ============================================================

print("🔄 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🔄 Loading FAISS database...")

vectorstore = FAISS.load_local(
    DATABASE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("🔄 Loading Ollama...")

llm = OllamaLLM(
    model=MODEL_NAME
)

print("✅ Everything loaded!")


# ============================================================
# EVALUATION
# ============================================================

retrieval_scores = []

answer_scores = []


print("\n")
print("=" * 65)
print("📊 REAL RAG EVALUATION")
print("=" * 65)


for number, test in enumerate(
    tests,
    start=1
):

    question = test["question"]

    keywords = test["keywords"]


    print("\n")
    print(f"QUESTION {number}")
    print("-" * 65)

    print(question)


    # ========================================================
    # RETRIEVAL
    # ========================================================

    documents = vectorstore.similarity_search(
        question,
        k=TOP_K
    )


    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # ========================================================
    # RETRIEVAL SCORE
    # ========================================================

    context_lower = context.lower()

    found_retrieval = []


    for keyword in keywords:

        if keyword.lower() in context_lower:

            found_retrieval.append(
                keyword
            )


    retrieval_score = (
        len(found_retrieval)
        /
        len(keywords)
        *
        100
    )


    retrieval_scores.append(
        retrieval_score
    )


    print(
        f"\n🔎 Retrieval score: "
        f"{retrieval_score:.0f}%"
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    prompt = f"""
You are a PDF research assistant.

Answer the question using ONLY
the provided PDF context.

If the answer cannot be found,
say:

"I couldn't find that information
in the uploaded PDFs."

PDF CONTEXT:

{context}

QUESTION:

{question}

Give a short and clear answer.
"""


    print("🤖 Generating answer...")


    answer = llm.invoke(
        prompt
    ).strip()


    print("\nAnswer:")

    print(answer)


    # ========================================================
    # ANSWER SCORE
    # ========================================================

    answer_lower = answer.lower()

    found_answer = []


    for keyword in keywords:

        if keyword.lower() in answer_lower:

            found_answer.append(
                keyword
            )


    answer_score = (
        len(found_answer)
        /
        len(keywords)
        *
        100
    )


    answer_scores.append(
        answer_score
    )


    print(
        f"\n💬 Answer score: "
        f"{answer_score:.0f}%"
    )


    # ========================================================
    # SOURCES
    # ========================================================

    print("\n📄 Sources:")


    shown = set()


    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_number",
            document.metadata.get(
                "page",
                "Unknown"
            )
        )


        source_key = (
            source,
            page
        )


        if source_key not in shown:

            print(
                f"  • {source} — Page {page}"
            )

            shown.add(
                source_key
            )


# ============================================================
# FINAL RESULTS
# ============================================================

average_retrieval = (
    sum(retrieval_scores)
    /
    len(retrieval_scores)
)


average_answer = (
    sum(answer_scores)
    /
    len(answer_scores)
)


overall_score = (
    average_retrieval
    +
    average_answer
) / 2


print("\n")
print("=" * 65)
print("🏆 FINAL RAG EVALUATION")
print("=" * 65)

print(
    f"Questions tested: "
    f"{len(tests)}"
)

print(
    f"Average retrieval score: "
    f"{average_retrieval:.0f}%"
)

print(
    f"Average answer score: "
    f"{average_answer:.0f}%"
)

print(
    f"Overall RAG score: "
    f"{overall_score:.0f}%"
)

print("=" * 65)