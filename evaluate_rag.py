from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "pdf_database"

TOP_K = 4


# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "What is machine learning?",
        "expected_keywords": [
            "machine learning",
            "data"
        ]
    },

    {
        "question": "What are the main types of machine learning?",
        "expected_keywords": [
            "supervised",
            "unsupervised"
        ]
    },

    {
        "question": "What is supervised learning?",
        "expected_keywords": [
            "labeled",
            "data"
        ]
    },

    {
        "question": "What is unsupervised learning?",
        "expected_keywords": [
            "unlabeled",
            "data"
        ]
    }
]


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("🔄 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

print("📚 Loading PDF database...")

vectorstore = FAISS.load_local(
    DATABASE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Database loaded!")


# ============================================================
# EVALUATION
# ============================================================

total_questions = len(test_questions)

passed_questions = 0


print("\n")
print("=" * 60)
print("📊 RAG RETRIEVAL EVALUATION")
print("=" * 60)


for number, test in enumerate(
    test_questions,
    start=1
):

    question = test["question"]

    expected_keywords = (
        test["expected_keywords"]
    )


    print("\n")
    print(f"QUESTION {number}")
    print("-" * 60)

    print(question)


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    results = vectorstore.similarity_search(
        question,
        k=TOP_K
    )


    # --------------------------------------------------------
    # COMBINE RETRIEVED TEXT
    # --------------------------------------------------------

    retrieved_text = "\n".join(
        document.page_content
        for document in results
    ).lower()


    # --------------------------------------------------------
    # CHECK KEYWORDS
    # --------------------------------------------------------

    found_keywords = []

    missing_keywords = []


    for keyword in expected_keywords:

        if keyword.lower() in retrieved_text:

            found_keywords.append(
                keyword
            )

        else:

            missing_keywords.append(
                keyword
            )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (
        len(found_keywords)
        /
        len(expected_keywords)
        *
        100
    )


    print("\nRetrieved chunks:")

    for index, document in enumerate(
        results,
        start=1
    ):

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

        print(
            f"\nChunk {index}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Page: {page}"
        )

        print(
            document.page_content[:300]
            .replace("\n", " ")
        )


    print("\nExpected keywords:")

    print(
        ", ".join(expected_keywords)
    )


    print("\nFound:")

    if found_keywords:

        print(
            "✅ " +
            ", ".join(found_keywords)
        )

    else:

        print("❌ None")


    if missing_keywords:

        print("\nMissing:")

        print(
            "❌ " +
            ", ".join(missing_keywords)
        )


    print(
        f"\nRetrieval score: {score:.0f}%"
    )


    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    if score >= 50:

        print("✅ PASS")

        passed_questions += 1

    else:

        print("❌ FAIL")


# ============================================================
# FINAL REPORT
# ============================================================

accuracy = (
    passed_questions
    /
    total_questions
    *
    100
)


print("\n")
print("=" * 60)
print("📊 FINAL RAG EVALUATION")
print("=" * 60)

print(
    f"Questions tested: {total_questions}"
)

print(
    f"Questions passed: {passed_questions}"
)

print(
    f"Retrieval accuracy: {accuracy:.0f}%"
)

print("=" * 60)