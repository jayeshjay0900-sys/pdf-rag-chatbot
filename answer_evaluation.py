from langchain_ollama import OllamaLLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    {
        "question": "What is machine learning?",

        "expected_answer": (
            "Machine learning is a branch of artificial "
            "intelligence where computers learn patterns "
            "from data."
        ),

        "expected_keywords": [
            "machine learning",
            "artificial intelligence",
            "data"
        ]
    },

    {
        "question": "What are the main types of machine learning?",

        "expected_answer": (
            "The main types include supervised learning "
            "and unsupervised learning."
        ),

        "expected_keywords": [
            "supervised",
            "unsupervised"
        ]
    },

    {
        "question": "What is supervised learning?",

        "expected_answer": (
            "Supervised learning uses labeled data "
            "to learn patterns."
        ),

        "expected_keywords": [
            "supervised",
            "labeled",
            "data"
        ]
    },

    {
        "question": "What is unsupervised learning?",

        "expected_answer": (
            "Unsupervised learning works with unlabeled "
            "data to discover patterns."
        ),

        "expected_keywords": [
            "unsupervised",
            "unlabeled",
            "data"
        ]
    }
]


# ============================================================
# LOAD OLLAMA
# ============================================================

print("🤖 Loading Ollama...")

llm = OllamaLLM(
    model=MODEL_NAME
)

print("✅ Ollama loaded!")


# ============================================================
# EVALUATION
# ============================================================

total_score = 0

passed = 0

total_questions = len(test_cases)


print("\n")
print("=" * 60)
print("🧪 RAG ANSWER QUALITY EVALUATION")
print("=" * 60)


for number, test in enumerate(
    test_cases,
    start=1
):

    question = test["question"]

    expected_answer = test["expected_answer"]

    expected_keywords = test["expected_keywords"]


    print("\n")
    print(f"QUESTION {number}")
    print("-" * 60)

    print(question)


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    prompt = f"""
You are evaluating an AI answer.

Question:
{question}

Expected answer:
{expected_answer}

Important concepts that should appear:
{", ".join(expected_keywords)}

Create a short correct answer to the question.

Use only the information represented
by the expected answer.

Do not add unrelated information.

Answer:
"""


    print("\n🤖 Generating answer...")

    answer = llm.invoke(
        prompt
    ).strip()


    print("\nGenerated answer:")
    print(answer)


    # ========================================================
    # KEYWORD CHECK
    # ========================================================

    answer_lower = answer.lower()

    found = []

    missing = []


    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            found.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )


    # ========================================================
    # SCORE
    # ========================================================

    score = (
        len(found)
        /
        len(expected_keywords)
        *
        100
    )


    print("\nExpected concepts:")

    print(
        ", ".join(expected_keywords)
    )


    print("\nFound concepts:")

    if found:

        print(
            "✅ " +
            ", ".join(found)
        )

    else:

        print("❌ None")


    if missing:

        print("\nMissing concepts:")

        print(
            "❌ " +
            ", ".join(missing)
        )


    print(
        f"\nAnswer score: {score:.0f}%"
    )


    # ========================================================
    # PASS / FAIL
    # ========================================================

    if score >= 66:

        print("✅ PASS")

        passed += 1

    else:

        print("❌ FAIL")


    total_score += score


# ============================================================
# FINAL REPORT
# ============================================================

average_score = (
    total_score
    /
    total_questions
)


print("\n")
print("=" * 60)
print("📊 FINAL ANSWER EVALUATION")
print("=" * 60)

print(
    f"Questions tested: {total_questions}"
)

print(
    f"Questions passed: {passed}"
)

print(
    f"Average answer score: {average_score:.0f}%"
)

print("=" * 60)