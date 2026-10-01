from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load the FAISS database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. Create the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 4. Load the local LLM
llm = OllamaLLM(
    model="llama3.2:3b"
)


print("\n📚 PDF RAG Chatbot")
print("Type 'exit' to quit.\n")


# 5. Start chatbot
while True:

    question = input("You: ")

    if question.lower() == "exit":
        break


    # Find relevant information from the PDF
    documents = retriever.invoke(question)


    # Combine retrieved text
    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # Create prompt for the LLM
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not present in the context,
say "I couldn't find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


    # Ask Llama 3.2
    answer = llm.invoke(prompt)


    # Display answer
    print("\nAI:", answer)
    print()