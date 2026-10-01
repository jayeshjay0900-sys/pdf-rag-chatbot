from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load PDF
loader = PyPDFLoader(
    "documents/Machine_Learning_10_Page_Knowledge_Base"
    "
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    "
    ".pdf"
)

documents = loader.load()

print(f"Loaded {len(documents)} pages")


# 2. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")


# 4. Create FAISS database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# 5. Save database
vectorstore.save_local("faiss_index")

print("✅ Database created successfully!")