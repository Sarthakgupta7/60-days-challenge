from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------
# 1️⃣ Initialize Components
# ---------------------------

llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings()

# ---------------------------
# 2️⃣ Sample Documents
# ---------------------------

documents = [
    Document(page_content="Artificial Intelligence enables machines to learn from data."),
    Document(page_content="Machine Learning is a subset of AI focused on learning patterns."),
    Document(page_content="Deep Learning uses neural networks with multiple layers."),
    Document(page_content="RAG systems combine retrieval with generation for better answers.")
]

# ---------------------------
# 3️⃣ Create Vector Store
# ---------------------------

vectorstore = FAISS.from_documents(documents, embeddings)

# ---------------------------
# 4️⃣ Retrieval Function
# ---------------------------

def retrieve(query, k=2):
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# ---------------------------
# 5️⃣ Generate Answer
# ---------------------------

def generate_answer(query, docs):
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content

# ---------------------------
# 6️⃣ Evaluation Function
# ---------------------------

def evaluate(query, docs, answer):
    print("\n--- Evaluation ---")


    print("Retrieved Documents:", len(docs))


    for i, doc in enumerate(docs):
        print(f"Doc {i+1}: {doc.page_content}")


    print("\nAnswer Length:", len(answer.split()))

    if query.lower().split()[0] in answer.lower():
        print("Relevance: Likely relevant")
    else:
        print("Relevance: Needs improvement")


query = "What is Machine Learning?"

print("\nUser Query:", query)

docs = retrieve(query)
answer = generate_answer(query, docs)

print("\nGenerated Answer:\n")
print(answer)

evaluate(query, docs, answer)
