from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# for documents
docs = {
    "Gangtok is the capital of Sikkim",
    "Patna is the capital of Bihar",
    "Bengaluru is the capital of Karnataka"
}

text = "India is in the Final of T20 World Cup"

vector = embedding.embed_query(text)

# for documents 
vector_docs = embedding.embed_documents(docs)


print(str(vector_docs))
