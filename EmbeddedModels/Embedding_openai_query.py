from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# for documents
docs = {
    "Gangtok is the capital of Sikkim",
    "Patna is the capital of Bihar",
    "Bengaluru is the capital of Karnataka"
}

result = embedding.embed_query("Gangtok is the capital of Sikkim")

# for documents 
result = embedding.embed_documents(docs)

print(str(result))