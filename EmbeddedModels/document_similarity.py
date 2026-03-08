from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The scientific study of celestial objects and phenomena is known as astronomy.",
    "A healthy diet consists of a variety of fruits, vegetables, and whole grains.",
    "Artificial intelligence is transforming the way we process and analyze large datasets.",
    "The capital city of France is Paris, which is famous for the Eiffel Tower.",
    "Regular physical exercise can significantly improve cardiovascular health and mood."
]

query = "Can you tell me which city serves as the administrative center of France?"
# query = "What exercise is good for cardiovascular health"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# values inside cosine similarity should always be in 2-D list
scores = cosine_similarity([query_embedding], doc_embedding)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score is ", score)
