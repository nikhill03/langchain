from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token= hf_token,
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

while True:
    user_input = input('You : ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI : ",result.content)