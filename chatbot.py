from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

# for context/ memory of the chat
chat_history = [
    SystemMessage(content="You are an helpful AI Assistant that answers in points")
]


while True:
    user_input = input('You : ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ",result.content)

print(chat_history)