from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
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

model = ChatHuggingFace(llm=llm)

messages = [ 
    SystemMessage(content="You are an helpful assistant"),
    HumanMessage(content="Explain Langchain to me")
]

result = model.invoke(messages)

messages.append(AIMessage(content= result.content))

print(messages)
