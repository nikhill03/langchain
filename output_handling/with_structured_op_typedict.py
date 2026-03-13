from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token = hf_token,
    task= 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# schema
class review(TypedDict):

    summary: str
    # summary: Annotated[str, 'A brief summary of the review']
    sentiment: str

structured_output = model.with_structured_output(review)

result = structured_output.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this """)

print(result)
print(result['summary'])
print(result['sentiment'])