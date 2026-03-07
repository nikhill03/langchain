from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatOpenAI(model="gpt-4")

result = chatmodel.invoke("")
print(result)
# print(result.content)  # for only text answer