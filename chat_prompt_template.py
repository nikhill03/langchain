from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

#dynamic templates
chat_template = ChatPromptTemplate([
    # SystemMessage(content='You are a helpful {domain} expert'),
    # HumanMessage(content='Expalin in simple terms what is {topic}')
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Expalin in simple terms what is {topic}')
])

prompt = chat_template.invoke({'domain' : 'cricket', 'topic' : 'dusra'})
print(prompt)