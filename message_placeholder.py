from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate(
    ('system', 'You are a helpful customer service assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
)

# load chat history
chat_history = []
with open('chat.history.txt') as f: # for practical use need to connect to db
    chat_history.extend(f.readlines())

print(chat_history)

# Dynamic prompt with chat history
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})   

# now the llm has complete context with past messages with MessagesPlaceholder() 