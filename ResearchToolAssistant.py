from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=hf_token,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

# Dynamic Inputs
paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code- Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')


if st.button('Summarize'):
    # Chains
    chain = template | model
    
    result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })

    st.text(result.content)