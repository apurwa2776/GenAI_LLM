import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from huggingface_hub import login

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Hugging Face Authentication
login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    # LLama2 model loading
    llm = CTransformers(
        model='D:/Resume_Projects/Blog_Generation_LLM_App/models/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Your local model path
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    
    # Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    # Create the prompt
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)
    
    # Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

# Streamlit App Configuration
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

# Input field for blog topic
input_text = st.text_input("Enter the Blog Topic")

# Two columns for additional input fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

# Display the generated blog if the button is pressed
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
