import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Page title
st.set_page_config(page_title='santu Summarization App')

# Display the logo at the top of the page
st.image("images.jpg")  # Adjust width as needed
st.divider()  # Horizontal rule
st.title('🦜🔗 santu Summarization App')
st.divider()  # Horizontal rule

# Get API Key from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found. Please add it to Streamlit secrets.")
    st.stop()

def generate_response(txt):
    # Instantiate the LLM model
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)
    
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(txt)
    
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
if txt_input.strip():  # Ensure input is not empty
    with st.form('summarize_form', clear_on_submit=True):
        submitted = st.form_submit_button('Submit')
        
        if submitted:
            with st.spinner('Summarizing...'):
                response = generate_response(txt_input)
                st.success("Summarization Complete:")
                st.info(response)
else:
    st.warning("Please enter text to summarize.")
