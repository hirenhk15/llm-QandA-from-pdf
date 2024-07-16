# LLM App to extract key data from a product review
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from enum import Enum
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Steps:
# 1. Input .pdf file
# 2. Format file
# 3. Split file
# 4. Create embeddings
# 5. Store embeddings in vector store
# 6. Input query
# 7. Run QA chain
# 8. Output

CREATIVITY=0
# os.environ["TOKENIZERS_PARALLELISM"] = False


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e


class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.sidebar.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.sidebar.success("Received valid API Key!")
            else:
                st.sidebar.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.sidebar.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Q&A from a PDF Document")
            st.markdown("<h1 style='text-align: center;'>Q&A from a PDF Document</h1>", unsafe_allow_html=True)

            # Select the model provider
            option_model_provider = st.sidebar.selectbox(
                    'Select the model provider',
                    ('GroqCloud', 'OpenAI')
                )

            # Input API Key for model to query
            api_key = self.get_api_key()

            # Upload a PDF file
            uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

            # Get the question from user
            query_text = st.text_input(
                "Enter your question:",
                placeholder="Write your question here",
                disabled=not uploaded_file
            )

            submitted = st.button("Submit", disabled=not (uploaded_file and query_text))
            if submitted:
                if not api_key:
                    st.warning("Please insert your API Key", icon="⚠️")
                    st.stop()
                    
                with st.spinner("Wait, please. I am working on it..."):
                    # Format file
                    reader = PdfReader(uploaded_file)
                    formatted_document = [page.extract_text() for page in reader.pages]

                    # Split the document
                    text_splitter = CharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=0
                    )
                    docs = text_splitter.create_documents(formatted_document)

                    # Create embeddings and store into FAISS vector db
                    embeddings = HuggingFaceEmbeddings()
                    vectorstore = FAISS.from_documents(docs, embeddings)

                    # Load the LLM model
                    llm_model = LLMModel(model_provider=option_model_provider)
                    llm = llm_model.load(api_key=api_key)

                    # Create retrieval chain for Q&A
                    retrieval_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever()
                    )
                    answer = retrieval_chain.invoke(query_text)

                    st.info(answer["result"])
                    del api_key


        except Exception as e:
            st.error(str(e), icon=":material/error:")



def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()