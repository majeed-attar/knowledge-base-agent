import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama
import os

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Knowledge Base Agent", page_icon="🤖")
st.title("🤖 Knowledge Base Agent")
st.write("Upload documents and ask questions! (Runs fully LOCAL using Ollama – no API key needed)")

# ---------- SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- FILE UPLOAD ----------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------- PROCESS DOCUMENTS ----------
# ---------- PROCESS DOCUMENTS ----------
if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents with local models..."):
        all_docs = []

        for file in uploaded_files:
            # Save temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Load and split
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)
            all_docs.extend(splits)

            # Cleanup temp file
            os.remove(temp_path)

        # Create vector store using FREE HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings
        )

        st.success(
            f"✅ Processed {len(all_docs)} chunks from {len(uploaded_files)} documents!"
        )


# ---------- CHAT INTERFACE ----------
if st.session_state.vectorstore:
    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with local model (mistral)..."):
                # Local LLM from Ollama
                llm = ChatOllama(model="mistral", temperature=0)

                # RetrievalQA chain using local LLM + local embeddings
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True
                )

                result = qa(query)
                response = result["result"]

                st.write(response)

                # Show sources
                with st.expander("📄 Sources from your PDFs"):
                    for doc in result["source_documents"][:3]:
                        st.write(doc.page_content[:300] + " ...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
else:
    st.info("👆 Upload documents above and click **Process Documents** to get started.")
