import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Knowledge Base Agent", page_icon="🤖")
st.title("🤖 Knowledge Base Agent (Offline & Free)")
st.write(
    "Upload PDF documents and ask questions. The app finds the most relevant parts "
    "from your documents using local AI embeddings (no API keys needed)."
)

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
if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents with local embeddings..."):
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

# ---------- CHAT / QUERY INTERFACE ----------
if st.session_state.vectorstore:
    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if query := st.chat_input("Ask a question about your documents..."):
        # Store and show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Get relevant document chunks (no LLM, just smart search)
        with st.chat_message("assistant"):
            with st.spinner("Searching in your documents..."):
                docs = st.session_state.vectorstore.similarity_search(query, k=3)

                if not docs:
                    answer = "I couldn't find relevant information in the uploaded documents."
                    st.write(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    snippets = []
                    for i, d in enumerate(docs, start=1):
                        snippet = d.page_content.strip().replace("\n", " ")
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."
                        snippets.append(f"**Snippet {i}:** {snippet}")

                    answer = (
                        "Here are the most relevant parts I found in your documents:\n\n"
                        + "\n\n---\n\n".join(snippets)
                    )

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

            # Optional: show raw sources
            with st.expander("📄 Full source chunks"):
                for d in docs:
                    st.write(d.page_content)
                    st.markdown("---")
else:
    st.info("👆 Upload documents above and click **Process Documents** to get started.")
