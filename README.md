\# 🤖 Knowledge Base Agent (Offline \& Free)



\## 📌 Overview

This project implements a \*\*Knowledge Base AI Agent\*\* that allows users to upload PDF documents and ask natural language questions based on their content.



The agent processes documents locally using \*\*semantic embeddings\*\* and retrieves the most relevant information from uploaded PDFs.  

It runs \*\*completely free\*\*, with \*\*no paid APIs\*\* and is suitable for students and academic projects.



---



\## 🎯 Objective

To build an AI-powered Knowledge Base Agent that:

\- Accepts PDF documents

\- Understands user queries

\- Retrieves relevant information using vector similarity search

\- Runs locally and can be deployed online



---



\## 🏗️ Architecture

High-level workflow:



User  

↓  

Streamlit Web Interface  

↓  

PDF Upload  

↓  

Text Chunking (RecursiveCharacterTextSplitter)  

↓  

Embedding Model (HuggingFace Sentence Transformer)  

↓  

Vector Database (ChromaDB)  

↓  

Similarity Search  

↓  

Relevant Answer Display  



---



\## 🚀 Features

\- Upload one or multiple PDF documents  

\- Automatic document chunking  

\- Semantic search using embeddings  

\- Displays most relevant document snippets as answers  

\- Shows source content for transparency  

\- Fully free and offline-capable  

\- Easy-to-use Streamlit UI  



---



\## 🧠 AI Techniques Used

\- \*\*Sentence Embeddings\*\* for semantic understanding  

\- \*\*Vector Similarity Search\*\* for retrieval  

\- \*\*Retrieval-Based AI Agent\*\* (Knowledge Base Agent)



---



\## 🛠️ Tech Stack

\- \*\*Programming Language:\*\* Python  

\- \*\*Frontend/UI:\*\* Streamlit  

\- \*\*Framework:\*\* LangChain \& LangChain Community  

\- \*\*Document Loader:\*\* PyPDFLoader  

\- \*\*Text Splitter:\*\* RecursiveCharacterTextSplitter  

\- \*\*Embeddings Model:\*\*  

&nbsp; - `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)  

\- \*\*Vector Database:\*\* ChromaDB  



---



\## ⚙️ Installation \& Setup (Local)



\### ✅ Prerequisites

\- Python 3.9 or above

\- Internet required only for first-time installation



\### ✅ Steps



1\. Clone the repository:

&nbsp;  ```bash

&nbsp;  git clone <your-repository-url>

&nbsp;  cd <repository-folder>



