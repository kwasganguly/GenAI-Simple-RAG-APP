# app.py

import streamlit as st
from rag_pipeline import RAGPipeline
import tempfile
import os

st.set_page_config(page_title="Simple RAG App")

st.title("🔎 Simple RAG with LangChain")

api_key = st.text_input("🔑 OpenAI API Key", type="password")
file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
#url = st.text_input("🌐 Or paste a webpage URL")
query = st.text_input("💬 Ask something about the document")

if api_key:
    rag = RAGPipeline(api_key)

    if (file) and st.button("🔄 Process"):
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            docs = rag.load_documents(file_path=tmp_path)
            os.remove(tmp_path)
        else:
            #docs = rag.load_documents(url=url)
            st.write("No document provided")

        rag.build_vectorstore(docs)
        st.success("✅ Document processed and indexed.")

        # Store rag in session just for the current interaction
        st.session_state["rag"] = rag

# Querying
if "rag" in st.session_state and query and st.button("💡 Get Answer"):
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state["rag"].query(query)
            st.subheader("🤖 Answer")
            st.write(result['result'])

            st.subheader("📚 Sources")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}**")
                st.write(doc.page_content[:400] + "...")
        except Exception as e:
            st.error(f"❌ Error: {e}")

