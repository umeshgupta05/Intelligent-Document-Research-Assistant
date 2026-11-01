import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import Agent
from memory import ChatMemory
from rag import HybridRAG

load_dotenv()

st.set_page_config(page_title="Intelligent Document Research Assistant", layout="wide")
st.title("Intelligent Document Research Assistant")


@st.cache_resource
def build():
    rag = HybridRAG()
    memory = ChatMemory()
    return rag, memory, Agent(rag, memory)


rag, memory, agent = build()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Index documents", disabled=not uploads):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in uploads:
                path = Path(tmp) / f.name
                path.write_bytes(f.getvalue())
                paths.append(path)
            added = rag.add_files(paths)
        st.success(f"Added {added} chunks")
    if rag.files:
        st.caption("Indexed: " + ", ".join(sorted(rag.files)))
    if st.button("Clear documents"):
        rag.reset()
        st.rerun()
    if st.button("Clear conversation"):
        memory.clear()
        st.rerun()

for m in memory.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["source"]:
            st.caption(f"Source: {m['source']}")

question = st.chat_input("Ask about your documents or anything else")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            result = agent.run(question)
        st.markdown(result["answer"])
        st.caption(f"Source: {result['source']}")
        st.caption("Steps: " + ", ".join(result["steps"]))
        if result["references"]:
            with st.expander("References"):
                for ref in result["references"]:
                    st.markdown(ref)
    memory.add("user", question)
    memory.add("assistant", result["answer"], result["source"])
