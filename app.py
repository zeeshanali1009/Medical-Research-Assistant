import streamlit as st
import os
import uuid
from backend.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Advanced Medical Research Assistant", layout="wide")
st.title("Advanced Medical Research Assistant")

# ensure data directories
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/memory", exist_ok=True)

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if 'session_id' not in st.session_state:
    # generate a per-user session id (persisted in Streamlit session)
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Upload documents")
    uploaded = st.file_uploader("Upload PDFs or DOCX (multiple allowed)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Ingest documents"):
        if not uploaded:
            st.warning("Upload files first")
        else:
            # save uploaded files temporarily and ingest
            paths = []
            for f in uploaded:
                fp = f"./data/{f.name}"
                with open(fp, "wb") as fh:
                    fh.write(f.getbuffer())
                paths.append(fp)
            st.session_state.pipeline.ingest(paths)
            st.success("Documents ingested")

st.header("Ask a question about the uploaded documents")
query = st.text_input("Enter question")
if st.button("Ask"):
    if not query:
        st.warning("Write a question first")
    else:
        if not st.session_state.pipeline.store:
            st.warning("Ingest documents first")
        else:
            resp = st.session_state.pipeline.query(query, session_id=st.session_state.session_id)
            st.subheader("Answer")
            st.write(resp['answer'])
            with st.expander("Sources"):
                for s in resp.get('sources', []):
                    st.write(s)
            with st.expander("Conversation History (recent)"):
                for q, a in resp.get('history', []):
                    st.markdown(f"**Q:** {q}\n\n**A:** {a}\n\n---")
