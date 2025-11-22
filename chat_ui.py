import streamlit as st
import requests

st.set_page_config(page_title="RAG Manual Assistant", layout="wide")

st.title("📘 Manual Assistant — RAG System")
st.write("Ask questions based on the uploaded technical manual (PDF).")

API_URL = "http://localhost:8000/query"

query = st.text_input(
    "Enter your question:",
    placeholder="Example: How do I reset the configuration in the Settings section?"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("⚠ Please enter a question")
    else:
        payload = {"query": query, "max_chunks": 5}
        with st.spinner("Retrieving answer..."):
            try:
                response = requests.post(API_URL, json=payload)
                data = response.json()

                st.subheader("🟢 Answer")
                st.write(data["answer"])

                st.subheader("⏱ Performance")
                st.json({
                    "retrieval_latency": data["retrieval_latency"],
                    "generation_latency": data["generation_latency"],
                    "used_filter": data["used_filter"]
                })

                with st.expander("📄 Context Used From PDF"):
                    for item in data.get("context_snippets", []):
                        meta = item["metadata"]
                        st.markdown(
                            f"**📌 Chapter:** {meta.get('chapter')}  |  **📄 Page:** {meta.get('page')}"
                        )
                        st.write(item["text"])
                        st.markdown("---")

            except Exception as e:
                st.error("❌ Cannot reach backend. Make sure `python query_api.py` is running.")
                st.exception(e)
