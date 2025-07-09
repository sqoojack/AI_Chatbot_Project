# streamlit run app.py
import streamlit as st
import re   # 這裡有改
from qa_backend import run_qa, load_documents_from_upload   # 這裡有改

st.title("🤖 AI_RAG問答機器人")

uploaded_files = st.file_uploader("**可上傳文字檔(txt, pdf) 或圖片檔(jpg, jpeg, png)**", type=["txt", "pdf", "jpg", "png"], accept_multiple_files=True)      # 這行有改

# ollama_url = "http://172.20.5.116:11434"
# ollama_url = "http://140.113.24.231:11434"   # PCheng's lab server website, Ollama port always is 11434
ollama_url = "http://127.0.0.1:11434"   # After ssh server connect

query = st.text_input("請輸入你的問題", value="請問抽菸要罰多少錢？")

# let thinking progress become collapsible(可收放的)
def to_collapsible(text: str) -> str:
    return re.sub(
        r"<think>([\s\S]*?)</think>",
        r"<details><summary>思考過程</summary>\1</details>",
        text)

if st.button("回答問題"):
    if uploaded_files and query:
        with st.spinner("處理中..."):
            try:
                all_docs = []   #  這段有改
                for f in uploaded_files:
                    all_docs.extend(load_documents_from_upload(f))
                raw = run_qa(all_docs, query, ollama_url)
                
                st.success("以下是模型回答:")
                result = to_collapsible(raw)
                st.markdown(result, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"錯誤：{e}")
    else:
        st.warning("請先上傳檔案並輸入問題。")