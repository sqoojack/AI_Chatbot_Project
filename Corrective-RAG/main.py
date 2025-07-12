
from crag import run_corrective_rag
# streamlit run app.py
import streamlit as st
import re
from qa_backend import run_qa, load_documents_from_upload
available_models = [
    "deepseek-r1:8b",
    "qwen3:14b",
    "qwen2.5:7b",
    "gemma3:27b",
    "deepseek-r1:32b"
]

st.title("🤖 AI_RAG問答機器人")

# 預設參數（可儲存）
if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "llm_model": "deepseek-r1:8b",
        "temperature": 0.7,
        "top_p": 0.9,
    }

# 按鈕點擊後的小視窗（popover）
with st.popover("⚙️ 模型參數設定", use_container_width=True):
    llm_model = st.selectbox(
        "選擇模型",
        options=available_models,
        index=available_models.index(st.session_state.model_settings["llm_model"]) if st.session_state.model_settings["llm_model"] in available_models else 0
    )
    temperature = st.slider("溫度 (temperature)", 0.0, 1.0, st.session_state.model_settings["temperature"])
    top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.model_settings["top_p"])

    if st.button("✅ 儲存參數", key="save_model_settings"):
        st.session_state.model_settings = {
            "llm_model": llm_model,
            "temperature": temperature,
            "top_p": top_p,
        }
        st.success("已儲存參數！")

uploaded_files = st.file_uploader("**可上傳文字檔(txt, pdf) 或圖片檔(jpg, jpeg, png)**", type=["txt", "pdf", "jpg", "png"], accept_multiple_files=True)      # 這行有改
# ollama_url = "http://172.20.5.116:11434"
ollama_url = "http://127.0.0.1:11434"    # Pohan's lab server

query = st.text_area("請輸入你的問題", value="華雄的絕學是什麼？他的死法為何？")

# let thinking progress become collapsible(可收放的)
def extract_answer_and_thought(text: str):
    # 拆出 <think> 和回答
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    if match:
        thought = match.group(1).strip()
        answer = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    else:
        answer = text.strip()
        thought = ""
    return answer, thought


if st.button("回答問題"):
    if uploaded_files and query:
        with st.spinner("處理中..."):
            try:
                all_docs = []
                for f in uploaded_files:
                    all_docs.extend(load_documents_from_upload(f, ollama_url))
                result = run_corrective_rag(
                    docs=all_docs,
                    query=query,
                    ollama_url=ollama_url,
                    llm_model=st.session_state.model_settings["llm_model"],
                    embedding_model="bge-m3",
                    temperature=st.session_state.model_settings["temperature"],
                    top_p=st.session_state.model_settings["top_p"],
                    k=5
                )

                # 左右分欄
                left_col, right_col = st.columns([2, 1])

                with left_col:
                    st.markdown("### 📘 回答：")
                    answer, thought = extract_answer_and_thought(result["answer"])
                    st.markdown(answer)

                    if thought:
                        with st.expander("💭 思考過程"):
                            st.markdown(thought)

                with right_col:
                    st.markdown("### 🔍 匹配段落：")
                    for i, chunk in enumerate(result["chunks"]):
                        with st.expander(f"Chunk #{i+1}（分數: {chunk['score']:.4f}）"):
                            st.write(chunk["content"])

            except Exception as e:
                st.error(f"錯誤：{e}")
    else:
        st.warning("請先上傳檔案並輸入問題。")

# 載入檔案與參數後


# 顯示初稿與最終校正回答
st.markdown("### 💭 初稿回答：")
st.markdown(result["draft"])
st.markdown("### 📘 最終校正回答：")
st.markdown(result["answer"])