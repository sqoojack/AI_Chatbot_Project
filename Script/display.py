
# streamlit run Script/display.py
import streamlit as st
import re
from backend import run_qa, load_documents_from_upload
from log_model import save_qa_log_to_csv
from datetime import datetime
import os
import json
import csv

available_llm_models = [
    "deepseek-r1:8b",
    "qwen3:14b",
    "qwen2.5:7b",
    "gemma3:27b",
    "deepseek-r1:32b",
    "llava:7b",
    "llava-llama3:8b",
    "llama4:16x17b"
]

available_img_models = [
    "gemma3:27b",
    "llava:7b",
    "llava-llama3:8b",
    "llama4:16x17b"
]


st.title("多模態RAG 問答機器人")

# 預設參數（可儲存）
if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "llm_model": "deepseek-r1:8b",
        "img_model": "gemma3:27b",
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 5,

    }

# 按鈕點擊後的小視窗（popover）
with st.popover("⚙️ 模型參數設定", use_container_width=True):
    llm_model = st.selectbox(
        "選擇最終生成LLM",
        options=available_llm_models,
        index=available_llm_models.index(st.session_state.model_settings["llm_model"]) if st.session_state.model_settings["llm_model"] in available_llm_models else 0
    )
    img_model = st.selectbox(
        "選擇圖像分析LLM",
        options=available_img_models,
        index=available_img_models.index(st.session_state.model_settings["img_model"]) if st.session_state.model_settings["img_model"] in available_img_models else 0
    )
    temperature = st.slider("溫度 (temperature)", 0.0, 1.0, st.session_state.model_settings["temperature"])
    top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.model_settings["top_p"])
    top_k = st.slider("Top-k", 2, 40, st.session_state.model_settings["top_k"])

    if st.button("✅ 儲存參數", key="save_model_settings"):
        st.session_state.model_settings = {
            "llm_model": llm_model,
            "img_model": img_model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        st.success("已儲存參數！")

uploaded_files = st.file_uploader("**可上傳文字檔(txt, pdf) 或圖片檔(jpg, jpeg, png)**", type=["txt", "pdf", "jpg", "png"], accept_multiple_files=True)
# ollama_url = "http://172.20.5.116:11434"
ollama_url = "http://127.0.0.1:11434"
query = st.text_area("請輸入你的問題", value="這是甚麼？")

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
    tn_1 = datetime.now()
    all_docs = []
    file_names = []
    for f in uploaded_files:
        file_names.append(f.name)
        all_docs.extend(load_documents_from_upload(f, ollama_url, img_model=st.session_state.model_settings["img_model"]))
    tn_2 = datetime.now()
    if query and len(all_docs) >= 0 :
        with st.spinner("處理中..."):
            try:
                result = run_qa(
                    all_docs,
                    query,
                    ollama_url,
                    llm_model=st.session_state.model_settings["llm_model"],
                    temperature=st.session_state.model_settings["temperature"],
                    top_p=st.session_state.model_settings["top_p"],
                    top_k=st.session_state.model_settings["top_k"],
                )
                tn_3 = datetime.now()
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

                save_qa_log_to_csv(
                    xlsx_path="qa_logs.xlsx",
                    question=query,
                    answer=answer,
                    thought=thought,
                    llm_model=st.session_state.model_settings["llm_model"],
                    img_model=st.session_state.model_settings["img_model"],
                    temperature=st.session_state.model_settings["temperature"],
                    top_p=st.session_state.model_settings["top_p"],
                    top_k=st.session_state.model_settings["top_k"],
                    file_names=file_names,
                    retrieval_time=(tn_2 - tn_1).total_seconds(),
                    generation_time=(tn_3 - tn_2).total_seconds(),
                    chunks=result["chunks"]
                )

            except Exception as e:
                st.error(f"錯誤：{e}")
    else:
        st.warning("檔案無法載入成功或請輸入問題")