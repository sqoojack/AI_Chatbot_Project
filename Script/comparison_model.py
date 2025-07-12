# comparison_model.py

import streamlit as st
import re
from backend import run_qa, load_documents_from_upload
from log_model import save_qa_log_to_csv
from datetime import datetime
import os
import json
import csv
from tqdm import tqdm     # 顯示目前進度
ollama_url = "http://172.20.5.116:11434"

available_img_models = [
    "gemma3:27b",
    "llava:7b",
    "llava-llama3:8b",
    "llama4:16x17b"
]
# files_path = r"C:\Users\CCCheng2\Desktop\Langchain\dataset\pics\\"

current_dir = os.getcwd()   # 取得目前的工作目錄
files_path = os.path.join(current_dir, "Dataset\\")   # 組合成 目前目錄的dataset資料夾

uploaded_files = [
    "dog.jpg",
    "cat.jpg",
    "國語字典.jpg",
    "熊.jpg",
    "車牌.jpg",
    "成績單.jpg",
    "川普.jpg",
    "monkey.jpg",
]

querys = [
    "圖片中有哪些動物?",
    "圖片中有哪些狗的品種?",
    "圖片中的這隻猴子正在做甚麼?",
    "圖片中有哪些有名人物",
    "圖片中車牌號碼是幾號",
    "圖片中查到了哪些中文字的說明?",
]

class FakeUpload:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._path = path

    def read(self):
        with open(self._path, "rb") as f:
            return f.read()

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


total_iter = len(querys) * len(available_img_models) * 5
p_bar = tqdm(total=total_iter, desc="Current progress", ncols=120)

# 全部的img_model都有測
for query in querys:
    for img_model in available_img_models:
        for i in range(5):
            all_docs = []
            file_names = []
            tn_1 = datetime.now()
            # 替換這段 loop：
            for f in uploaded_files:
                file_names.append(f)
                fake_file = FakeUpload(files_path + f)
                all_docs.extend(load_documents_from_upload(fake_file, ollama_url, img_model))
            tn_2 = datetime.now()

            try:
                tn_3 = datetime.now()
                result = run_qa(
                    all_docs,
                    query,
                    ollama_url,
                    # llm_model="deepseek-r1:8b",
                    llm_model="gemma3:27b",
                    # llm_model="llama4:16x17b",
                    temperature = 0.0,
                    top_p = 0.9,
                    top_k = 10,
                )
                tn_4 = datetime.now()
                answer, thought = extract_answer_and_thought(result["answer"])

                save_qa_log_to_csv(
                    xlsx_path="Output/Models_comparison_gemma.xlsx",
                    question=query,
                    answer=answer,
                    thought=thought,
                    # llm_model="deepseek-r1:8b",
                    llm_model="gemma3:27b",
                    # llm_model="llama4:16x17b",
                    img_model=img_model,
                    temperature = 0.0,
                    top_p= 0.9,
                    top_k= 10,
                    file_names=file_names,
                    retrieval_time=(tn_2 - tn_1).total_seconds(),
                    generation_time=(tn_4 - tn_3).total_seconds(),
                    chunks=result["chunks"]
                )

            except Exception as e:
                st.error(f"錯誤：{e}")

            p_bar.update(1)     # 每執行一次進度就加1