
# log_model.py
import csv
import os
from datetime import datetime 
import pandas as pd
import numpy as np

def save_qa_log_to_csv(
    xlsx_path: str,
    question: str,
    answer: str,
    thought: str,
    llm_model: str,
    img_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    file_names: list,
    retrieval_time: float,
    generation_time: float,
    chunks: list
):
    top_chunks = chunks[:2]  # 只記前2個 chunk
    chunk_1_score = top_chunks[0]["score"] if len(top_chunks) > 0 else ""
    chunk_1_content = top_chunks[0]["content"][:100].replace("\n", " ") if len(top_chunks) > 0 else ""
    chunk_2_score = top_chunks[1]["score"] if len(top_chunks) > 1 else ""
    chunk_2_content = top_chunks[1]["content"][:100].replace("\n", " ") if len(top_chunks) > 1 else ""

    headers = [
        "timestamp", "llm_model", "img_model", "temperature", "top_p", "top_k","question", "answer", "thought",
        "retrieval_time_sec", "generation_time_sec", "uploaded_files",
        "chunk_1_score", "chunk_1_content", "chunk_2_score", "chunk_2_content"
    ]
    row = [
        datetime.now().isoformat(),
        llm_model,
        img_model,
        temperature,
        top_p,
        top_k,
        question,
        answer,
        thought,
        round(retrieval_time, 2),
        round(generation_time, 2),
        ";".join(file_names),
        chunk_1_score,
        chunk_1_content,
        chunk_2_score,
        chunk_2_content
    ]
    '''
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    '''
    df_row = pd.DataFrame([row], columns=headers)

    if os.path.exists(xlsx_path):
        existing_df = pd.read_excel(xlsx_path)
        df_row = pd.concat([existing_df, df_row], ignore_index=True)
    df_row.to_excel(xlsx_path, index=False)

    # print("save finish!")