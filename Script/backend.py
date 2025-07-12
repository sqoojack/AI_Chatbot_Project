
# backend.py
import requests
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
from scipy.stats import norm
from load_data import load_documents_from_upload
import datetime


# ✅ 自訂 Ollama Embedding 類別
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        vec = np.array(response.json()["embedding"], dtype=np.float32)      # 新增兩行
        vec /= np.linalg.norm(vec) + 1e-8 # 單位化
        
        return vec.tolist()



# ✅ 建立 FAISS 向量資料庫
def build_FAISS(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

# 將 Cosine distance 轉到 0~1 區間的 similarity
def cosine_dist_to_sim(dist: float) -> float:
    cos_sim = 1 - (dist**2) / 2     # 在單位向量下： dist^2 = 2 - 2cosθ  →  cosθ = 1 - dist^2/2
    return float((cos_sim + 1) / 2)     # 線性映射到 [0,1]


# ✅ 執行問答流程
def run_qa(uploaded_file, query: str, ollama_url: str,
        llm_model: str = "deepseek-r1:8b",
        embedding_model: str = "bge-m3",
        temperature: float = 0.7,  top_p: float = 0.9, top_k: int = 5
        ) -> str:
    if isinstance(uploaded_file, list):     # 這段有改
        docs = uploaded_file
    else:
        docs = load_documents_from_upload(uploaded_file)
    # print(f"number of docs: {len(docs) if docs else 'None or empty'}")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    vectorstore = build_FAISS(docs, embeddings)
    # print(f"建立向量成功  vectorstore chunks 數量: {len(vectorstore.docstore._dict)}")

    #建立 Retriever（關鍵步驟）
    chunks_with_scores = vectorstore.similarity_search_with_score(query, k=top_k) # 回傳距離分數
    similarity_score = [(doc, round(cosine_dist_to_sim(dist), 4)) for doc, dist in chunks_with_scores]  # 新增這行, 將distance轉成 0 ~ 1 similarity

    context = "\n\n".join([
        f"段落{i+1}：{chunk.page_content.strip()}"
        for i, (chunk, _) in enumerate(chunks_with_scores)
    ])
    # 有稍微改一下prompt template, 因為deepseek有時會說無法在同一張圖片找到所有動物
    prompt_template = PromptTemplate.from_template(
        """你是一個AI的助理，請使用繁體中文回答使用者的問題。 請注意資料會來自不同的, 多筆的資料來源
        以下是檢索資料，如果沒有可以回答問題資料，請表示"檢索無相關資料，我不知道"：
        {context}

        問題：{question}

        請以繁體中文作答："""
    )
    # 塞入 prompt
    prompt = prompt_template.format(context=context, question=query)
    
    llm = Ollama(model=llm_model, base_url=ollama_url, temperature=temperature, top_p=top_p)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # 將變數帶入 run() 裡
    answer = chain.run({"context": context, "question": query})

    return {
        "answer": answer,
        "chunks": [
            {"content": chunk.page_content, "score": score}
            for chunk, score in similarity_score      # 這行有改
        ]
    }