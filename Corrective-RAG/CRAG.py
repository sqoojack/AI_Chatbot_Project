import requests
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
import numpy as np
from scipy.stats import norm
import io
from PyPDF2 import PdfReader
from PIL import Image, ImageFile

# 自訂 Ollama Embeddings 類別
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
        return response.json()["embedding"]

# 建立 FAISS 向量資料庫
def build_FAISS(docs: List[Document], embeddings: Embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

# Corrective RAG 主流程
def run_corrective_rag(
    docs: List[Document],
    query: str,
    ollama_url: str,
    llm_model: str = "deepseek-r1:8b",
    embedding_model: str = "bge-m3",
    temperature: float = 0.7,
    top_p: float = 0.9,
    k: int = 5
) -> dict:
    # 建立向量資料庫
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    vectorstore = build_FAISS(docs, embeddings)

    # 第一次檢索 + 初稿生成
    chunks = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([c.page_content for c in chunks])
    llm = Ollama(model=llm_model, base_url=ollama_url, temperature=temperature, top_p=top_p)

    prompt_draft = PromptTemplate.from_template(
        """你是一個 AI 助理，請使用繁體中文回答。
【檢索結果】：
{context}

請根據上面資料，回答問題：{question}
如果沒資料，就說「檢索無相關資料，我不知道」。"""
    )
    draft_answer = llm.generate(
        prompt=prompt_draft.format(context=context, question=query)
    ).get("response", "").strip()

    # 事實校正流程
    prompt_correct = PromptTemplate.from_template(
        """你是一個 AI 校正助理，請使用繁體中文執行下列任務：

1. 檢閱「初稿回答」，找出和檢索結果不符或有疑慮的敘述。
2. 依據【檢索結果】中的內容，修正那些錯誤或不完整之處。
3. 輸出「最終回答」。

【檢索結果】：
{context}

【初稿回答】：
{draft}

請開始校正："""
)
    
final_answer = llm.generate(prompt=prompt_correct.format(context=context, draft=draft_answer)).get("response", "").strip()

    # 返回結果
    return {
        "draft": draft_answer,
        "answer": final_answer,
        "chunks": [{"content": c.page_content} for c in chunks]
    }
