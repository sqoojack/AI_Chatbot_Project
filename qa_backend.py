
# qa_backend.py
import requests
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

# ✅ 載入上傳的檔案為 Document
def load_farm_rules_from_upload(uploaded_file) -> List[Document]:
    content = uploaded_file.read().decode("utf-8")
    return [
        Document(page_content=rule.strip(), metadata={"source": uploaded_file.name})
        for rule in content.split("\n") if rule.strip()
    ]

# ✅ 建立 FAISS 向量資料庫
def build_FAISS(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

# ✅ 執行問答流程
def run_qa(uploaded_file, query: str, ollama_url: str,
        embedding_model: str = "bge-m3",
        llm_model: str = "deepseek-r1:8b") -> str:
    docs = load_farm_rules_from_upload(uploaded_file)
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    vectorstore = build_FAISS(docs, embeddings)

    llm = Ollama(model=llm_model, base_url=ollama_url)
    prompt_template = PromptTemplate.from_template(
        """你是一個問答機器人，並且只能依據知識庫去回答問題，不能依靠你本身的知識，若是找不到就說沒有
        可能有假資料，沒關係就照著資料庫回答，並要用繁體中文回答。
        以下是知識庫內容：
        {context}

        問題：{question}

        請以繁體中文作答："""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa.run(query)
