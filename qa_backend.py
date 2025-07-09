
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
import io   # 有新增
from PyPDF2 import PdfReader    # 有新增
from PIL import Image, ImageFile
import pytesseract  # 不能用pip3 要用sudo apt install tesseract-ocr tesseract-ocr-chi-tra

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

# 改動部分
def load_documents_from_upload(uploaded_file):
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()
    if filename.endswith(".pdf"):
        # PDF 直接用 bytes
        reader = PdfReader(io.BytesIO(content))
        docs = []
        for i, p in enumerate(reader.pages, start=1):
            txt = p.extract_text() or ""
            docs.append(
                Document(page_content=txt, metadata={"source": uploaded_file.name, "page": i})
            )
        return docs
    
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允許載入截斷的影像檔
        image = Image.open(io.BytesIO(content))     # 用 PIL 讀圖
        text = pytesseract.image_to_string(image, lang="chi_tra+eng")  # 使用 OCR 進行文字抽取, lang="chi_tra+eng": 中英混合
        # 拆成段落或行
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [
            Document(page_content=line, metadata={"source": uploaded_file.name})
            for line in lines
        ]

    else:
        # 文字檔才 decode
        text = content.decode("utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [
            Document(page_content=line, metadata={"source": uploaded_file.name})
            for line in lines
        ]
    # return [
    #     Document(page_content=rule.strip(), metadata={"source": uploaded_file.name})
    #     for rule in content.split("\n") if rule.strip()
    # ]

# ✅ 建立 FAISS 向量資料庫
def build_FAISS(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

# ✅ 執行問答流程
def run_qa(uploaded_file, query: str, ollama_url: str,
        embedding_model: str = "bge-m3",
        llm_model: str = "deepseek-r1:8b") -> str:
    if isinstance(uploaded_file, list):     # 這段有改
        docs = uploaded_file
    else:
        docs = load_documents_from_upload(uploaded_file)
        
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
