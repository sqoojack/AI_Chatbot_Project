
# load_data.py
from langchain.schema import Document
import io   
from PyPDF2 import PdfReader    
from PIL import Image, ImageFile
import pytesseract  
import base64
import requests
import re
import json

def get_image_caption_llava(image_bytes, ollama_url, img_model):
    b64_image = base64.b64encode(image_bytes).decode()

    payload = {
        "model": img_model,
        "prompt": "請描述這張圖片的內容，讓視障者可以理解。",
        "images": [b64_image]
    }

    response = requests.post(f"{ollama_url}/api/generate", json=payload)
    
    result = ""
    for line in response.iter_lines():
        if line:
            result += line.decode("utf-8")
    responses = []

    # 使用正規表示法擷取所有 JSON 區塊
    json_objects = re.findall(r'\{.*?"response":.*?\}', result)

    for obj in json_objects:
        try:
            data = json.loads(obj)
            if "response" in data:
                responses.append(data["response"])
        except json.JSONDecodeError:
            continue  # 忽略壞掉的 JSON 片段

    # 合併所有回傳內容為完整敘述
    return "".join(responses).strip()

# ✅ 載入上傳的檔案為 Document
def load_documents_from_upload(uploaded_file, ollama_url, img_model):
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()
    if filename.endswith(".pdf"):
        # PDF 直接用 bytes
        reader = PdfReader(io.BytesIO(content))     # 這邊也有改, 解決chunk_size只有10幾個的問題
        full_text = ""
        for p in reader.pages:
            full_text += p.extract_text() or ""

        return [
            Document(page_content=full_text.strip(), metadata={"source": uploaded_file.name})
        ]
    
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允許載入截斷的影像檔
        image = Image.open(io.BytesIO(content)).convert("RGB")     # 用 PIL 讀圖
        ocr_text = pytesseract.image_to_string(image, lang="chi_tra+eng")  
        ocr_lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        ocr_paragraph = "\n".join(ocr_lines)
        # 拆成段落或行
        caption = get_image_caption_llava(content, ollama_url, img_model) or "(LLaVA 無回應)"

        full_text = f"""【圖像描述（LLaVA）】\n{caption}\n\n【圖片內文字（OCR）】\n{ocr_paragraph}"""
        #print("圖片描述:", full_text)
        return [
            Document(page_content=full_text.strip(), metadata={"source": uploaded_file.name, "type": "image+ocr+llava"})
        ]

    else:
        # 文字檔才 decode
        text = content.decode("utf-8")
        full_text = "\n".join([l.strip() for l in text.splitlines() if l.strip()])      # 這邊改成full_text, 總共一個document
        return [
            Document(page_content=full_text, metadata={"source": uploaded_file.name})
        ]