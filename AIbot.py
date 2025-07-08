
# Command line: streamlit run AIbot.py

import streamlit as st
import subprocess   # used to activate external system command in python


class OllamaDisplay:
    def __init__(self, model_name, system_prompt):
        self.model_name = model_name
        self.system_prompt = system_prompt
        
    def ensure_model_download(self):
        try:
            cur_model_list = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            st.error(f"無法列出本機模型：\n{e.stderr or e.stdout}")
            return False
        
        if self.model_name not in cur_model_list.stdout:
            with st.spinner(f"**下載 {self.model_name} 中...**"):
                try:
                    pull = subprocess.run(
                            ["ollama", "pull", self.model_name],
                            capture_output=True, text=True, check=True
                        )
                    st.success(f"{self.model_name} 下載完成！")
                except subprocess.CalledProcessError as e:
                    st.error(f"下載模型失敗：\n{e.stderr or e.stdout}")
                    return False
        return True
        
    def generate_answer(self, user_prompt):
        result = subprocess.run([
                "ollama", "run", self.model_name,
                "你是一個根據公司內部資料庫問答的機器人",   # system prompt
                user_prompt
            ], capture_output = True, text = True, check = True
        )
        answer = result.stdout.strip()
        return answer
    
    def display(self):
        st.title("Ollama model Interface testing")
        st.subheader("Please input your question:")
        if not self.ensure_model_download():
            return
        
        user_prompt = st.text_area("", height=150)  # provide text area for user's input
        
        if st.button('送出') and user_prompt.strip():
            with st.spinner("**模型思考中...**"):
                try:
                    answer = self.generate_answer(user_prompt)
                    st.write(answer)
                except subprocess.CalledProcessError as e:
                    err = e.stderr.strip() or e.stdout.strip()
                    st.error(f"呼叫 Ollama失敗: \n{err}")
                    
# Stream = Streamlit(model_name="phi3", system_prompt="你是一個根據公司內部資料庫問答的機器人")
Stream = OllamaDisplay(model_name="mistral", system_prompt="你是一個根據公司內部資料庫問答的機器人")
Stream.display()