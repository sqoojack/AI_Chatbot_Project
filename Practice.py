import time

import streamlit as st

import numpy as np
import pandas as pd

st.title('My first streamlit pratice')

# st.write("Create **table**!")

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })
# df

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),     # create a array of shape (20, 3)
#     columns=['a', 'b', 'c']
# )
# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(100, 2) / [50, 50] + [24.778811, 121.019160],
#     columns=['lat', 'lon']
# )
# st.map(map_data)

# if st.button('Put the botton, it will show snow!'):
# 	st.snow()

# option = st.sidebar.selectbox(
#     'Which animal do you like?',
#     ['Dog', 'Cat', 'Bird', 'Horse']
# )

# st.text(f"You like {option}")

# tab1, tab2 = st.tabs(["Cat 介紹", "Dog 介紹"])

# with tab1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

# with tab2:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
   
   
   
# message = st.chat_message("AI assistant")  # 或者寫 "ai"
message = st.chat_message("assistant", avatar="🦖")  # 自訂頭像
message.write("你好！我是 ChatBot 🤖，可以回答各種問題，提供資訊。")
message.write("有什麼我可以幫助你的嗎？")

st.chat_input("Say something...")
