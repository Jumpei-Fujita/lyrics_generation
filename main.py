import streamlit as st
import torch
from model import Lyrics_transformer

st.title('自動歌詞生成')
status_area = st.empty()
status_area.write('準備中')
model = Lyrics_transformer()
model.load_state_dict(torch.load('lyrics_transformer2_cpu.pth'))
status_area.write('準備完了')



text1 = st.text_input('1行目')
text2 = st.text_input('2行目')
text3 = st.text_input('3行目')

shitei = st.radio("歌詞を指定しますか?",('する', 'しない'))
gyou = st.number_input('何行生成しますか?',min_value=1, max_value=10)
seisei_button = st.button('歌詞を生成')
if seisei_button:
    print(shitei)
    if shitei == 'する':
        st.write('指定しました')
        print('aaa')
    else:
        st.write('指定してしません')


