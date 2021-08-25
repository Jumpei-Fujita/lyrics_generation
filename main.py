import streamlit as st
import torch
from model import Lyrics_transformer



def get_x(shitei_array):
    sep = ['<', '>', '{']
    new_shitei_array = shitei_array[-3:]
    x = ''
    for i in range(len(new_shitei_array)):
        x = x + new_shitei_array[i] + sep[i]
    return x, len(new_shitei_array)

st.title('自動歌詞生成')
status_area = st.empty()
status_area.write('準備中')

st.sidebar.write('sampling parameter(サンプリング手法)')
tmp = st.sidebar.slider("tmperature", min_value=0.5, max_value=1.0, step=0.01, value=1.0)
top_k = st.sidebar.slider("top-k", min_value=2, max_value=100, step=1, value=40)

d_model = 336
nhead = 4
num_layers = 3
dim_feedforward = 672
model = Lyrics_transformer(d_model, nhead, num_layers, dim_feedforward)

model.load_state_dict(torch.load('lyrics_transformer2_cpu.pth'))
status_area.write('準備完了')



text1 = st.text_input('1行目 例(くだらねえ)')
text2 = st.text_input('2行目 例(いつになりゃ終わる)')
text3 = st.text_input('3行目 例(なんか死にて気持ちで)')
texts = [text1, text2, text3]
shitei_array = []
for t in texts:
    if t.replace(' ','').replace('　','') != '':
        shitei_array.append(t)
shitei_num = len(shitei_array)
shitei = st.radio("歌詞を指定しますか?",('する', 'しない'))

if shitei_num == 0:
    shitei = 'しない'
gyou = st.number_input('何行生成しますか?',min_value=1, max_value=10, value=4)
seisei_button = st.button('歌詞を生成')

kashi_c = ['歌詞1','歌詞2','歌詞3']
if seisei_button:
    if shitei == 'する':
        st.header('指定しました')
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for co in range(len(cols)):
            col = cols[co]
            col.write('*'+kashi_c[co]+'*')
            shitei_array = []

            for t in texts:
                if t.replace(' ','').replace('　','') != '':
                    shitei_array.append(t)
            
            for i in range(gyou):
                x, g_num = get_x(shitei_array)
                gene = model.generate(False, x, top_k=top_k, tmp=tmp)[0].replace("「","").replace("」","").replace("(","").replace(")","")
                sep = ['<', '>', '{']
                for s in sep:
                    gene = gene.replace(s ,'||')
                gene = gene.split('||')

                gene_app = gene[-4+g_num:]
                for g in gene_app:
                    shitei_array.append(g)
                if len(shitei_array) > gyou - 1 + shitei_num:
                    break
            
            for ly in shitei_array:
                col.write(ly)
    else:
        st.header('指定してしません')
        shitei_num = 0
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for co in range(len(cols)):
            col = cols[co]
            col.write('*'+kashi_c[co]+'*')
            shitei_array = []
            
            for i in range(gyou):
                x, g_num = get_x(shitei_array)
                gene = model.generate(False, x, top_k=top_k, tmp=tmp)[0].replace("「","").replace("」","").replace("(","").replace(")","")
                sep = ['<', '>', '{']
                for s in sep:
                    gene = gene.replace(s ,'||')
                gene = gene.split('||')
        
                gene_app = gene[-4+g_num:]
                for g in gene_app:
                    shitei_array.append(g)
                if len(shitei_array) > gyou - 1 + shitei_num:
                    break
            
            for ly in shitei_array:
                col.write(ly)




