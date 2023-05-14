import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

labels = ["Classic/Jazz", "Electronic/Experimental", "Metal", "R&B/Hip-Hop", "Rock/Pop"]
n_class = len(labels)
img_size = 224
n_result = 3  # 上位3つの結果を表示

model = {'Xception1': './my_model.h5',
         'Xception2': './my_model2.h5',
         'Dense Net': './my_model_DN.h5',
         'Efficient Net': './my_model_EN.h5',
         'Mobile Net': './my_model_MN.h5'
         }

st.title('AI Music Album Jacket Classifier')
st.write('アルバムのジャケットから音楽のジャンルを推定します.')
st.write('あなたのアルバムがどのジャンルに見えるのか推定します.')
st.write(f'ジャンルは{labels[0]}、{labels[1]}、{labels[2]}、{labels[3]}、{labels[4]}です.')
st.write('')
model_selection = st.selectbox(label='モデルを選択してください.', options=model.keys())


@st.cache_resource
def load_model(selection):
    return tf.keras.models.load_model(model[selection])


new_model = load_model(model_selection)
uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image_show = np.array(image)

    st.image(image_show, caption='uploaded image', use_column_width=True)

    image = image.resize((img_size, img_size))
    image = np.array(image)
    image = image.astype('float') / 255.0
    image = tf.reshape(image, [1, 224, 224, 3])

    pred = new_model.predict(image)
    sorted_idx = np.argsort(-pred[0])  # 降順でソート

    st.header('Result')

    for i in range(n_result):
        idx = sorted_idx[i]
        ratio = pred[0][idx]
        label = labels[idx]
        st.write(f'{round(ratio * 100, 1)}%の割合で{label}の要素が含まれています.')

    chart_data = pd.DataFrame(
        pred[0] * 100,
        index=labels,
        columns=['probability(%)'])
    st.bar_chart(chart_data)
