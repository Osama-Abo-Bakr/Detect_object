import numpy as np
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd

model = YOLO('yolov8s.pt')
class_names = model.names

st.set_page_config(page_title="Detect Object", page_icon="🤖", layout="centered")
st.title('Detect Object In Image')

object_select = st.selectbox('Select Object', options=list(class_names.values()))
upload_image = st.file_uploader('Upload The Image', type=['jpg', 'png', 'jpeg'])

if upload_image is not None:
    image = Image.open(upload_image)
    image = np.array(image)
    prediction = model.predict(image)[0].boxes.data
    final_pred = pd.DataFrame(prediction).astype(float)

    for idx, row in final_pred.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        cls = class_names[int(row[5])]
        confidence = row[4]

        if object_select == cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 165), 1)
            cv2.rectangle(image, (x1, y1), (x2, y1 + 30), (0, 255, 165), -1)
            cv2.putText(image, f'{cls}', (x1, y1 + 15), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
            cv2.putText(image, f'{round(confidence * 100, 2)}%', (x1, y1 + 30), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)



    if st.button('Show Prediction', key='show_prediction'):
        st.image(image, caption='output_image', use_column_width=True)