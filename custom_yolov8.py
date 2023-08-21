import glob
import streamlit as st
import wget
from PIL import Image
import cv2
import os
import time
import numpy as np
import supervision as sv
# from onnxruntimer import prediction_onnx
from foziljon_trt import run_tensorrt
# from TensorRT_uchun import run_tensorrt
import time
from with_RT import run_tensorrt
 # very important
import pycuda.autoinit
st.set_page_config(layout="wide")
cfg_model_path = 'models/yolov8n.engine'
model = None
confidence = .25

box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

def image_input(data_src):
    img_file = None
    if data_src == 'Bor manbadan foydalanish':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Test uchun rasm tanlang", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Videoni yuklash", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Tanlangan rasm")
        with col2:
            open_cv_image = np.array(img_file) 
            img = infer_image(model_path=cfg_model_path, img=open_cv_image, TensorRT=True)
            img = img[0]
            img = Image.fromarray(img.plot()[:,:,::-1])
            st.image(img, caption="Model bashorati")


def video_input(data_src):
    vid_file = None
    if data_src == 'Bor manbadan foydalanish':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Videoni yuklash", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Balandlik")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Kenglik")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Tasvirni o'qib bo'lmadi. Chiqilmoqda ....")
                break
            frame = cv2.resize(frame, (width, height))
            
            output_img = infer_image(model_path=cfg_model_path, img=frame, TensorRT=True)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            # plt.plot(int(fps))

        cap.release()


def infer_image(img, model_path = cfg_model_path, size=None, onnx = False, TensorRT = False):
    return run_tensorrt(enggine_path= model_path, image=img)
    

@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model", ["Fayl yuklash", "Internet manzili"], key = int(time.time()))
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Modelni yuklash", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("Model URL addresi")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # global variables
    global model, confidence, cfg_model_path
    st.empty()
    st.title("Tasvirdan obyektni topib olish")

    # Model file mavjudligini tekshirish
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file mavjud emas! Iltimos model qo'shing.", icon="⚠️")
    else:
        print("Model checked-->>-->>")
        st.sidebar.markdown("---")
        data_src = st.sidebar.radio("Manbani tanlang: ", ['Bor manbadan foydalanish', 'File ko\'rsatish'], key = int(time.time()))
        video_input(data_src)


if __name__ == "__main__":
    main()
