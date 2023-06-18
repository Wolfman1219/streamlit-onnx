import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import supervision as sv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from onnxruntimer import prediction_onnx
from TensorRT_uchun import run_tensorrt
import pycuda.driver as cuda


st.set_page_config(layout="wide")
# chart_placeholder = st.empty()
# plt = st.pyplot()

cfg_model_path = 'models/yolov8n.pt'
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
            # open_cv_image = np.array(img_file) 
            img = model.predict(img_file)
            img = img[0]
            img = Image.fromarray(img.plot()[:,:,::-1])
            st.image(img, caption="Model bashorati")


def video_input(data_src):
    vid_file = None
    if data_src == 'Bor manbadan foydalanish':
        vid_file = "data/sample_videos/sample2.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Videoni yuklash", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Frame o'lchamini belgilash")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Kenglik", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Balandlik", min_value=120, step=20, value=height)

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
            if cfg_model_path.endswith(".onnx"):
                output_img = infer_image(model_path=cfg_model_path, img=frame, onnx=True)
            elif cfg_model_path.endswith(".engine"):
                output_img = infer_image(model_path=cfg_model_path.replace(".engine", ".onnx"), img=frame, onnx=True)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_img = infer_image(frame)
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
    if TensorRT:
        return run_tensorrt(enggine_path= model_path, image=img)
    
    elif onnx:
        return prediction_onnx(model_path=model_path, image=img)
    
    model.conf = confidence
    result = model(img, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [f"Car {confidence:0.2f}" for _, mask, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=img, detections=detections, labels=labels) 
    print("Box annotator\n\n\n")
    return frame
    




# @st.experimental_singleton
@st.cache_resource
def load_model(path, device):
    model_ = YOLO(path)
    model_.to(device)
    cuda.Context.pop()
    print("qurilma: ", device)
    return model_


# @st.experimental_singleton
@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model", ["Fayl yuklash", "Internet manzili"])
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

    st.title("Tasvirdan obyektni topib olish")

    st.sidebar.title("Sozlamalar")

    # upload model
    model_src = st.sidebar.radio("Modelni tanla", ["YOLO model", "ONNX runtime", "TensorRT"])
    # URL, upload file (max 200 mb)
    if model_src == "ONNX runtime":
        user_model_path = "models/yolov8n.onnx"
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")
    elif model_src == "TensorRT":
        user_model_path = "models/yolov8n.engine"
        if user_model_path:
            cfg_model_path = user_model_path


    # Model file mavjudligini tekshirish
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file mavjud emas! Iltimos model qo'shing.", icon="⚠️")
    else:
        #device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Qurilmani tanla", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Qurilmani tanla", ['cpu', 'cuda'], disabled=True, index=0)
        

        # load model
        if cfg_model_path.endswith(".pt"):
            model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Ishonchlilik', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        # if st.sidebar.checkbox("Classlar"):
        #     model_names = list(model.names.values())
        #     assigned_class = st.sidebar.multiselect("Classni tanla", model_names, default=[model_names[0]])
        #     classes = [model_names.index(name) for name in assigned_class]
        #     model.classes = classes
        # else:
        #     model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Tanlang: ", ['Rasm', 'Video'])

        # input src option
        data_src = st.sidebar.radio("Manbani tanlang: ", ['Bor manbadan foydalanish', 'File ko\'rsatish'])

        if input_option == 'Rasm':
            image_input(data_src)
        else:
            video_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
