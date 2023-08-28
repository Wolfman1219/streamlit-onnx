import cv2
import numpy as np
import torch
import streamlit as st
import streamlit_webrtc as webrtc
import tempfile
from PIL import Image
import onnxruntime
from onnxruntimer import prediction_onnx
import time
opt_session = onnxruntime.SessionOptions()
opt_session.enable_mem_pattern = False
opt_session.enable_cpu_mem_arena = False
opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
# model_path = 'models/best.onnx'
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession('models/yolov8n.onnx', providers=EP_list)

def infer_image(img, model = ort_session):
    return prediction_onnx(ort_session=model, image=img)


def main():
    DEFAULT_VIDEO_PATH = "data/sample_videos/sample.mp4"
# Create a video file uploader
    webrtc_ctx = webrtc.StreamlitWebRTC()
    st.header("Upload a video for inference")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    # Create a radio button for selecting between default video and uploaded video
    video_selection = st.radio(
        "Select video for inference:",
        ("Use default video", "Use uploaded video")
    )

    # If the user chooses to use the default video
    if video_selection == "Use default video":
        video_path = DEFAULT_VIDEO_PATH

    # If the user chooses to use the uploaded video
    elif video_selection == "Use uploaded video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # If there's a video to process, do the inference
    if video_path is not None:
        if webrtc_ctx.state.playing:
            frame = webrtc_ctx.video_frame.copy()
            output = infer_image(img=frame)
        # Load the video with cv2
            cap = cv2.VideoCapture(video_path)
            outputing = st.empty()
            fps = 0
            prev_time = 0
            curr_time = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run the inference
                output = infer_image(img=frame, model=ort_session)

                # Convert the output to an image that can be displayed
                output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

                # Display the image
                webrtc_ctx.video_frame = np.array(output_image)
                # outputing.image(output_image)
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                # print(fps)
            cap.release()
        else:
            st.write("Please upload a video file or choose to use the default video.")

if __name__ == "__main__":
    main()
