import cv2
import numpy as np
import torch
import streamlit as st
import tempfile
from PIL import Image

class Deployment:
    def __init__(self):
        self.model = "models/yolov8n.engine"
        self.colors = [(255,0,0),(0,0,255),(0,0,0)]
        
    def ImageBox(self, image):
        
        new_shape=(640, 640)
        width, height, channel = image.shape
        
        ratio = min(new_shape[0] / width, new_shape[1] / height)
        new_unpad = int(round(height * ratio)), int(round(width * ratio))
        dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

        if (height, width) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.colors[-1])
        
        return image, ratio, (dw, dh)

    def face_detection_for_image_with_engine(self, image):
      
        model = TRTEngine(self.model)
        
        img, ratio, dwdh = self.ImageBox(image)
        tensor = blob(img, return_seg=False)
        tensor = torch.asarray(tensor)

        dwdh = np.array(dwdh * 2, dtype=np.float32)

        results = model(tensor)

        bboxes, scores, labels = det_postprocess(results)
        bboxes = (bboxes-dwdh)/ratio
        
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cv2.rectangle(image, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , self.colors[1], 2)
    
        return image

def run_tensorrt(image, enggine_path):

    deployment = Deployment()    

    predicted_image = deployment.face_detection_for_image_with_engine(image)
    return predicted_image



def main():
    DEFAULT_VIDEO_PATH = "/home/hasan/Public/yolo_with_streamlit/data/sample_videos/sample.mp4"

# Create a video file uploader
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
        # Load the video with cv2
        cap = cv2.VideoCapture(video_path)
        outputing = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the inference
            output = run_tensorrt(image=frame, enggine_path='models/yolov8n.engine')

            # Convert the output to an image that can be displayed
            output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

            # Display the image
            outputing.image(output_image)

        cap.release()
    else:
        st.write("Please upload a video file or choose to use the default video.")

if __name__ == "__main__":
    main()
