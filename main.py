#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="Artificial Intelligence 2 - Machine Project",
    page_icon="assets/icon.png",  # change if you have a custom icon
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# Initialize page_selection
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # default page

def set_page_selection(page):
    st.session_state.page_selection = page

#######################
# Sidebar
with st.sidebar:
    st.title('Artificial Intelligence 2\nMachine Project')

    st.subheader("Pages")
    if st.button("Project Description", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
    if st.button("Model Predictions", use_container_width=True, on_click=set_page_selection, args=('predictions',)):
        st.session_state.page_selection = 'predictions'
 

    st.subheader("Members")
    st.markdown("1. Tan, Gabriel Christian D.\n2. Novesteras, Aaron Gabriel L.\n3. Zerda, Thomas Kaden K.\n4. Brown, Ian Miguel A.")

    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training a Brain Tumor Predictor using the Brain Tumor dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/data)")
    st.markdown("📔 [Google Colab Notebook](https://colab.research.google.com/drive/1-lZr_QGnVoqgKUNHxGWbknkzz7m-hjWp?usp=sharing)")
    st.markdown("🗄️ [GitHub Repository](https://github.com/OrnnlyFans/AI2-Machine-Project.git)")
#######################
# Pages
# About Page
if st.session_state.page_selection == "about":
    st.header("📘 Project Description")

    st.markdown("""
### **Abstract**
This project implements a **custom YOLOv12 model** to automatically detect and classify **brain tumors** from MRI scans.  
The model was trained on a curated dataset consisting of three primary tumor types: **Glioma**, **Meningioma**, and **Pituitary**.  
Through optimized preprocessing, data augmentation, and GPU-accelerated training on Google Colab, the model achieved **strong detection accuracy** and **balanced precision–recall performance**.  
Results show that YOLOv12 effectively identifies complex medical features with **robust spatial consistency**, demonstrating its suitability for real-world medical image analysis.

**Keywords:** Brain tumor detection, YOLOv12, deep learning, MRI classification, object detection, image segmentation, medical imaging, computer vision.  

---

### **I. Introduction**
Brain tumor detection is a critical challenge in medical imaging. Manual MRI interpretation is often subjective, time-consuming, and prone to human error.  
With the rise of deep learning, **object detection models such as YOLO (You Only Look Once)** provide automated and consistent analysis of complex medical images.  

In this study, a **custom-trained YOLOv12** model was developed to **classify and localize tumors** across three categories: Glioma, Meningioma, and Pituitary.  
The model’s transformer-based backbone enhances feature extraction and spatial awareness, leading to improved accuracy in both **bounding box detection** and **segmentation** tasks.  

The implementation of such automated systems aims to assist radiologists in diagnostic workflows, enabling faster and more reliable tumor identification compared to traditional methods.

---

### **Objectives**
The objectives of this project are to:  
1. **Develop** a YOLOv12-based model for brain tumor detection and classification from MRI scans.  
2. **Apply** preprocessing and augmentation strategies to enhance model generalization.  
3. **Evaluate** performance using detection and segmentation metrics (Precision, Recall, mAP50, mAP50–95).  
4. **Analyze** results to assess YOLOv12’s effectiveness in real-world medical imaging applications.
""")


elif st.session_state.page_selection == "dataset":
    import os
    import random
    from PIL import Image

    st.header("📊 Dataset Overview")
    st.write("""
    The dataset contains MRI images labeled under categories such as **Meningioma**, **Glioma**, and **Pituitary**.  
    It is used to train YOLOv12 for segmentation and detection tasks.  
    Below are sample images for each category, which you can also download for testing the model.
    """)

    # Path to sample image folder
    base_path = "sample_img"
    categories = ["Meningioma", "Glioma", "Pituitary"]

    # Shuffle button
    if st.button("🔄 Shuffle Images"):
        if "random_imgs" in st.session_state:
            del st.session_state["random_imgs"]
        st.rerun()

    # Initialize session state to store random selections
    if "random_imgs" not in st.session_state:
        st.session_state.random_imgs = {}

    # Load and display random images per category
    for cat in categories:
        st.subheader(f"🧩 {cat}")

        img_dir = os.path.join(base_path, cat)
        if os.path.exists(img_dir):
            files = os.listdir(img_dir)
            if len(files) == 0:
                st.warning(f"No images found in {img_dir}")
                continue

            # Pick 3 random images once per session
            if cat not in st.session_state.random_imgs:
                st.session_state.random_imgs[cat] = random.sample(files, min(3, len(files)))

            imgs = st.session_state.random_imgs[cat]
            cols = st.columns(len(imgs))

            for i, col in enumerate(cols):
                with col:
                    img_path = os.path.join(img_dir, imgs[i])
                    st.image(Image.open(img_path), use_container_width=True)
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f,
                            file_name=f"{cat}_{imgs[i]}",
                            mime="image/jpeg"
                        )
        else:
            st.warning(f"⚠️ Directory not found for {cat}")
# Model Predictions Page
elif st.session_state.page_selection == "predictions":
    st.header("🧠 Model Prediction (YOLO)")

    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image
    from matplotlib import pyplot as plt

    # Load YOLO model once
    @st.cache_resource
    def load_model():
        return YOLO("model/best.pt")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image as OpenCV BGR
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)[:, :, ::-1].copy()  # RGB → BGR

        # Run inference
        results = model(img)
        r = results[0]
        names = model.names

        output_img = img.copy()
        detections_text = []

        # Overlay segmentation masks
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (output_img.shape[1], output_img.shape[0]))
                colored_mask = np.zeros_like(output_img)
                colored_mask[:, :, 2] = mask  # Red overlay
                output_img = cv2.addWeighted(output_img, 1.0, colored_mask, 0.5, 0)

        # Draw bounding boxes and labels
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[cls_id]} {score:.2f}"
                detections_text.append(label)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display using matplotlib (like your reference)
        h, w = output_img.shape[:2]
        dpi = 300
        fig_w, fig_h = w / dpi, h / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        st.pyplot(fig)

        # Show detected tumor types
        if detections_text:
            st.subheader("🧩 Detected Tumor Types")
            for det in detections_text:
                st.write(f"- {det}")
        else:
            st.warning("No tumor detected.")
