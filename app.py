import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os

import sys
try:
    import imghdr
except ImportError:
    # Polyfill for Python 3.13+ where imghdr is removed
    from types import ModuleType
    imghdr = ModuleType("imghdr")
    imghdr.what = lambda f, h=None: None
    sys.modules["imghdr"] = imghdr

# Import the custom logic from your detection.py file
from detection import divide_shelf_into_zones, compute_zones_coverage

# --- 1. App Configuration ---
st.set_page_config(page_title="Retail OOS Detector", page_icon="🛒", layout="wide")
st.title("🛡️ Retail Out-of-Stock (OOS) Detector")
st.write("Real-time shelf monitoring using YOLOv11 and Spatial Occupancy Grids.")

# --- 2. Sidebar for Metadata ---
st.sidebar.header("Store Location Metadata")
aisle_input = st.sidebar.text_input("Current Aisle ID", value="Aisle_04")
camera_input = st.sidebar.text_input("Camera ID", value="Cam_North_01")
conf_threshold = st.sidebar.slider("Model Confidence", 0.1, 1.0, 0.3)

# --- 3. Resource Loading (Cached for Speed) ---
@st.cache_resource
def load_model():
    """Loads the YOLO model into memory once and caches it."""
    return YOLO(os.path.join(os.getcwd(),"best.pt"))

model = load_model()

# --- 4. Core Processing Logic ---
def process_shelf_image(img_file):
    if img_file is not None:
        # Convert Streamlit upload to OpenCV format
        image = Image.open(img_file)
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        H, W = img_cv.shape[:2]

        st.image(image, caption='Source Image', width="stretch")
        
        with st.spinner('Calculating shelf occupancy...'):
            # Run Inference
            results = model(img_cv, conf=conf_threshold)[0]
            
            # Guard against no detections
            if len(results.boxes) == 0:
                boxes = np.empty((0, 4))
            else:
                boxes = results.boxes.xyxy.cpu().numpy()

            # Run Geometric Spatial Logic (3 Rows, 10 Columns)
            zones = divide_shelf_into_zones(W, H, cols=10, rows=3)
            zones = compute_zones_coverage(zones, boxes, coverage_threshold=0.15)

            # Generate Results
            oos_alerts = []
            shelf_names = ["Top Shelf", "Middle Shelf", "Bottom Shelf"]
            
            for z in zones:
                if not z["stocked"]:
                    # Determine Position (Left/Center/Right)
                    col_pct = z["col"] / 9 
                    pos = "Left" if col_pct < 0.33 else "Center" if col_pct < 0.66 else "Right"
                    
                    # Determine Shelf Level
                    level = shelf_names[z["row"]] if z["row"] < 3 else f"Level {z['row']}"
                    
                    oos_alerts.append({
                        "shelf": level,
                        "position": pos,
                        "severity": "HIGH" if z["coverage"] < 0.05 else "MEDIUM"
                    })

            # --- 5. Display Results ---
            st.subheader(f"Analysis Results: {len(oos_alerts)} Gaps Detected")
            
            if len(oos_alerts) > 0:
                cols = st.columns(2)
                for idx, alert in enumerate(oos_alerts):
                    with cols[idx % 2]:
                        msg = f"{alert['shelf']} - {alert['position']} ({aisle_input})"
                        if alert['severity'] == "HIGH":
                            st.error(f"🚨 **HIGH PRIORITY**: {msg}")
                        else:
                            st.warning(f"⚠️ **MEDIUM PRIORITY**: {msg}")
            else:
                st.success(f"✅ All zones in {aisle_input} are fully stocked!")

# --- 6. User Interface Tabs ---
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Store Capture"])

with tab1:
    uploaded_file = st.file_uploader("Upload shelf photo", type=["jpg", "png", "jpeg"])
    if st.button('Analyze Uploaded Photo'):
        process_shelf_image(uploaded_file)

with tab2:
    st.info("Ensure the camera is leveled with the shelf for best results.")
    cam_file = st.camera_input("Capture live shelf image")
    if cam_file:
        process_shelf_image(cam_file)