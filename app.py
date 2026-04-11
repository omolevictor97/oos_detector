import streamlit as st
import requests
from PIL import Image
import io

import subprocess
import time

if "api_process" not in st.session_state:
    st.session_state.api_process = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])
    time.sleep(5) # Give the API time to wake up


# App Configuration
st.set_page_config(page_title="Retail OOS Detector", page_icon="🛒")
st.title("🛡️ Retail Out-of-Stock (OOS) Detector")
st.write("Detect gaps and misplaced products in real-time.")

# Sidebar for Aisle Configuration
st.sidebar.header("Store Location Metadata")
aisle_input = st.sidebar.text_input("Current Aisle ID", value="Aisle_04")
camera_input = st.sidebar.text_input("Camera ID", value="Cam_North_01")

# Create two tabs for the two different input methods
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Store Capture"])

def process_image(img_file):
    """Helper to send image to FastAPI and display results"""
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption='Source Image', width="stretch")
        
        with st.spinner('Analyzing shelf occupancy...'):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            files = {'file': img_byte_arr.getvalue()}
            
            # Request to your local FastAPI server
            try:
                params = {'aisle_id': aisle_input}
                response = requests.post("http://127.0.0.1:8000/detect_oos", params=params, files=files)
                data = response.json()
                
                # Results Display
                st.subheader(f"Analysis Results: {data['total_gaps']} Gaps Detected")
                st.write(data["alerts"])
                
                if data['total_gaps'] > 0:
                    for alert in data.get('alerts', []):
                        shelf = alert.get('shelf', 'Unknown Shelf')
                        pos = alert.get('position', 'Unknown Position')
                        sev = alert.get('severity', 'LOW')
                        
                        # Ensure aisle_input exists (it must be defined earlier in your script)
                        aisle = aisle_input if 'aisle_input' in locals() else "Aisle 1"

                        if sev == "HIGH":
                            st.error(f"🚨 **HIGH PRIORITY**: {shelf} - {pos} ({aisle})")
                        else:
                            st.warning(f"⚠️ **MEDIUM PRIORITY**: {shelf} - {pos} ({aisle})")
                else:
                    st.success("✅ Shelf is fully stocked!")
            except Exception as e:
                st.error(f"{str(e)}")
                print(f"Error during API call: {str(e)}")

# --- Tab 1: Manual Upload ---
with tab1:
    uploaded_file = st.file_uploader("Upload shelf photo", type=["jpg", "png", "jpeg"])
    if st.button('Analyze Uploaded Photo'):
        process_image(uploaded_file)

# --- Tab 2: Camera Capture ---
with tab2:
    st.info("Ensure the camera is leveled with the shelf for best results.")
    cam_file = st.camera_input("Capture live shelf image")
    
    # In Streamlit, camera_input triggers immediately upon capture
    if cam_file:
        process_image(cam_file)