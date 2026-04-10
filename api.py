from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import numpy as np
import io
import os

from detection import detect_oos, divide_shelf_into_zones, compute_zones_coverage

best_model_path = os.path.join(os.getcwd(), "best.pt")

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)
model = YOLO(best_model_path)

@app.post("/detect_oos/")
async def detect_oos_api(aisle_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    results = model(img, conf=0.3)[0]

    if len(results.boxes) == 0:
        # If no products are found, the whole shelf is technically "empty"
        # or the photo was too blurry/dark to see anything.
        boxes = np.empty((0, 4)) 
    else:
        boxes = results.boxes.xyxy.cpu().numpy()

    zones = divide_shelf_into_zones(W, H, cols=10, rows=3)
    zones = compute_zones_coverage(zones, boxes, coverage_threshold=0.15)

    oos_alerts = []
    shelf_names = ["Top Shelf", "Middle Shelf", "Bottom Shelf"]
    for z in zones:
        if not z["stocked"]:
            col_pct = z["col"] / (10 - 1)
            pos = "Left" if col_pct < 0.33 else "Center" if col_pct < 0.66 else "Right"
            oos_alerts.append({
                "aisle" : aisle_id,
                "shelf" : shelf_names[z["row"]] if z["row"] < len(shelf_names) else f"Shelf {z['row']}",
                "position" : pos,
                "severity" : "High" if z["coverage"] < 0.05 else "Medium"
            })
    return {"alerts": oos_alerts, "total_gaps": len(oos_alerts), "total_zones": len(zones)}