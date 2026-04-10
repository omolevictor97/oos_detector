import imghdr
from turtle import position
import cv2
import numpy as np
from sympy.polys.polytools import _sorted_factors
from ultralytics import YOLO
from typing import List, Tuple
from dataclasses import dataclass
import os

aisle_id = "Aisle_1"  # Example aisle ID; in practice, this would be dynamic
camera_id = "Cam_1"  # Example camera ID; in practice, this would be dynamic
gap_threshold = 0.1  # Confidence threshold for detecting gaps
row_overlap_IOU = 0.5  # IOU threshold for merging rows
best_path = os.path.join(os.getcwd(), "best.pt")

model = YOLO(best_path)

def divide_shelf_into_zones(img_width : int,img_height : int,cols : int = 10,rows : int = 3):
    """
    Divide the shelf into image a grid of zones.
    Each zone is independently checked for product presence, allowing for more accurate detection of gaps.
    This removes dependency on the model knowing what backgroubnd is 
    """ 
    zone_w = int(img_width / cols)
    zone_h = int(img_height / rows)

    zones = []
    for row in range(rows):
        for col in range(cols):
            zones.append(
                {
                    "row" : row,
                    "col" : col,
                    "x1" : col * zone_w,
                    "y1" : row * zone_h,
                    "x2" : (col + 1) * zone_w,
                    "y2" : (row + 1) * zone_h,
                    "stocked" : False, # Initially assume all zones are empty
                    "coverage" : 0.0 # Percentage of the zone covered by detected products
                }
            )
    return zones

def compute_zones_coverage(zones:list, boxes:np.ndarray, coverage_threshold=.15):
    """
    For each zones, calculate the percentage of the zone covered by detected products.
    If coverage < theeshold -> then zone is out of stock
    completely geometric
    """

    for zone in zones:
        zx1, zy1, zx2, zy2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        zone_area = (zx2 - zx1) * (zy2 - zy1)

        covered_area = 0.0
        for box in boxes:
            bx1, by1, bx2, by2 = box[:4].astype(int)
            ix1 = max(zx1, bx1)
            iy1 = max(zy1, by1)
            ix2 = min(zx2, bx2)
            iy2 = min(zy2, by2)
            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                covered_area += intersection_area
        coverage = covered_area / zone_area
        zone["coverage"] = round(coverage, 3)
        zone["stocked"] = coverage >= coverage_threshold
    return zones

def detect_oos(
    image_path : str,
    aisle_id : str = aisle_id,
    camera_id : str = camera_id,
    grid_cols : int = 10,
    grid_rows : int = 3):

    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    # Detect products
    results = model(image_path, conf=.3, iou=.45)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # Get bounding boxes as numpy array
    print(f"Detected {len(boxes)} products in the image.")

    zones = divide_shelf_into_zones(W, H, grid_cols, grid_rows)

    zones = compute_zones_coverage(zones, boxes, coverage_threshold=gap_threshold)

    oos_zones = [z for z in zones if not z["stocked"]]
    ok_zones  = [z for z in zones if z["stocked"]]

    print(f"\n  Shelf grid: {grid_rows} rows × {grid_cols} cols "
          f"= {len(zones)} zones")
    print(f" Out-of-stock zones: {len(oos_zones)}")
    print(f"    Stocked zones: {len(ok_zones)}")
    shelf_names = ["Top Shelf", "Middle Shelf", "Bottom Shelf"]

    for z in oos_zones:
        shelf = shelf_names[z["row"]] if z["row"] < len(shelf_names) else f"Shelf {z['row']}"
        col_pct = z["col"] / (grid_cols - 1) if grid_cols > 1 else 0
        pos_label = position_label(col_pct)
        coverage = z["coverage"] * 100

        severity = "HIGH" if coverage < 5 else "MEDIUM" if coverage < 15 else "LOW"
        print(f"  [OOS ALERT] {shelf} - {pos_label} | Gap Coverage: {coverage:.1f}% | Priority: {severity}")
    
    annotated = draw_zone_grid(img.copy(), zones, boxes)
    out_path = image_path.replace(".jpg", "_annotated.jpg")    
    cv2.imwrite(out_path, annotated)
    return oos_zones

def draw_zone_grid(img, zones, boxes):
    for z in zones:
        color = (0, 255, 0) if z["stocked"] else (0, 0, 255)
        alpha = 0.15
        overlay = img.copy()
        cv2.rectangle(overlay, (z["x1"], z["y1"]), (z["x2"], z["y2"]), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (z["x1"], z["y1"]), (z["x2"], z["y2"]), color, 1)
        label = f"{z['coverage']*100:.1f}%"
        cv2.putText(img, label, (z["x1"] + 5, z["y1"] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return img

def position_label(col_pct):
    if col_pct < 0.2: return "Far Left"
    elif col_pct < 0.4: return "Left"
    elif col_pct < 0.6: return "Center"
    elif col_pct < 0.8: return "Right"
    else: return "Far Right"


if __name__ == "__main__":
    test_image = r"C:\Users\HP\Downloads\retail_shelf_detector\data\test\images\test_48_jpg.rf.U2wFbNySExivS4YX1KGs.jpg"
    detect_oos(test_image)


