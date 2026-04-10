from ultralytics import YOLO
import torch

def train_shelf_detector(epochs=30): # Increased epochs; 20 is usually too low for convergence
    # load model - consider moving to yolo11m if s is underperforming
    model = YOLO("yolo11s.pt")

    # Training starts
    result = model.train(
        data="/teamspace/studios/this_studio/retail_shelf_detector/data/data.yaml",
        epochs=epochs,
        imgsz=1280,         # INCREASED: Retail items are small; 1280 helps capture tiny gaps
        batch=-1,           # Auto-batch: finds the max your GPU can handle
        device="cuda" if torch.cuda.is_available() else "cpu",
        resume=True,        # Resumes training if interrupted; saves time and GPU resources
        
        # --- AUGMENTATIONS TO FIX BACKGROUND ERRORS ---
        auto_augment = "randaugment",
        mosaic=1.0,         # Combines 4 images; helps the model see objects in context
        mixup=0.15,         # Blends two images; reduces over-reliance on specific backgrounds
        copy_paste=0.3,     # Takes "OOS" gaps and pastes them on other shelves (Great for small datasets)
        hsv_v=0.4,          # Random brightness (Essential for dark shelf shadows)
        degrees=5.0,        # Slight rotation for tilted camera angles
        
        # --- HYPERPARAMETERS ---
        label_smoothing=0.1, # Helps the model generalize; reduces "overconfidence" in wrong labels
        close_mosaic=10,     # Disables mosaic for the last 10 epochs to "clean up" bounding boxes
        
        save=True,
        exist_ok=True,
        seed=43
    )
    return result

if __name__ == "__main__":
    train_shelf_detector()
