import cv2
import numpy as np
import os
import re
from ultralytics import YOLO
import pytesseract
from PIL import Image
import requests
import hashlib
from collections import deque
import matplotlib.pyplot as plt
# Import your database
from core.database import AthleteDatabase

from core.image_cache import ImageCache, get_image_hash

# This stays alive as long as the backend is running
try:
    ATHLETE_DB = AthleteDatabase()
except Exception as e:
    print("Database connection error:", e)

try:
    RESULT_CACHE = ImageCache(db=2)
except Exception as e:
    print(f"Cache connection error: {e}")
    RESULT_CACHE = None

# ==========================================
# GLOBAL CONFIGURATION & MODEL LOADING
# ==========================================

#multiple background threads limit OS problem prevent thread explosion
os.environ["OMP_THREAD_LIMIT"] = "1"

try:
    #pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

except Exception as e:
    print(f"Tesseract config warning: {e}")

# Expand the home directory symbol '~' and set the base models folder
MODELS_DIR = os.path.expanduser("~/Senior_Project/final_project/didatades_project/models")
MODEL_DIR_WINDOWS = "C:/Users/Luke/Documents/Semester8/SeniorProject/project/final_project/spotlight_project"
#MODEL_DIR_WINDOWS = r"C:\Users\Luke\Documents\Semester8\SeniorProject\project\final_project\spotlight_project"
PERSON_MODEL_PATH = os.path.join(MODEL_DIR_WINDOWS, 'models', 'yolov8s.pt')
BIB_MODEL_PATH = os.path.join(MODEL_DIR_WINDOWS, 'models', 'number632resol.pt') #best.pt

print("Loading global YOLO models...")

try:
    GLOBAL_PERSON_MODEL = YOLO(PERSON_MODEL_PATH)
except Exception as e:
    print(f"YOLO Person Model Load Error: {e}. Defaulting to None.")
    GLOBAL_PERSON_MODEL = None

try:
    GLOBAL_BIB_MODEL = YOLO(BIB_MODEL_PATH)
except Exception as e:
    print(f"YOLO Bib Model Load Error: {e}. Defaulting to None.")
    GLOBAL_BIB_MODEL = None


# ==========================================
# GLOBAL HELPER FUNCTIONS
# ==========================================

def load_image(url_or_path):
    """Global helper to load an image from a URL or local file path."""
    if cv2 is None: return None # Safety check

    if url_or_path.startswith("http"):
        try:
            response = requests.get(url_or_path, timeout=10)
            response.raise_for_status()

            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
    else:
        if not os.path.exists(url_or_path):
            return None
        return cv2.imread(url_or_path)

# ==========================================
# MODULAR DETECTION & OCR FUNCTIONS
# ==========================================

def detect_person(frame, conf_thresh=0.5, p_pad=0.0):
    """
    Runs the YOLO person model with padding hyperparameters.
    Returns a list of dictionaries with coordinates and cropped images.
    """
    detected_persons = []
    if GLOBAL_PERSON_MODEL is None or frame is None:
        return detected_persons

    results = GLOBAL_PERSON_MODEL(frame, classes=[0], conf=conf_thresh, verbose=False)
    h, w = frame.shape[:2]

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            
            # Apply hyperparameter padding (p_pad)
            pw, ph = int((x2 - x1) * p_pad), int((y2 - y1) * p_pad)
            px1, py1 = max(0, x1 - pw), max(0, y1 - ph)
            px2, py2 = min(w, x2 + pw), min(h, y2 + ph)

            detected_persons.append({
                "x1": px1, "y1": py1, "x2": px2, "y2": py2,
                "frame_crop": frame[py1:py2, px1:px2] 
            })
            
    return detected_persons


def detect_bib(athlete_crop, conf_thresh=0.28, b_pad=0.0):
    """
    Runs the YOLO bib model on an athlete crop with padding hyperparameters.
    Returns the cropped bib image, or None if not found.
    """
    if GLOBAL_BIB_MODEL is None or athlete_crop is None:
        return None

    results = GLOBAL_BIB_MODEL(athlete_crop, conf=conf_thresh, verbose=False)
    h, w = athlete_crop.shape[:2]

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            
            if cls == 0 and conf > conf_thresh: 
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().tolist())
                
                # Apply hyperparameter padding (b_pad)
                bw, bh = int((bx2 - bx1) * b_pad), int((by2 - by1) * b_pad)
                px1, py1 = max(0, bx1 - bw), max(0, by1 - bh)
                px2, py2 = min(w, bx2 + bw), min(h, by2 + bh)
                
                return athlete_crop[py1:py2, px1:px2]
                
    return None


def extract_bib_number(bib_crop, psm=7, scale=1.5, area=200, method='otsu', border=0):
    """
    Processes a bib crop using advanced, parameter-driven OCR.
    Strictly extracts numbers and eliminates the letter fallback logic.
    """
    try:
        # 1. Grayscale & Resize
        gray = cv2.cvtColor(bib_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Add Hyperparameter Border Padding
        if border > 0:
            gray = cv2.copyMakeBorder(
                gray, border, border, border, border,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        # 3. Thresholding Choice (Adaptive vs Otsu)
        if method == 'adaptive':
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 4. Filter Noise by Contour Area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        mask = np.zeros(thresh.shape, dtype=np.uint8)
        for c in cnts:
            if cv2.contourArea(c) > area:
                cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # 5. Clean and Invert
        cleaned = cv2.bitwise_and(thresh, mask)
        cleaned = cv2.bitwise_not(cleaned)

        # Save debug images so you can see exactly what Tesseract sees
        cv2.imwrite("ocr_target_bib_crop.jpg", bib_crop)
        cv2.imwrite("ocr_debug_threshold.jpg", cleaned)

        # 6. OCR (Strict Digits Only)
        pil_img = Image.fromarray(cleaned)
        config = (
            f"--psm {psm} "
            f"-c tessedit_char_whitelist=0123456789 "
            f"-c load_system_dawg=0 "
            f"-c load_freq_dawg=0"
        )
        text = pytesseract.image_to_string(pil_img, config=config)

        # Strip out anything that isn't a digit
        bib_number = "".join(filter(str.isdigit, text))

        if bib_number:
            print(f"[DEBUG] Found bib: {bib_number}")
            return bib_number
        else:
            print("[DEBUG] OCR found absolutely no digits on the bib.")
            return None

    except Exception as e:
        # Catch unexpected crashes and return None gracefully
        print(f"[ERROR] OCR pipeline failed: {e}")
        return None


# ==========================================
# CORE VISION PIPELINE FUNCTIONS
# ==========================================
def process_entire_frame(frame):
    """
    Takes a raw frame, detects all people, finds their bibs, and extracts the numbers.
    Returns the enriched list of dictionaries, the count of bibs found, and a list of numbers.
    """
    # 1. Get the list of people in the current frame (using our modular function)
    current_frame_boxes = detect_person(frame, conf_thresh=0.5, p_pad=0.0)
    
    bibs_in_frame = 0
    detected_numbers = []

    # 2. Loop through every person found and proactively find their bib/number
    for person in current_frame_boxes:
        # Detect bib on the person crop
        bib_crop = detect_bib(person["frame_crop"], conf_thresh=0.28, b_pad=0.0)
        
        # Store properties directly in the memory dictionary
        person["bib_crop"] = bib_crop 
        person["bib_number"] = None
        
        if bib_crop is not None:
            bibs_in_frame += 1
            
            # Run OCR proactively with your tuned hyperparameters
            bib_number = extract_bib_number(
                bib_crop=bib_crop, 
                scale=1.5, 
                area=50, 
                border=0, 
                psm=11
            )
            
            if bib_number:
                person["bib_number"] = bib_number
                detected_numbers.append(bib_number)

    # Return the fully enriched data and the stats
    return current_frame_boxes, bibs_in_frame, detected_numbers


def fast_person_detection(frame):
    """Wrapper utilizing the modularized person detection."""
    # Note: p_pad is set to 0.0 by default, tweak as needed!
    return detect_person(frame, conf_thresh=0.5, p_pad=0.0)


def find_clicked_athlete_crop(click_x, click_y, history_snapshot):
    """
    history_snapshot: A list of tuples [(image, boxes), (image, boxes)...]
    We check the history backwards (most recent first).
    """
    for frame_img, frame_boxes in history_snapshot:
        for box in frame_boxes:
            # Check if click is within the person's bounding box
            if box["x1"] <= click_x <= box["x2"] and box["y1"] <= click_y <= box["y2"]:
                
                # CROP LOGIC: We crop from the frame_img associated with these boxes
                # Adding slight padding to the person crop helps the bib detector
                y1, y2 = max(0, box["y1"]), min(frame_img.shape[0], box["y2"])
                x1, x2 = max(0, box["x1"]), min(frame_img.shape[1], box["x2"])
                
                athlete_crop = frame_img[y1:y2, x1:x2]
                
                print(f"[SYSTEM] Temporal Match! Found athlete in history.")
                return athlete_crop
            
    return None


def process_clicked_athlete(athlete_crop_img):
    """
    Enhanced with Redis DB 2 Result Caching.
    Checks memory before running expensive OCR.
    """
    if athlete_crop_img is None:
        return None

    # 1. Generate a unique hash for this specific crop
    img_hash = get_image_hash(athlete_crop_img)

    # 2. Check Redis DB 2 first
    if RESULT_CACHE:
        cached_result = RESULT_CACHE.check_cache(img_hash)
        if cached_result:
            print(f"[CACHE HIT] Reusing Bib: {cached_result.get('bib')}")
            return cached_result.get('bib')

    # --- CACHE MISS: Run Heavy Computation ---
    
    # A. Run YOLO Bib Detector
    bib_crop = detect_bib(athlete_crop_img, conf_thresh=0.28, b_pad=0.0)
    
    if bib_crop is None:
        # Even a 'None' result is worth caching for 2 seconds to prevent re-scanning a blank area
        if RESULT_CACHE:
            RESULT_CACHE.save_to_cache(img_hash, {"bib": None}, expire_in_hours=0) 
            # Note: Tweak your save_to_cache to accept seconds or use a very small hour fraction
        return None

    # B. Run Pytesseract OCR
    bib_number = extract_bib_number(
        bib_crop=bib_crop, 
        scale=3.0, 
        psm=7
    )

    # 3. Save result to Redis DB 2 (Expires in 5 seconds)
    # This prevents duplicate OCR if the runner stays on screen for 25 frames
    if RESULT_CACHE and bib_number:
        RESULT_CACHE.save_to_cache(img_hash, {"bib": bib_number}, expire_in_seconds=10)
        print(f"[CACHE SAVE] Stored result for hash {img_hash[:8]}...")

    return bib_number