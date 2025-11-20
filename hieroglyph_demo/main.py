import cv2
import threading
from collections import deque
from pathlib import Path
import os
import sys
import distutils.core
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ctypes 
import argparse



# -------------------------------------------------------------------------
# Argument parser for camera index
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hieroglyph Prototype Demo")
parser.add_argument(
    "--camera",
    type=int,
    default=1, # Default camera if no argument is provided
    help="Camera index to use (default: 1)"
)
args = parser.parse_args()
camera_index = args.camera



# -------------------------------------------------------------------------
# Custom modules
# -------------------------------------------------------------------------
from .utils import (
    multi_frame_denoising,
    draw_process_text,
    draw_legend
)

from .hieroglyph_detection import (
    clone_repo,
    draw_bbs_lines,
    draw_classes,
    load_Hierogloph_models,
    find_longest_lines,
    center_lines,
    process_and_label_image,
    determine_arrangement)

from .hieroglyph_translation import (
    translate_hieroglyph_symbols,
    draw_translation_under_bbs,
    ensure_hiero_ready
)

# -------------------------------------------------------------------------
# Detectron2 imports (after cloning detectron2)
# -------------------------------------------------------------------------
clone_repo("https://github.com/facebookresearch/detectron2.git", "detectron2")
sys.path.insert(0, os.path.abspath("detectron2"))
dist = distutils.core.run_setup("detectron2/setup.py")
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# -------------------------------------------------------------------------
# Loading hieroglyph models...
# -------------------------------------------------------------------------
print("\nLoading hieroglyph models...")
weights_dir = r"weights"
load_Hierogloph_models(weights_dir)

# YOLO models for hieroglyph pipeline
model1 = YOLO(os.path.join(weights_dir, "YOLO11n_Leserichtung.pt"))   # reading direction
model2 = YOLO(os.path.join(weights_dir, "YOLO11m_15_01_000001.pt"))   # symbol classification

# Detectron2 config for hieroglyph segmentation (Mask R-CNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # single class "Hieroglyph"
cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "segm_4000_512_01.pth")
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
predictor = DefaultPredictor(cfg)

# Load and initialize the hieroglyph translation model
ensure_hiero_ready()



# -------------------------------------------------------------------------
# Hieroglyph processing thread
# -------------------------------------------------------------------------
def Hyrogloph(frame_den):
    """
    Run the hieroglyph pipeline in a separate thread:
    1. Predict reading direction (model1).
    2. Detect hieroglyph instances with Detectron2.
    3. Group them into lines and determine arrangement.
    4. Classify symbols (model2).
    5. Optionally translate the sequence.
    """
    global hyr_running, hyr_done, hyr_bbs
    global bboxes_hyr, arrangement, centered_lines
    global predicted_classes, all_image_crops_and_bboxes, translated_text

    hyr_running = True

    # 1) Predict reading direction
    results_dir_model = model1.predict(frame_den, verbose=False)
    predicted_class_index = results_dir_model[0].probs.top1
    direction = model1.names[predicted_class_index]

    # 2) Mask R-CNN hieroglyph detection
    outputs = predictor(frame_den)
    instances = outputs["instances"].to("cpu")
    bboxes_hyr = instances.pred_boxes.tensor.numpy()

    # 3) Determine lines and arrangement
    longest_lines = find_longest_lines(bboxes_hyr, frame_den.shape[1], frame_den.shape[0])
    centered_lines = center_lines(longest_lines)
    arrangement, x_midpoints, y_midpoints = determine_arrangement(centered_lines, bboxes_hyr)
    print(arrangement)
    hyr_bbs = True

    # 4) Symbol classification 
    predicted_classes, all_image_crops_and_bboxes = process_and_label_image(
        frame_den,
        bboxes_hyr,
        "LR",
        model2,
        centered_lines,
        arrangement,
        x_midpoints,
        y_midpoints,
    )
  
    # 5) Optional translation of the symbol sequence
    if tr:
        print("Predicted classes:", predicted_classes)
        translated_text = translate_hieroglyph_symbols(
            predicted_classes,
            src_lang="en",
            tgt_lang="en",
        )
        print("Translated text:", translated_text)

    hyr_running = False
    hyr_done = True


def reset_hieroglyph_state():
    """
    Reset all variables related to the hieroglyph mode.
    This is called when the user presses 'n'.
    """
    global hyr_running, hyr_done, hyr_bbs
    global bboxes_hyr, centered_lines, arrangement
    global predicted_classes, all_image_crops_and_bboxes, translated_text, tr, Hieroglyph_Modus
    hyr_running = False
    hyr_done = False
    hyr_bbs = False
    bboxes_hyr = []
    centered_lines = []
    arrangement = None
    predicted_classes = []
    all_image_crops_and_bboxes = []
    translated_text = ""
    tr = False  
    Hieroglyph_Modus = False



# -------------------------------------------------------------------------
# Screen resolution
# -------------------------------------------------------------------------
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)   # 0 = SM_CXSCREEN
screen_height = user32.GetSystemMetrics(1)  # 1 = SM_CYSCREEN

# -------------------------------------------------------------------------
# Video capture initialisation
# -------------------------------------------------------------------------
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"[ERROR] Could not open camera with index {camera_index}.")
    print("Please choose a valid camera using --camera <index>.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)# -------------------------------------------------------------------------
cv2.namedWindow("Prototyp", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Prototyp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.resizeWindow("Prototyp", screen_width, screen_height)
cv2.moveWindow("Prototyp", 0, 0)

# -------------------------------------------------------------------------
# Global state variables
# -------------------------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 2
color = (255, 0, 0)
font_scale2 = 0.8
frame_count = 0
Hilfe = True
frame_queue = deque(maxlen=5)
reset_hieroglyph_state()

# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Denoise / temporal smoothing
    frame = multi_frame_denoising(frame, frame_queue)
    frame_den = frame
    frame_display = frame_den.copy()

    frame_count += 1

    # ---------------------------------------------------------------------
    # Normal mode: text detection and feedback projection
    # ---------------------------------------------------------------------
    if hyr_running and not hyr_done and not hyr_bbs:
        draw_process_text(frame_display, "Hieroglyphs are determined...", font, font_scale, thickness)

    if hyr_bbs:
        draw_bbs_lines(frame_display, bboxes_hyr, arrangement, centered_lines)
        if not hyr_done:
            draw_process_text(frame_display, "Symbols are determined", font, font_scale, thickness)

    if hyr_done:
        draw_classes(frame_display, predicted_classes, all_image_crops_and_bboxes, font_scale2)
        if tr:
            draw_translation_under_bbs(
                frame_display,
                predicted_classes,
                all_image_crops_and_bboxes,
                translated_text,
            )

    # ---------------------------------------------------------------------
    # Common overlays and display
    # ---------------------------------------------------------------------
    if Hilfe:
        draw_legend(frame_display, tr)

    cv2.imshow("Prototyp", frame_display)
    prev_frame = frame.copy()

    # ---------------------------------------------------------------------
    # Keyboard control
    # ---------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    # Toggle hieroglyph mode and start processing thread
    if key in [ord("\r"), ord("\n")]:
        Hieroglyph_Modus = not Hieroglyph_Modus
        if Hieroglyph_Modus:
            # Start hieroglyph processing on the current frame
            thread = threading.Thread(target=Hyrogloph, args=(frame_den,))
            thread.start()

    elif key == ord("h"):
        Hilfe = not Hilfe

    elif key == ord("t"):
        tr = not tr

    elif key == ord("x"):
        font_scale2 += 0.125

    elif key == ord("y"):
        font_scale2 -= 0.125

    elif key == ord("n"):
        # Reset state
        reset_hieroglyph_state()
        Hieroglyph_Modus = False

    elif key == ord("q"):
        break

# -------------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
