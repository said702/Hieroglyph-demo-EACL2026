import numpy as np
import subprocess
import os
from typing import List
import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont


def multi_frame_denoising(curr_frame, frame_queue):
    """
    Apply simple temporal denoising by averaging the last frames.
    """
    frame_queue.append(curr_frame)
    denoised_frame = np.mean(frame_queue, axis=0).astype(np.uint8)
    return denoised_frame


def draw_legend(frame, tr, font_scale=0.8, thickness=2):
    """
    Draw the interactive legend at the bottom-left corner of the frame.
    Items are listed from bottom to top.
    """
    legend = [
        "h - Show/Hide Help",
        "q - Quit Program",
        "n - Reset Program",
        "y - Decrease Font Size",
        "x - Increase Font Size",
        "t - Disable Hieroglyph Translation" if tr else "t - Enable Hieroglyph Translation",
        "ENTER - Start Hieroglyph Classification",
    ]

    x_offset = 10  # Left margin

    # Dynamically compute text height
    (text_w, text_h), _ = cv2.getTextSize(
        "Sample", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    line_height = text_h + 10  # Add small spacing

    # Start from bottom-left and go upwards
    y_start = frame.shape[0] - 20  # margin from bottom

    for i, text in enumerate(legend):
        y = y_start - i * line_height
        cv2.putText(
            frame,
            text,
            (x_offset, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 140, 255),  # Dark orange (BGR)
            thickness,
            cv2.LINE_AA,
        )


def draw_process_text(frame, text, font, font_scale, thickness):
    """
    Draw a centered processing message in the middle of the frame.
    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 100, 0),  # Dark green
        thickness,
        cv2.LINE_AA,
    )


