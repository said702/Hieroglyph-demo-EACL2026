import os
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import cv2
import time

# Zielordner für das Modell (relativ zu Main.py)
_HIERO_DIR = os.path.join(os.getcwd(), "translator_hieroglyph")

# Globale Caches
_HIERO_TOKENIZER = None
_HIERO_MODEL = None

# Benötigte Dateien
_HF_FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "pytorch_model.bin",
]


def _ensure_hiero_files():
    """Sorgt dafür, dass alle benötigten Dateien lokal liegen."""
    os.makedirs(_HIERO_DIR, exist_ok=True)

    for fname in _HF_FILES:
        fpath = os.path.join(_HIERO_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            continue

        hf_hub_download(
            repo_id="mattiadc/hiero-transformer",
            filename=fname,
            local_dir=_HIERO_DIR,
            local_dir_use_symlinks=False,
        )


def _load_hiero_model_once():
    """Lädt Tokenizer & Modell nur einmal."""
    global _HIERO_TOKENIZER, _HIERO_MODEL

    if _HIERO_TOKENIZER is not None and _HIERO_MODEL is not None:
        return

    # Torch-Version prüfen
    try:
        _torch_ver_str = torch.__version__.split("+")[0]
        _torch_ver = tuple(int(x) for x in _torch_ver_str.split(".")[:2])
    except Exception:
        _torch_ver = (0, 0)

    if _torch_ver < (2, 6):
        raise RuntimeError(
            f"Torch {torch.__version__} ist zu alt. Bitte torch>=2.6.0 installieren."
        )

    # Laden
    _HIERO_TOKENIZER = AutoTokenizer.from_pretrained(
        _HIERO_DIR,
        local_files_only=True,
        use_fast=False,
    )

    _HIERO_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        _HIERO_DIR,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()


def ensure_hiero_ready():
    """Public Init."""
    _ensure_hiero_files()
    _load_hiero_model_once()


def translate_hieroglyph_symbols(symbols, src_lang="en", tgt_lang="en",
                                 num_beams=4, max_length=128):
    """
    Übersetzt Hieroglyphen-Codes (['P5','I14',...]).
    """
    if not isinstance(symbols, (list, tuple)):
        raise ValueError("symbols muss eine Liste sein")

    text = " ".join(symbols)

    _HIERO_TOKENIZER.src_lang = src_lang
    _HIERO_TOKENIZER.tgt_lang = tgt_lang

    inputs = _HIERO_TOKENIZER([text], return_tensors="pt")

    with torch.no_grad():
        out = _HIERO_MODEL.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            forced_bos_token_id=_HIERO_TOKENIZER.get_lang_id(_HIERO_TOKENIZER.tgt_lang),
        )

    decoded = _HIERO_TOKENIZER.batch_decode(out, skip_special_tokens=True)[0]
    return decoded


def draw_translation_under_bbs(frame, predicted_classes,
                               all_image_crops_and_bboxes, translated_text):
    """Schreibt die Übersetzung unter die Bounding Boxes."""
    if isinstance(translated_text, list):
        translated_text = " ".join(translated_text)

    if not translated_text.strip():
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    font_scale = 0.9
    thickness = 2

    try:
        max_y2 = max(bbox[1][3] for bbox in all_image_crops_and_bboxes)
    except Exception:
        return

    (tw, th), baseline = cv2.getTextSize(translated_text, font, font_scale, thickness)

    cx = frame.shape[1] // 2
    text_x = cx - tw // 2
    text_y = int(max_y2 + th + 30)

    if text_y > frame.shape[0] - 10:
        text_y = frame.shape[0] - 20

    cv2.rectangle(frame,
                  (text_x - 8, text_y - th - 8),
                  (text_x + tw + 8, text_y + baseline + 8),
                  (0, 0, 0), -1)

    cv2.putText(frame, translated_text, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
