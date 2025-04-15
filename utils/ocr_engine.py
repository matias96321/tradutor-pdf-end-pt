#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleOCR ONNX Runner - A command-line tool for OCR using PaddleOCR with ONNX runtime.
"""
from typing import List, Tuple, Dict
import numpy as np

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


from ml_models.paddleocr.ppocr_onnx.ppocr_onnx import PaddleOcrONNX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for PaddleOCR ONNX runtime."""
    # General parameters
    use_gpu: bool = False
    
    # Detection parameters
    det_algorithm: str = 'DB'
    det_model_dir: str = './ml_models/paddleocr/en_PP-OCRv3_det_infer.onnx'
    det_limit_side_len: int = 960
    det_limit_type: str = 'max'
    det_box_type: str = 'quad'
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    max_batch_size: int = 10
    use_dilation: bool = False
    det_db_score_mode: str = 'fast'
    
    # Recognition parameters
    rec_algorithm: str = 'SVTR_LCNet'
    rec_model_dir: str = './ml_models/paddleocr/en_PP-OCRv3_rec_infer.onnx'
    rec_image_shape: str = '3, 48, 320'
    rec_batch_num: int = 6
    rec_char_dict_path: str = './ml_models/paddleocr/en_dict.txt'
    use_space_char: bool = True
    drop_score: float = 0.5
    
    # Classification parameters
    use_angle_cls: bool = False
    cls_model_dir: str = './ml_models/paddleocr/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    cls_image_shape: str = '3, 48, 192'
    label_list: List[str] = field(default_factory=lambda: ['0', '180'])
    cls_batch_num: int = 6
    cls_thresh: float = 0.9
    
    # Other parameters
    save_crop_res: bool = False
    
   
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class OCREngine:
    """Processes images using PaddleOCR with ONNX runtime."""

    def __init__(self, config=None):
        """
        Initializes the PaddleOCR processor with the required configuration.
        
        Args:
            config (OCRConfig, optional): Custom configuration for the OCR engine. If not provided, the default configuration is used.
        """
        self.config = config if config else OCRConfig()
        self.ocr_engine = PaddleOcrONNX(self.config)
    
    @staticmethod
    def _validate_image(image: np.ndarray):
        """Ensures the provided image is a valid NumPy array before processing."""
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if image.size == 0:
            raise ValueError("The input image is empty.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento na imagem para melhorar a qualidade do OCR.
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold adaptativo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Reduzir ruído
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Aumentar nitidez
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Converter de volta para BGR
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return processed

    def extract_text(self, image: np.ndarray) -> Tuple[List, List, Dict]:
        """
        Extracts text and relevant metadata from an image using OCR.
        """
        self._validate_image(image)
        # Adicionar pré-processamento antes do OCR
        processed_image = self.preprocess_image(image)
        return self.ocr_engine(processed_image)