from dataclasses import dataclass
from typing import List, Literal
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from ml_models.unilm.dit.object_detection.ditod import add_vit_config

@dataclass
class Layout:
    type: Literal["text", "title", "list", "table", "figure"]
    bbox: tuple[int, ...]
    score: float
    image: np.ndarray = None

class LayoutAnalyzer:
    def __init__(self):
        self.device: str = "cpu"
        self.config_path="/ml_models/unilm/config/cascade_dit_base.yaml",
        self.weights_path="/ml_models/unilm/config/publaynet_dit-b_cascade.pth"
        self.cfg = self._setup_config(self.config_path, self.weights_path, self.device)
        self.predictor = DefaultPredictor(self.cfg)
        self.class_id_dictionary = { 0: "text", 1: "title", 2: "list", 3: "table", 4: "figure" }

    def _setup_config(self, config_path: str, weights_path: str, device: str):
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file("./ml_models/unilm/config/cascade_dit_base.yaml")
        cfg.MODEL.WEIGHTS = "./ml_models/unilm/config/publaynet_dit-b_cascade.pth"
        cfg.MODEL.DEVICE = device
        return cfg

    def analyze(self, image: np.array, score_threshold: float = 0.90,device: str = "cpu") -> List[Layout]:
        
        output = self.predictor(image)["instances"].to(device)
        layouts = []

        for class_id, box, score in zip(
            output.pred_classes.numpy(),
            output.pred_boxes.tensor.numpy().astype(int),
            output.scores.numpy(),
        ):
            if score > score_threshold and self.class_id_dictionary[class_id] == 'text' or self.class_id_dictionary[class_id] == 'title':
                layouts.append(
                    Layout(
                        type=self.class_id_dictionary[class_id],
                        bbox=tuple(box),
                        score=score,
                        image=self._extract_region(image, tuple(box)),
                    )
                )

        return self._non_maximum_suppression(layouts)

    @staticmethod
    def _extract_region(image: np.ndarray, bbox: tuple[int, ...]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[int(y1):int(y2), int(x1):int(x2)]

    @staticmethod
    def _calculate_iou(boxa, boxb) -> float:
        x1_min, y1_min, x1_max, y1_max = boxa
        x2_min, y2_min, x2_max, y2_max = boxb

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min + 1)
        inter_height = max(0, y_inter_max - y_inter_min + 1)
        intersection_area = inter_width * inter_height

        boxa_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        boxb_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

        union_area = boxa_area + boxb_area - intersection_area
        return intersection_area / union_area if union_area != 0 else 0

    def _non_maximum_suppression(self, layouts: List[Layout], iou_threshold: float = 0.5) -> List[Layout]:
        if not layouts:
            return []

        layouts.sort(key=lambda x: x.score, reverse=True)
        non_overlapping_layouts = []

        while layouts:
            current_layout = layouts.pop(0)
            if all(self._calculate_iou(current_layout.bbox, other.bbox) <= iou_threshold for other in non_overlapping_layouts):
                non_overlapping_layouts.append(current_layout)

        return non_overlapping_layouts