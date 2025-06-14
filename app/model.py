import torch
from ultralytics import YOLO
from easyocr import Reader
import re
import os
import numpy as np


class ALPR():
    def __init__(
        self, 
        weigth_path = os.path.join(os.getcwd(), 'weightS', 'best.pt')
    ):
        self.weight_path = weigth_path
        self.yolo = self._load_yolo_model()
        self.ocr = self._load_ocr_model()

    def _load_yolo_model(self) -> YOLO:
        yolo = YOLO(self.weight_path)
        return yolo
    
    def _load_ocr_model(self) -> Reader:
        use_gpu = torch.cuda.is_available()
        ocr = Reader(['en'], gpu=use_gpu)
        return ocr
    
    def detect(self, img_arr):
        h, w = img_arr.shape[:2]
        yolo_result = self.yolo.predict(source=img_arr, verbose=False)
        boxes = yolo_result[0].boxes

        if boxes is None:
            return []

        cleaned_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(w, int(x_max))
            y_max = min(h, int(y_max))
            cleaned_boxes.append([x_min, y_min, x_max, y_max])

        return cleaned_boxes
    
    def _normalize(self, text):
        norm_text = re.sub(r'[^a-zA-Z0-9]', '', text)
        return norm_text

    
    def predict(self, img_arr: np.ndarray):
        boxes = self.detect(img_arr)
        
        result_texts = []

        for x_min, y_min, x_max, y_max in boxes:
            cropped_plate = img_arr[y_min:y_max, x_min:x_max]
            ocr_results = self.ocr.readtext(cropped_plate)

            if not ocr_results:
                continue

            plate_text = ''.join([text.upper() for _, text, _ in ocr_results])
            normalized_text = self._normalize(plate_text)
            result_texts.append(normalized_text)

        return result_texts
    