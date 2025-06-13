import torch
from ultralytics import YOLO
from easyocr import Reader
import re
import cv2


class ALPR():
    def __init__(self, weigth_path) -> None:
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
    
    def _detect(self, img_arr):
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

    
    def predict(self, img_arr):
        boxes = self._detect(img_arr)
        result_texts = []

        for x_min, y_min, x_max, y_max in boxes:
            cropped_plate = img_arr[y_min:y_max, x_min:x_max]
            ocr_results = self.ocr.readtext(cropped_plate)

            if ocr_results is None:
                continue

            plate_text = ''.join([text for _, text, _ in ocr_results])
            normalized_text = self._normalize(plate_text)
            result_texts.append(normalized_text)

        return result_texts
    

if __name__ == '__main__':
    alpr = ALPR(r'D:\VSCodeProjects\license_plate_regconition\weight\best.pt')
    img_path = r'D:\VSCodeProjects\license_plate_regconition\imgs\plate2.png'
    img_arr = cv2.imread(img_path)
    result = alpr.predict(img_arr)
    print(result)