import cv2


def filter_objects_in_plate(objects, plate):
    (x1, y1), (x2, y2) = plate["landmark"]
    height = plate["height"]

    plate_x_min = int(x1)
    plate_x_max = int(x2)
    plate_y_min = int(y1 - height / 2)
    plate_y_max = int(y2 + height / 2)

    norm_objects = []
    for obj in objects:
        cx, cy = obj["center"]

        if plate_x_min <= cx <= plate_x_max and plate_y_min <= cy <= plate_y_max:
            norm_objects.append(obj)

    return norm_objects


def remove_character_duplicate_boxes(objects, iou_thresh):
    def iou_score(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
    
    filtered_objs = []
    while objects:
        current = objects.pop(0)
        filtered_objs.append(current)
        objects = [obj for obj in objects if iou_score(current['box'], obj['box']) < iou_thresh]

    return filtered_objs

def sort_objects(objects, plate):
    if plate["label"] == "one_row":
        objects = sorted(objects, key=lambda obj: obj["center"][0])

    elif plate["label"] == "two_row":
        (x1, y1), (x2, y2) = plate["landmark"]
        if x2 != x1:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
        else:
            a = float("inf")
            b = 0

        upper, lower = [], []
        for obj in objects:
            cx, cy = obj["center"]

            if a != float("inf"):
                y_line = a * cx + b
                if cy < y_line:
                    upper.append(obj)
                else:
                    lower.append(obj)
            else:
                if cx < x1:
                    upper.append(obj)
                else:
                    lower.append(obj)

        upper = sorted(upper, key=lambda obj: obj["center"][0])
        lower = sorted(lower, key=lambda obj: obj["center"][0])

        objects = upper + lower

    else:
        raise ValueError(
            f"Unsupported plate label '{plate['label']}'. "
            "Expected one of {'car_plate', 'moto_plate'}."
        )

    return objects


def smart_padding(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    mean_val = cv2.mean(gray, mask=mask)[0]
    mean_val = int(mean_val)

    h, w = gray.shape
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    padded = cv2.copyMakeBorder(
        gray, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=mean_val
    )

    return padded
