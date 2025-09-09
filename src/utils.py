import cv2


def validate_objects(objects, plate):
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