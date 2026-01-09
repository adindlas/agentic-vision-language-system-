import cv2

def draw_boxes(image_path, detections):
    img = cv2.imread(image_path)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f"{d['label']} {d['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 0, 0), 2
        )
    return img
