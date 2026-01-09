from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image_path, conf_threshold=0.4):
        results = self.model(image_path)
        detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf >= conf_threshold:
                    detections.append({
                        "label": self.model.names[int(box.cls)],
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist()
                    })
        return detections
