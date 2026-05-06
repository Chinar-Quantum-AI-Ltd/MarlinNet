from ultralytics import YOLO

waste_detector = YOLO('yolov8s.pt')

def detect_waste(obs):
    img = (obs.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    results = waste_detector(img)
    return [waste_detector.names[int(b.cls)] for r in results for b in r.boxes]
