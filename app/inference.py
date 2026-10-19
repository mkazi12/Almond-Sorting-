import cv2
import numpy as np

def mock_almond_model(image):
    """
    Simulates a machine learning model.
    """
    height, width = image.shape[:2]
    
    # Deterministic test: if the image is mostly gray (our test_empty.jpg), return no detections
    if np.mean(image) < 210:
        return []
        
    # Otherwise, return our dummy almond detection
    return [
        {"class": "Almond", "confidence": 0.92, "box": (int(width*0.3), int(height*0.3), int(width*0.4), int(height*0.4))}
    ]

def run_inference_and_overlay(image_path):
    """
    Loads an image, runs inference, and draws bounding boxes or a no-detection message.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    detections = mock_almond_model(image)

    # Documented No-Detection State
    if not detections:
        cv2.putText(image, "STATE: No Almonds Detected", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image

    # Render Overlays for Detections
    for det in detections:
        label = det["class"]
        conf = det["confidence"]
        x, y, w, h = det["box"]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {int(conf * 100)}%"
        
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), (0, 255, 0), -1)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image