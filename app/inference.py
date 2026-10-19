import cv2
import numpy as np

def mock_almond_model(image):
    """
    Simulates a machine learning model.
    Returns a list of detections: [class_name, confidence, (x, y, w, h)]
    """
    height, width = image.shape[:2]
    
    # Simulate a "no-detection state" 20% of the time to test the fallback logic
    if np.random.rand() > 0.8:
        return []
        
    # Return a dummy detection in the center of the image
    return [
        {"class": "Almond", "confidence": 0.92, "box": (int(width*0.3), int(height*0.3), int(width*0.4), int(height*0.4))}
    ]

def run_inference_and_overlay(image_path):
    """
    Loads an image, runs inference, and draws bounding boxes.
    """
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 2. Run Inference
    detections = mock_almond_model(image)

    # 3. Handle No-Detection State
    if not detections:
        print("Inference complete: No objects detected.")
        return image

    # 4. Render Overlays
    for det in detections:
        label = det["class"]
        conf = det["confidence"]
        x, y, w, h = det["box"]

        # Draw bounding box (Green)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare text: "Almond 92%"
        text = f"{label} {int(conf * 100)}%"
        
        # Draw text background for readability
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), (0, 255, 0), -1)
        
        # Render text
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    print(f"Inference complete: Detected {len(detections)} object(s).")
    return image