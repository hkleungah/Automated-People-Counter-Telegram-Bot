import cv2
import numpy as np
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from typing import Tuple
from ultralytics import YOLO

def count_people(image_path: np.ndarray, model: YOLO) -> Tuple[np.ndarray, int]:
    """
    Count people in an image and annotate the image.

    Parameters:
        image (np.ndarray): The image where people will be counted.
        model (YOLO): The YOLO object detection model used for detecting people.

    Returns:
        np.ndarray: The annotated image.
        int: The number of people detected in the image.
    """
    # image_path = "AgACAgUAAxkBAANLZS7W_wEX6hJMhDcd0PaDEahuj44AAhW5MRtRqnlVOKe7Kbnh2g4BAAMCAAN5AAMwBA.jpg"
    image = cv2.imread(image_path)

    # Get the class names from the model
    CLASS_NAMES_DICT = model.model.names
    
    # Get the class id for 'person'
    PERSON_CLASS_ID = 0
    
    # Get the detections using YOLO model
    results = model(image)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )

    # Filter the detections for 'person' class
    #person_detections = detections[detections.class_id == PERSON_CLASS_ID]
    
    # Format custom labels
    # labels = [
    #     f"{CLASS_NAMES_DICT[int(class_id)]} {confidence:0.2f}"
    #     for _, confidence, class_id, _ in person_detections
    # ]

    person_detections = [detection for detection in detections if detection[2] == 0]

    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in person_detections
    ]
    
    # Create instance of BoxAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
    
    # Annotate and return the frame and the number of people detected
    annotated_image = box_annotator.annotate(frame=image, detections=person_detections, labels=labels)
    
    return annotated_image, len(person_detections)

        