import cv2
import numpy as np
from ultralytics import Cnn
from ultralytics.utils.plotting import Annotator
import os

# Load the model
model = cnn(r'G:\car damage poc\cnn_model.pth')
class_labels = ["Broken", "dent", "scratched",]

# Create a folder to save the detected images
output_folder = r"G:\car damage poc\output 1"  # Folder to save processed images
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# Define confidence scores for each class
conf_scores = {
    'Broken': 0.05,
    'dent': 0.03,
    'scratched': 0.00,
    
}

# Define the input folder path
folder_path = r"G:\car damage poc\new images"  # Folder containing the images

# List all images in the folder (assuming .jpg and .png formats)
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', 'JPEG'))]

# Ensure there are images in the folder
if len(image_files) == 0:
    print("No images found in the specified folder.")
    exit()

# Loop through each image in the folder
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    # Read the image
    frame = cv2.imread(image_path)
    
    # Check if the frame is successfully captured
    if frame is not None:
        resize_img = cv2.resize(frame, (1024, 1024))
        img = resize_img.copy()

        # Predict using the YOLO model
        results = model.predict(source=resize_img, conf=min(conf_scores.values()), imgsz=1024, iou=0.8)

        annotator = Annotator(img)

        # Store the best detection per class
        best_detections = {label: None for label in class_labels}

        # Loop through each detected defect
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                b = box.xyxy[0]
                c = int(box.cls)
                class_label = class_labels[c]
                confidence = box.conf.item()

                # Check if the prediction confidence is above the set threshold for this class
                if confidence > conf_scores[class_label]:
                    # Update best detection if it's the highest confidence so far
                    if best_detections[class_label] is None or confidence > best_detections[class_label][1]:
                        best_detections[class_label] = (b, confidence)

        # Annotate the image with the best detections
        for label, detection in best_detections.items():
            if detection is not None:
                b, confidence = detection
                annotator.box_label(b, label, color=(0, 255, 0))

        # Save the annotated frame
        annotated_frame = annotator.result()
        output_image_path = os.path.join(output_folder, f"detected_{image_file}")
        cv2.imwrite(output_image_path, annotated_frame)

        print(f"Processed and saved: {output_image_path}")

print("All images processed.")