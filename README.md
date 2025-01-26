# car_dent_task
This project is designed to detect car damages like scratches, dents, and cracks using advanced computer vision techniques. By leveraging deep learning models such as YOLO and CNN, the system efficiently identifies and classifies various damage types with high accuracy.
 # Dataset selection and preparation process.
 goal to choose this project: Automates the process of assessing car damages, reducing the time required for manual inspection. The system can quickly identify and classify damages such as dents, scratches, or cracks from images or videos.
This leads to faster claim processing, enabling insurers to provide quicker responses to customers.
# prerequisite
python 3.11.4
process of data cleaning or image processing 
1. download datadet from kaggle
2. annotate the data with makesense.ai and then augment it with roboflow 100 image i have get as sample dataset then after augmentation flip,rotate,blur,noise like that 3x data.
3. train model in yolov8n and get the pt(pytorch)file.
4. we test it on visual studio code and for testing the code i am used i am upload it.
5. get best output with datset and make pdf of it and upload it.
# confusion matrix of yolov8
![confusion_matrix](https://github.com/user-attachments/assets/e94d8004-d240-447d-b52f-2509e004da21)
# confusion matrix normalized
![confusion_matrix_normalized](https://github.com/user-attachments/assets/3819de9e-07b4-4495-905d-071e7fbf5e23)
# precision and recall curve
![PR_curve](https://github.com/user-attachments/assets/6cacb315-2453-4098-add0-0d2749a37cb2)
# f1 curve 
![F1_curve](https://github.com/user-attachments/assets/a9df9c6e-4107-4803-853f-bdc60a698c3d)
# precision confidence curve
![P_curve](https://github.com/user-attachments/assets/5f31f725-9577-41b8-920b-9f8791b85f40)
# accuracy
with this datadet we get 70 to 80% accuracy with unseen and real world data to improve the accuracy we have such more or large data for better accuracy.
## convolution neural network model 
1. we train same data on cnn for comparison of two model accuracy
2. convert it on coco or json format and then train it with code that i have uploaded the google collab link https://colab.research.google.com/drive/1SHh5h6gGj3AUFihWBHY9GAPfu_jQwTkh?usp=sharing
# performance and evaluation of cnn model 
![WhatsApp Image 2025-01-26 at 3 24 33 PM](https://github.com/user-attachments/assets/1c7e21f0-00bf-40a2-b7ee-e88b378f63d0)
# output of cnn model testing
output of testing cnn i have upload it as car_dent_task_cnn_pdf
# conclusion 
After evaluating both the YOLOv8 and CNN models, we found that YOLOv8 delivers the best performance for object detection tasks. Its ability to process images efficiently, coupled with high accuracy, makes it a preferred choice for real-time applications. While CNNs are powerful for various tasks, YOLOv8's speed and precision in detecting objects provide superior output, confirming it as the optimal model for object detection in our project.
# comparison of yolov8 and cnn model is following
The primary difference between YOLOv8 and CNN models for object detection is that YOLOv8 is specifically optimized for real-time performance, achieving high accuracy and speed in detecting multiple objects in an image. CNNs, while effective at feature extraction and classification, generally do not perform as well in terms of detection speed and accuracy for complex object detection tasks. YOLOv8 uses advanced techniques like anchor boxes, multi-scale predictions, and a fast processing pipeline to deliver superior accuracy in object detection compared to traditional CNNs.






