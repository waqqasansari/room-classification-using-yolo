# **Room Classification and Object Detection using YOLOv8**

This project demonstrates how to use **YOLOv8** for object detection and room classification based on the objects present in an image. The model is pretrained on the **COCO dataset** and is capable of detecting key objects that help in classifying various room types (e.g., bedroom, kitchen, bathroom). The project is implemented in a **Python notebook**.

## **Project Overview**

The goal of this project is to detect objects in still images and classify the room type (such as a bedroom, bathroom, kitchen, etc.) based on the detected objects. The model uses a pretrained YOLOv8 architecture for object detection and leverages a **weighted voting system** for room classification, where key objects contribute more to certain room types.

## **Key Features**
- Object detection using YOLOv8.
- Room classification based on detected objects using a **weighted voting system**.
- Pretrained model on the COCO dataset.

## **Getting Started**

### **Requirements**

Ensure you have the following dependencies installed:

- Python 3.6+
- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- TensorFlow (optional, for TensorFlow Lite conversion)
- OpenCV
- Matplotlib

You can install the dependencies using the following command:

```bash
pip install ultralytics opencv-python matplotlib
```

### **Running the Notebook**

1. **Clone this repository**:

    ```bash
    git clone https://github.com/your-username/room-classification-yolov8.git
    cd room-classification-yolov8
    ```

2. **Run the Jupyter Notebook**:

    Open the `Object_detection_and_classification.ipynb` notebook in Jupyter Lab or Jupyter Notebook:

3. **Run the Cells**:

    Execute the notebook cells to load the pretrained YOLOv8 model and run object detection and room classification on your images.

    You can modify the `image_path` variable to test different images.

## **Room Classification Logic**

The model uses a **weighted voting system** to classify the room based on detected objects. The detected objects are assigned different weights based on how indicative they are of a specific room. For example, a **toilet** has a higher weight for a **bathroom**, and a **bed** has a higher weight for a **bedroom**.

Here’s an example of how object weights contribute to room classification:

```python
object_weights = {
    'toilet': {'bathroom': 2},
    'sink': {'bathroom': 1, 'kitchen': 1},
    'oven': {'kitchen': 2},
    'bed': {'bedroom': 2},
    'sofa': {'living room': 2},
    'dining table': {'dining room': 2},
    # ... other objects and their weights
}
```

### **Room Classification Example**

If the model detects a **toilet** and a **sink**, the weighted voting system will classify the image as a **bathroom** since the toilet has a higher weight for the bathroom. If the detected objects include a **refrigerator** and **oven**, the image will be classified as a **kitchen**.

### **Voting Process**

- Each object detected in the image contributes a score to one or more room types based on its relevance.
- The room with the highest total score is selected as the final classification.

### **Objects Considered for Room Classification**

- **Bedroom**: Bed, pillow, wardrobe, nightstand, lamp.
- **Bathroom**: Toilet, sink, bathtub, shower, towel.
- **Kitchen**: Oven, refrigerator, stove, microwave, sink.
- **Living Room**: Sofa, TV, coffee table, bookshelf, armchair.
- **Dining Room**: Dining table, chairs, plates.

## **Improving the Model**

Here are a few ways to enhance the performance and extend the capabilities of the model:

### **1. Fine-Tuning on Custom Dataset**
If you need the model to detect additional objects that are not in the COCO dataset (e.g., specific furniture or decor), you can fine-tune the model:
- **Collect a dataset** with the new objects you want to detect.
- Use a tool like **LabelImg** to label the objects in the images.
- Fine-tune the YOLOv8 model by following these steps:

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8n.pt')

# Train the model with your custom dataset
model.train(data='custom_dataset.yaml', epochs=50, imgsz=640)
```

### **2. Add More Room-Specific Objects**
You can increase classification accuracy by fine-tuning the model with more room-specific objects. For example:
- **Bedroom**: Lamps, paintings, mirrors.
- **Kitchen**: Blenders, coffee makers, cutting boards.
- **Bathroom**: Toothbrushes, soap dispensers.

### **3. Post-Training Quantization for Mobile Deployment**
If you plan to deploy this model on mobile devices, you can use **TensorFlow Lite** to optimize it for mobile environments:

```python
# Export to TensorFlow Lite format
model.export(format="tflite")
```

This will generate a lightweight `.tflite` model that can run efficiently on mobile devices.

### **4. Increase Dataset Diversity**
To make the model more robust, you can gather a more diverse dataset with varying lighting conditions, camera angles, and object positions. This will help the model generalize better to different environments.

## **Contributing**

Contributions are welcome! If you'd like to contribute to the project, feel free to submit a pull request or open an issue on GitHub.

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
