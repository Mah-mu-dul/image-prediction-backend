from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

# Load the YOLOv8 model
model = YOLO("./model.pt")

# List of class names (replace with your actual class names)
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    results = model(image)
    # print(results)
    return results


@app.route("/")
def home():
    return "Welcome to YOLOv8 prediction API"


@app.route("/predict/", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()

        # Print the file name
        print(f"File received: {file.filename}")

        results = predict(image_bytes)

        # Process results
        results_list = []
        for result in results:
            boxes = (
                result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, "xyxy") else []
            )
            confidences = (
                result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else []
            )
            classes = (
                result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else []
            )

            if len(boxes) != len(confidences) or len(boxes) != len(classes):
                continue  # Skip this result if there is a mismatch

            for box, confidence, cls in zip(boxes, confidences, classes):
                result_dict = {
                    "box": box.tolist(),
                    "confidence": float(confidence),
                    "class_name": CLASS_NAMES[int(cls)],
                }
                results_list.append(result_dict)

        if not results_list:
            return jsonify({"message": "No objects detected"}), 200

        response = {"results": results_list}

        return jsonify(response)
    except Exception as e:
        print(f"Error: {e}")  # Print the error to the console for debugging
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
