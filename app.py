import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision import models
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import requests
from io import BytesIO

app = Flask(__name__)

# Load detection model
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load ImageNet classification model
classification_model = models.resnet50(pretrained=True)
classification_model.eval()


# Load image captioning model and tokenizer
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(captioning_device)

# Define ImageNet transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Map ImageNet class indices to class labels
imagenet_labels = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()

# Add an "uploads" folder to store processed images
UPLOADS_FOLDER = 'uploads'
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
os.makedirs(os.path.join(app.instance_path, UPLOADS_FOLDER), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADS_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # Get the uploaded images
        image_files = request.files.getlist('images')

        # Process each image
        processed_images = []
        for image_file in image_files:
            # Save the image
            image_path = os.path.join(app.config['UPLOADS_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform detection
            detection_results = detect_objects(image_path)

            # Perform classification
            classification_result = classify_image(image_path)

            # Perform image captioning
            caption_results = predict_step(image_path)

            # Create output image with detection, classification, and captioning
            output_image_path = process_and_save_image(image_path, detection_results, classification_result, caption_results)

            processed_images.append(output_image_path)

        # Return the processed images filenames
        return render_template('result.html', processed_images=processed_images)

def detect_objects(image_path, confidence_threshold=0.1):
    img = Image.open(image_path)
    results = detection_model(img, size=640)  # Adjust the size parameter as needed

    # Filter results based on confidence threshold
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > confidence_threshold]
    print(results.xyxy[0])
    return filtered_results.cpu().numpy()

def classify_image(image_path):
    img = Image.open(image_path)
    img = preprocess(img)
    img = img.unsqueeze(0)

    # Perform classification
    with torch.no_grad():
        output = classification_model(img)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)

    # Get the predicted class label
    predicted_label = imagenet_labels[predicted_idx.item()]

    return predicted_label

def predict_step(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images = [i_image]

    pixel_values = captioning_feature_extractor(images=images, return_tensors="pt", return_attention_mask=True).pixel_values
    pixel_values = pixel_values.to(captioning_device)

    output_ids = captioning_model.generate(pixel_values)

    preds = captioning_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def process_and_save_image(original_image_path, detection_results, classification_results, caption_results):
    # Open the original image
    img = Image.open(original_image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw bounding boxes and add captions based on detection, classification, and captioning results
    for result, classification, caption in zip(detection_results, classification_results, caption_results):
        box = result[:4]

        # Ensure the box has the correct format (x_min, y_min, x_max, y_max)
        if len(box) == 4:
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]

            # Draw bounding box
            draw.rectangle(box, outline="red", width=2)

        # Calculate text position above the bounding box for classification
        text_position_classification = (box[0] + (box[2] - box[0]) / 2, box[1] - 10)
        draw.text(text_position_classification, f'Class: {classification}', fill="black", anchor="ma")

        # Calculate text position below the bounding box for captioning
        text_position_caption = (box[0] + (box[2] - box[0]) / 2, box[3] + 10)
        font = ImageFont.truetype("arial.ttf", 50)  # Adjust the font size as needed
        draw.text(text_position_caption, f'Caption: {caption}', fill="black", font=font, anchor="ma")

    # Save the processed image in the "uploads" folder
    output_image_path = os.path.join(app.config['UPLOADS_FOLDER'], 'processed_' + os.path.basename(original_image_path))
    img.save(output_image_path)

    return os.path.basename(output_image_path)

if __name__ == '__main__':
    app.run(debug=True)
