import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageDraw
import colorsys

def get_color(confidence):
    # Map confidence score to a color. This is just an example, feel free to adjust.
    # Here, high confidence detections are green and low are red.
    return int(255 * (1 - confidence)), int(255 * confidence), 0

# Disable eager execution to work with hub.Module in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# Clear default graph (optional but can avoid variable reuse issues)
tf.compat.v1.reset_default_graph()

# Read and preprocess image
image_path = '/home/saashiv/d/emerRP/data/image_data/00003e7c13af160c.jpg'
image = Image.open(image_path)
image_np = np.array(image) / 255.0  # Normalize to [0,1]
image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

# Convert to tensor
image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)

print('loading model')
# Load model
model = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# Run detection
detector_output = model(image_tensor, as_dict=True)

# Fetch results
with tf.compat.v1.Session() as sess:
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    output_dict = sess.run(detector_output)
    class_names = output_dict['detection_class_entities']

# Assuming the image shape is in the form (height, width, channels)
image_width, image_height = image.size

# Determine the number of detections
num_detections = len(output_dict['detection_scores'])

# Create an empty list to store bounding box information
bounding_boxes = []

print('looping')
# Loop through the list of detections
for i in range(num_detections):
    # Convert normalized coordinates to pixel coordinates
    y_min, x_min, y_max, x_max = output_dict['detection_boxes'][i]
    x_min = int(x_min * image_width)
    x_max = int(x_max * image_width)
    y_min = int(y_min * image_height)
    y_max = int(y_max * image_height)
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    # Get the label and confidence score
    label = output_dict['detection_class_entities'][i]
    confidence = output_dict['detection_scores'][i]
    
    # Get color based on the confidence score
    color = get_color(confidence)

    if confidence>0.5:
        # Append this bounding box to the list
        bounding_boxes.append({
            'label': label,
            'confidence': confidence,
            'x': x_min,
            'y': y_min,
            'width': width,
            'height': height,
            'color':color
    })

# Now, bounding_boxes contains your boxes in the format you described

# Load an image from file
image = cv2.imread(image_path)

print('looping 2')
# Loop through the list of bounding boxes and draw them on the image
for box in bounding_boxes:
    # Extract the details for each box
    label = box['label']
    confidence = box['confidence']
    x, y, width, height,color = box['x'], box['y'], box['width'], box['height'],box['color']
    
    # Draw a rectangle around the object
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    
    # Add a label and confidence score
    text = f"{label} {confidence * 100:.2f}%"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

print('displaying')
# Display the image with the bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# export CUDA_VISIBLE_DEVICES=-1