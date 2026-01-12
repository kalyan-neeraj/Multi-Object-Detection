def preprocess_image(image):
    # Function to preprocess the image before passing it to the model
    # This can include resizing, normalization, etc.
    return image

def format_detections(detections):
    # Function to format the detections into a desired structure
    formatted = []
    for detection in detections:
        formatted.append({
            'label': detection['name'],
            'confidence': detection['confidence'],
            'boundingBox': {
                'x': detection['xmin'],
                'y': detection['ymin'],
                'width': detection['xmax'] - detection['xmin'],
                'height': detection['ymax'] - detection['ymin']
            }
        })
    return formatted