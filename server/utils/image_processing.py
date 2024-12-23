from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image
