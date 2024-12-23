from fastapi import FastAPI, UploadFile, Form, Request
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Middleware to log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

# Load Qwen2-VL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").to(device)

# Update processor for larger token limits
processor.tokenizer.model_max_length = 4096  # Adjust based on requirements

# Function to resize images to a larger resolution
def preprocess_image(image, target_size=(640, 360)):  # Adjust target size as needed
    return image.resize(target_size)

@app.post("/predict")
async def predict(file: UploadFile, message: str = Form(...)):
    try:
        # Log the received file and message
        logging.info(f"Received file: {file.filename}, message: {message}")

        # Read and preprocess the image
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            logging.info(f"Original Image size: {image.size}, format: {image.format}")
        except Exception as e:
            logging.error(f"Invalid image file: {str(e)}")
            return {"status": "error", "message": "Invalid image format"}

        # Resize the image using preprocess function
        image = preprocess_image(image, target_size=(640, 360))

        # Update model parameters for 1280x720
        processor.tokenizer.model_max_length = 4096  # Max tokens (image + text)
        instruction = f"{message.strip()} Predict [x, y] coordinates from this image."

        # Prepare inputs for the model
        inputs = processor(
            text=[instruction],
            images=[image],
            padding="longest",
            truncation=True,
            max_length=500,  # Limit text tokens to 500
            return_tensors="pt",
        ).to(device)

        # Run inference
        generated_ids = model.generate(**inputs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the output
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Log and return the result
        logging.info(f"Inference result: {output_text}")
        return {"status": "success", "response": output_text}

    except Exception as e:
        # Log any errors during processing
        logging.error(f"Error during processing: {str(e)}")
        return {"status": "error", "message": str(e)}
