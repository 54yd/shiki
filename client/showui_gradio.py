import os
import torch
import gradio as gr
import numpy as np
from qwen_vl_utils import process_vision_info
from PIL import Image
from datetime import datetime
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
MODEL_ID = "showlab/ShowUI-2B"               # HF repo for ShowUI
LOCAL_DIR = "./showui-2b"                    # Local path to store downloaded weights
PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"   # Qwen2-VL processor ID
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28
MAX_NEW_TOKENS = 128

# --------------------------------------------------------------------------------
# 1) Utility: Save Uploaded Image to Disk
# --------------------------------------------------------------------------------
def array_to_image_path(image_array: np.ndarray) -> str:
    """
    Convert a NumPy array to a PIL Image, save it to disk, 
    and return the absolute path. 
    """
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")

    img = Image.fromarray(np.uint8(image_array))  # ensure correct data type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)

    return os.path.abspath(filename)

# --------------------------------------------------------------------------------
# 2) Ensures the Model is Downloaded
# --------------------------------------------------------------------------------
def ensure_model_downloaded(model_id: str, local_dir: str) -> None:
    """
    If the local_dir is empty, use snapshot_download from Hugging Face Hub.
    Otherwise, assume it's already present.
    """
    if not os.path.exists(local_dir) or len(os.listdir(local_dir)) == 0:
        print(f"Downloading model [{model_id}] to [{local_dir}]...")
        snapshot_download(repo_id=model_id, local_dir=local_dir)
    else:
        print(f"Model folder [{local_dir}] already exists. Skipping download.")

# --------------------------------------------------------------------------------
# 3) Load Model & Processor
# --------------------------------------------------------------------------------
def load_model_and_processor():
    """
    Download (if needed) and load the ShowUI model and its Qwen2-VL processor.
    """
    ensure_model_downloaded(MODEL_ID, LOCAL_DIR)

    # Load model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        LOCAL_DIR,
        torch_dtype=torch.bfloat16,  # or float16/float32 depending on hardware
        device_map="cpu",          # requires accelerate
    ) 
    model.eval()
    #model = model.to("cpu")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        PROCESSOR_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )

    return model, processor

# --------------------------------------------------------------------------------
# 4) Prediction Function (Gradio)
# --------------------------------------------------------------------------------
def predict(image_array: np.ndarray, text: str) -> str:
    """
    Takes an uploaded NumPy image + text query,
    saves the image to disk, then runs inference with Qwen2-VL.
    """
    _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

    global model
    model = model.to("cpu")
    

    if image_array is None:
        return "Error: No image provided."

    # -- Convert array to local file --
    image_path = array_to_image_path(image_array)

    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SYSTEM},
                {"type": "image", "image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": text}
            ],
        }
    ]


    print(f"image path : {image_path}")
    # -- Open the saved image as PIL --
    pil_image = Image.open(image_path).convert("RGB")
    print(f"pil image : {pil_image}")
    
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    # -- Preprocess --
    inputs = processor(
        text=[text],       # wrap text in a list to match batch dimension
        images=image_inputs,  # single PIL image
        return_tensors="pt",
        padding=True,
    )

    inputs = inputs.to("cpu")

    # -- Generate response --

       # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
   
    response = output_text
    return response

# --------------------------------------------------------------------------------
# 5) Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 5a) Load model & processor once
    model, processor = load_model_and_processor()

    # 5b) Define Gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Upload Image"),
            gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text")
        ],
        outputs=gr.Textbox(label="Model Response"),
        title="ShowUI (Qwen2-VL) Gradio Integration",
        description="Uploads a NumPy image, saves it to disk, and runs Qwen2-VL (ShowUI) inference."
    )

    # 5c) Launch
    interface.launch(server_name="0.0.0.0", server_port=7860 )#share=True)
    #interface.launch(server_name="0.0.0.0", server_port=7860 ,share=True)




# import os
# import torch
# import gradio as gr
# from PIL import Image
# from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
# import base64
# # --------------------------------------------------------------------------------
# # Constants & Config
# # --------------------------------------------------------------------------------
# MODEL_ID = "showlab/ShowUI-2B"          # HF repo ID for the ShowUI weights
# LOCAL_DIR = "./showui-2b"              # Local directory to store downloaded model
# PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"  # If your processor is in another repo
# MIN_PIXELS = 256 * 28 * 28
# MAX_PIXELS = 1344 * 28 * 28
# MAX_NEW_TOKENS = 128

# # --------------------------------------------------------------------------------
# # Download Model if not present locally
# # --------------------------------------------------------------------------------
# def ensure_model_downloaded(model_id: str, local_dir: str) -> None:
#     """
#     Ensures the model weights are downloaded locally via snapshot_download().
#     Skips download if the folder already exists and is non-empty.
#     """
#     if not os.path.exists(local_dir) or len(os.listdir(local_dir)) == 0:
#         print(f"Downloading model [{model_id}] to [{local_dir}]...")
#         snapshot_download(repo_id=model_id, local_dir=local_dir)
#     else:
#         print(f"Model folder [{local_dir}] already exists. Skipping download.")


# # --------------------------------------------------------------------------------
# # Resize Image
# # --------------------------------------------------------------------------------

# def resize_image(image: Image.Image, max_dim=448):
#     # e.g., keep aspect ratio, but max dimension is 448
#     w, h = image.size
#     scaling_factor = max_dim / max(w, h)
#     if scaling_factor < 1:
#         new_w = int(w * scaling_factor)
#         new_h = int(h * scaling_factor)
#         image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
#     return image
# # --------------------------------------------------------------------------------
# # Load Model & Processor
# # --------------------------------------------------------------------------------
# def load_model_and_processor(
#     model_id: str = MODEL_ID,
#     processor_id: str = PROCESSOR_ID,
#     local_dir: str = LOCAL_DIR
# ):
#     """
#     Download (if needed) and load the model and its processor.
#     Uses device_map='auto' for automatic GPU/CPU placement.
#     """
#     ensure_model_downloaded(model_id, local_dir)

#     print("Loading model...")
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         local_dir,
#         torch_dtype=torch.bfloat16,   # Use BF16 if supported
#         device_map=None,           # Auto-assign layers to GPU/CPU
#     )
#     model.eval()

#     print("Loading processor...")
#     processor = AutoProcessor.from_pretrained(
#         processor_id,
#         # min_pixels=MIN_PIXELS,
#         # max_pixels=MAX_PIXELS,
#     )
#     return model, processor

# # --------------------------------------------------------------------------------
# # Gradio Prediction Function
# # --------------------------------------------------------------------------------
# def predict(image: Image.Image, text: str):
#     """
#     Predict function used by Gradio.  
#     1) Preprocess image + text with the Qwen processor.  
#     2) Generate text using Qwen2VLForConditionalGeneration.  
#     3) Return the decoded response.  
#     """
#     if image is None:
#         return "Error: No image provided."

#     image_resized = resize_image(image, max_dim=448)

#     # Preprocess the inputs
#     inputs = processor(
#         text=[text],
#         images=image_resized,
#         return_tensors="pt",
#         padding=True,
#     ).to("cpu")  # make sure to move data to the same device

    
#     outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
#     response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

#     return response

# # --------------------------------------------------------------------------------
# # Main
# # --------------------------------------------------------------------------------
# if __name__ == "__main__":

#     # 1) Load everything
#     model, processor = load_model_and_processor()

#     # 2) Set up Gradio interface
#     interface = gr.Interface(
#         fn=predict,
#         inputs=[
#             gr.Image(type="pil", label="Input Image"),
#             gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text")
#         ],
#         outputs=gr.Textbox(label="Model Response"),
#         title="ShowUI Gradio Integration",
#         description=(
#             "Interact with the ShowUI model (Qwen-based) by providing an image and a text input. "
#             "The model will generate a textual response."
#         ),
#     )

#     # 3) Launch
#     interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
