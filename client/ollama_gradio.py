import gradio as gr
import requests
import base64
import os

# Define the Ollama server URL
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2-vision"  # Replace with the actual model name if different

def encode_image(image):
    """Encode the image to Base64 format."""
    try:
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return None  # Return None if encoding fails

def send_to_ollama(message, image):
    """Send a message and image to the Ollama server."""
    try:
        # Construct the payload with the message
        payload = {
            "model": MODEL_NAME,
            "prompt": message
        }

        # Add the image only if it exists
        if image and os.path.exists(image):
            image_b64 = encode_image(image)
            if image_b64:
                payload["image"] = image_b64
            else:
                return "Error: Unable to process the image."

        # Send the request to the Ollama server
        response = requests.post(OLLAMA_URL, json=payload)

        # Debug the raw response
        print(f"Response content: {response.text}")

        # Check the server response
        if response.status_code == 200:
            try:
                data = response.json()
                return data.get("text", "No response text found.")
            except ValueError as e:
                return f"Error: Server returned invalid JSON. Response: {response.text}"
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"


# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Ollama Vision Test Client")
        gr.Markdown("Send a text prompt and optionally an image to the Ollama server using Llama 3.2 Vision.")

        with gr.Row():
            input_message = gr.Textbox(label="Message", placeholder="Enter your prompt here...")
            input_image = gr.Image(label="Upload Image (Optional)", type="filepath")
            submit_button = gr.Button("Send")

        output_result = gr.Textbox(label="Server Response")

        # Connect the button to the function
        submit_button.click(send_to_ollama, inputs=[input_message, input_image], outputs=output_result)

    return demo

# Run the Gradio app
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
