import client.test_gradio as gr
import requests

# Define server URL
SERVER_URL = "http://localhost:11434/api/generate"

# Function to send a message and an image to the Qwen2-VL server
def send_message_and_image(message, image):
    # Save the uploaded image temporarily
    temp_path = "temp_image.png"
    image.save(temp_path)

    # Prepare the request
    with open(temp_path, "rb") as img_file:
        response = requests.post(
            SERVER_URL,
            files={"file": img_file},
            data={"message": message}  # Send the message as form-data
        )

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        return f"Server Response: {result}"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2-VL Test Client")
    gr.Markdown("Send a text message and an image to the Qwen2-VL server.")

    with gr.Row():
        input_message = gr.Textbox(label="Message", placeholder="Enter your message here...")
        input_image = gr.Image(label="Upload Image", type="pil")
        submit_button = gr.Button("Send")

    output_result = gr.Textbox(label="Server Response")

    # Connect the button to the function
    submit_button.click(send_message_and_image, inputs=[input_message, input_image], outputs=output_result)

# Launch the Gradio app
demo.launch()
