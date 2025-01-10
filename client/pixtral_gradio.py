import gradio as gr
import requests
import base64
import os
from mistralai import Mistral

# Mistral API Configuration
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "Bearer Dxew4p4TzJ4aiUKTusUtwfxjf9xOE4gV"  # Replace with your actual API key
MODEL_NAME = "pixtral-large-latest"

def send_text_and_image_url(prompt, image_url):
    """Send a prompt and image URL to the Mistral API."""
    headers = {
        "Authorization": MISTRAL_API_KEY,
        "Content-Type": "application/json",
    }

    # Construct the payload
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url}
                ],
            }
        ],
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response text found.")
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except ValueError:
        return "Error: Invalid JSON response from the server."

# def send_text_and_base64_image(prompt, image_path):
#     """Send a prompt and Base64-encoded image to the Mistral API."""
#     headers = {
#         "Authorization": MISTRAL_API_KEY,
#         "Content-Type": "application/json",
#     }

#     # Encode the image in Base64
#     def encode_image(image_path):
#         try:
#             with open(image_path, "rb") as img_file:
#                 return base64.b64encode(img_file.read()).decode("utf-8")
#         except Exception as e:
#             print(f"Error encoding image: {e}")
#             return None

#     image_b64 = encode_image(image_path)
#     if not image_b64:
#         return "Error: Unable to process the image."

#     # Construct the payload
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image", "image": image_b64}
#                 ],
#             }
#         ],
#     }

#     try:
#         response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         return data.get("choices", [{}])[0].get("message", {}).get("content", "No response text found.")
#     except requests.exceptions.RequestException as e:
#         return f"Request error: {e}"
#     except ValueError:
#         return "Error: Invalid JSON response from the server."


def send_text_and_base64_image(prompt, image_path):
    """Send a prompt and Base64-encoded image to the Mistral API using the Mistral SDK."""
    try:
        # Encode the image to Base64
        def encode_image(image_path):
            """Encode the image to Base64."""
            try:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            except FileNotFoundError:
                print(f"Error: The file {image_path} was not found.")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

        base64_image = encode_image(image_path)
        if not base64_image:
            return "Error: Unable to encode image to Base64."

        # Retrieve the API key from environment variables
        api_key = MISTRAL_API_KEY
        if not api_key:
            return "Error: MISTRAL_API_KEY environment variable not set."

        # Initialize the Mistral client
        client = Mistral(api_key=api_key)

        # Define the messages for the chat
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"  # Correct field for Base64
                    }
                ]
            }
        ]

        # Get the chat response
        chat_response = client.chat.complete(
            model=MODEL_NAME,
            messages=messages
        )

        # Return the content of the response
        return chat_response.choices[0].message.content

    except Exception as e:
        return f"Exception occurred: {e}"



# Gradio Interface
def gradio_interface():
    """Create the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Mistral Pixtral Large Test Client")
        gr.Markdown("Send a text prompt and optionally an image URL or Base64 image to the Mistral API.")

        with gr.Row():
            input_message = gr.Textbox(label="Message", placeholder="Enter your prompt here...")
            input_image_url = gr.Textbox(label="Image URL", placeholder="Enter an image URL here...")
            input_image_file = gr.Image(label="Upload Image", type="filepath")
            submit_url_button = gr.Button("Send with Image URL")
            submit_file_button = gr.Button("Send with Base64 Image")

        output_result = gr.Textbox(label="API Response")

        # Connect the buttons to their respective functions
        submit_url_button.click(send_text_and_image_url, inputs=[input_message, input_image_url], outputs=output_result)
        submit_file_button.click(send_text_and_base64_image, inputs=[input_message, input_image_file], outputs=output_result)

    return demo

# Run the Gradio app
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
