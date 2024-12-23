import requests

def send_request(image_path, server_url):
    with open(image_path, "rb") as f:
        response = requests.post(server_url, files={"file": f})
    return response.json()
