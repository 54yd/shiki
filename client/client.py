from langchain.graph import Graph
from langchain.graph.nodes import FunctionNode
import requests
import pyautogui
import time

SERVER_URL = "http://localhost:11434"

def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    return "screenshot.png"

def send_to_server(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(f"{SERVER_URL}/predict", files={"file": f})
    result = response.json()
    return result["coordinates"]

def click_coordinates(coordinates):
    x, y = coordinates
    pyautogui.moveTo(x, y)
    pyautogui.click()

def wait_between_steps():
    time.sleep(2)

graph = Graph()
screenshot_node = FunctionNode("Take Screenshot", take_screenshot)
server_node = FunctionNode("Send to Server", send_to_server)
click_node = FunctionNode("Click Coordinates", click_coordinates)
wait_node = FunctionNode("Wait", wait_between_steps)

graph.connect(screenshot_node, server_node)
graph.connect(server_node, click_node)
graph.connect(click_node, wait_node)
graph.connect(wait_node, screenshot_node)

if __name__ == "__main__":
    while True:
        graph.run()
