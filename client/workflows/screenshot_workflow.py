from pyautogui import screenshot

def capture_screenshot():
    screenshot("current_screenshot.png")
    return "current_screenshot.png"