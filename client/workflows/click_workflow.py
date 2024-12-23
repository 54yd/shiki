import pyautogui

def click_on_coordinates(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()
