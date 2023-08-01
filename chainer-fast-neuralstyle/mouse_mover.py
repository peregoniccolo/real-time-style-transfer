import time
import pyautogui

time.sleep(2)
    
while(True):
    print('moving')
    pos = pyautogui.position()
    pyautogui.moveTo(pos[0] + 50, pos[1], duration = 1)
    time.sleep(300)

    print('clicking')
    pyautogui.click()
    time.sleep(300)
    
    print('moving back')
    pyautogui.moveTo(pos[0], pos[1], duration = 1)
    time.sleep(300)