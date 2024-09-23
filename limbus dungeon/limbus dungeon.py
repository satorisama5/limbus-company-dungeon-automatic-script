import os
import time
import pyautogui
import pygetwindow as gw
import cv2
import numpy as np


def is_window_open(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    return len(windows) > 0


# 打开边狱巴士
shortcut_path = r"C:\Users\zsy14\Desktop\Limbus Company.url"



# 使用 os.startfile 打开快捷方式
os.startfile(shortcut_path)

window_title = "LimbusCompany"
while True:
    if is_window_open(window_title):
        time.sleep(10)
        pyautogui.click(clicks=5)
        break
    else:
        time.sleep(1)  # 每秒检查一次


# 定义找图片函数，传入路径
print("已经成功进入游戏。")
def find_and_click_icons(template_paths):

    while template_paths:
        # 捕获屏幕截图并转换为灰度图像
        screen = np.array(pyautogui.screenshot())
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        template_path = template_paths[0]  # 获取当前模板路径
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_w, template_h = template.shape[::-1]

        # 执行模板匹配
        result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 设定一个匹配阈值
        if max_val >= 0.7:
            # 计算图标中心位置
            top_left = max_loc
            center_x = top_left[0] + template_w // 2
            center_y = top_left[1] + template_h // 2

            # 移动鼠标到图标位置并点击
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            pyautogui.click(clicks=2)

            template_paths.pop(0)  # 移除已找到的模板路径
            time.sleep(0.5)  # 等待一会儿，确保点击效果可见
        else:
            time.sleep(0.2)  # 继续检查当前模板

    # 执行后续代码



# 使用模板图标的路径
template_paths = [
    r'E:\python project\limbus dungeon\drive.png', r'E:\python project\limbus dungeon\mirrordungeons.png',
    r'E:\python project\limbus dungeon\simulation.png', r'E:\python project\limbus dungeon\enter.png',
    r'E:\python project\limbus dungeon\confirm.png', r'E:\python project\limbus dungeon\bleed.png',
    r'E:\python project\limbus dungeon\wound.png', r'E:\python project\limbus dungeon\plu.png',
    r'E:\python project\limbus dungeon\ego at begin.png'
]
find_and_click_icons(template_paths)
print("所有图标已检查完毕。")
# 反斜杠最好用r加前面
time.sleep(1)
for _ in range(2):   # _表示占位符表示我们不在乎range生成的具体值(0,1,2)，3是指随机生成0 1 2三个数也就是三次
    pyautogui.press('enter')
    time.sleep(0.5)
time.sleep(0.5)
template_paths20 = [r'E:\python project\limbus dungeon\bleed1.png']
find_and_click_icons(template_paths20)
time.sleep(0.5)
pyautogui.press('enter')
time.sleep(0.5)
#选卡部分
# 读取目标图片并获取其尺寸

time.sleep(5)
#判断卡是否出现
# 读取目标图像和检查图像列表
target_image = cv2.imread('refresh.png', cv2.IMREAD_GRAYSCALE)
target_height, target_width = target_image.shape

check_images = [
    cv2.imread('unloving.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('violet.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('burning.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('flood.png',cv2.IMREAD_GRAYSCALE),
    cv2.imread('sloth.png',cv2.IMREAD_GRAYSCALE),
    cv2.imread('slicer.png',cv2.IMREAD_GRAYSCALE),
    cv2.imread('flesh.png',cv2.IMREAD_GRAYSCALE),
    cv2.imread('timekilling.png',cv2.IMREAD_GRAYSCALE),
    cv2.imread('murder.png',cv2.IMREAD_GRAYSCALE)
]

check_threshold = 0.5

drag_start = (265, 405)
drag_end = (265, 800)

starlight_image = cv2.imread('starlight.png', cv2.IMREAD_GRAYSCALE)
starlight_threshold = 0.7

enter_image = cv2.imread('enter1.png', cv2.IMREAD_GRAYSCALE)
enter_threshold = 0.7

click_coords = [(1445, 168), (1445, 591), (1445, 1019)]
click_interval = 0.5  # 点击间隔时间（秒）

template_paths_party = cv2.imread('party.png', cv2.IMREAD_GRAYSCALE)
party_threshold = 0.7

winrate_image = cv2.imread('winrate.png', cv2.IMREAD_GRAYSCALE)
winrate_threshold = 0.7

skip_image = cv2.imread('skip.png', cv2.IMREAD_GRAYSCALE)
skip_threshold = 0.7

veryhigh_image = cv2.imread('veryhigh.png', cv2.IMREAD_GRAYSCALE)
veryhigh_threshold = 0.7

high_image = cv2.imread('high.png', cv2.IMREAD_GRAYSCALE)
high_threshold = 0.7

continue_image = cv2.imread('continue.png', cv2.IMREAD_GRAYSCALE)
continue_threshold = 0.7

confirm_image = cv2.imread('confirm.png', cv2.IMREAD_GRAYSCALE)
confirm_threshold = 0.7

proceed_image = cv2.imread('proceed.png', cv2.IMREAD_GRAYSCALE)
proceed_threshold = 0.7

normal_image = cv2.imread('normal.png', cv2.IMREAD_GRAYSCALE)
normal_threshold = 0.7

product_image = cv2.imread('product.png', cv2.IMREAD_GRAYSCALE)
product_threshold = 0.7

leave_image = cv2.imread('leave.png', cv2.IMREAD_GRAYSCALE)
leave_threshold = 0.7

commence_image = cv2.imread('commence.png', cv2.IMREAD_GRAYSCALE)
commence_threshold = 0.7

encounter_image = cv2.imread('encounter.png', cv2.IMREAD_GRAYSCALE)
encounter_threshold = 0.7

egogift_image = cv2.imread('egogift.png', cv2.IMREAD_GRAYSCALE)
egogift_threshold = 0.7

refresh_image = cv2.imread('refresh.png', cv2.IMREAD_GRAYSCALE)
refresh_threshold = 0.7

exp_image = cv2.imread('exp.png', cv2.IMREAD_GRAYSCALE)
exp_threshold = 0.7

commencebattle_image = cv2.imread('commencebattle.png', cv2.IMREAD_GRAYSCALE)
commencebattle_threshold = 0.7

confirm3_image = cv2.imread('confirm3.png', cv2.IMREAD_GRAYSCALE)
confirm3_threshold = 0.7

low_image = cv2.imread('low.png', cv2.IMREAD_GRAYSCALE)
low_threshold = 0.7

verylow_image = cv2.imread('verylow.png', cv2.IMREAD_GRAYSCALE)
verylow_threshold = 0.7

def perform_drag_action(start, end):
    print(f"执行鼠标拖动操作从 {start} 到 {end}！")
    pyautogui.moveTo(start)
    pyautogui.mouseDown()
    pyautogui.moveTo(end, duration=2)
    pyautogui.mouseUp()
    print("拖动操作完成")


def image_in_rectangle(frame, start, end, images, threshold):
    x1, y1 = start
    x2, y2 = end
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    region = frame[y_min:y_max, x_min:x_max]

    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

    for i, image in enumerate(images):
        # 对于每张图像，进行模板匹配
        res = cv2.matchTemplate(gray_region, image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        # 打印当前图像的索引和相似度值
        print(f"Checking image {i}...")
        print(f"Image {i}: Similarity = {max_val}")

        # 你可以在这里添加更多的解析或调试语句
        # 例如，记录图像的大小，类型等
        print(f"Image {i} - Size: {image.shape}")

        # 如果相似度超过阈值，返回 True
        if max_val >= threshold:
            return True

    return False

def find_and_click_image(image_path, threshold=0.7):
    # 截取整个屏幕
    screen = np.array(pyautogui.screenshot())
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # 读取模板图像并转换为灰度图
    template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Image at path {image_path} not found or invalid.")

    template_w, template_h = template.shape[::-1]

    # 使用模板匹配查找图像
    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 如果匹配度高于阈值，则点击
    if max_val >= threshold:
        top_left = max_loc
        center_x = top_left[0] + template_w // 2
        center_y = top_left[1] + template_h // 2
        pyautogui.moveTo(center_x, center_y, duration=0.2)
        pyautogui.click()
        print(f"Image found and clicked at position: ({center_x}, {center_y})")
        return True

    print("Image not found.")
    return False


def perform_click_action(coords):
        for x, y in coords:
            pyautogui.click(x, y)
            print(f"点击坐标: {x}, {y}")
            time.sleep(click_interval)
def image_exists(frame, image, threshold):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        res1 = cv2.matchTemplate(gray_frame, image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res1)
        return max_val >= threshold

while True:
    screenshot1 = pyautogui.screenshot()
    frame1 = np.array(screenshot1)

    if image_exists(frame1, starlight_image, starlight_threshold) and not image_exists(frame1, product_image, product_threshold):
        time.sleep(1)
        print("检测到 starlight.png，开始点击操作")
        while image_exists(frame1, starlight_image, starlight_threshold):
            perform_click_action(click_coords)
            # 检查是否出现 enter1.png
            screenshot1 = pyautogui.screenshot()
            frame1 = np.array(screenshot1)
            if image_exists(frame1, enter_image, enter_threshold):
                print("检测到 enter1.png，模拟按键 'enter'")
                pyautogui.press('enter')
            # 休眠时间，以防过于频繁的操作
            time.sleep(2)
            # 更新屏幕截图
            screenshot1 = pyautogui.screenshot()
            frame1 = np.array(screenshot1)
    if image_exists(frame1, product_image, product_threshold)  and image_exists(frame1, starlight_image, starlight_threshold):
        template_paths6 = [r'E:\python project\limbus dungeon\leave.png']
        time.sleep(0.5)
        find_and_click_icons(template_paths6)
        time.sleep(0.5)
        pyautogui.press('enter')
    if not image_exists(frame1, starlight_image, starlight_threshold) and  image_exists(frame1, refresh_image, refresh_threshold):
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangle_x2 = drag_start[0] + 335
        rectangle_y2 = drag_end[1]
        rectangle_start = (drag_start[0], drag_start[1])
        rectangle_end = (rectangle_x2, rectangle_y2)

        res = cv2.matchTemplate(frame, target_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        print(f"max_val: {max_val}")

        if max_val >= 0.8:
            if image_in_rectangle(frame, rectangle_start, rectangle_end, check_images, check_threshold):
                drag_start = (drag_start[0] + 425, drag_start[1])
                drag_end = (drag_end[0] + 425, drag_end[1])
                print(f"Updated drag_start: {drag_start}, drag_end: {drag_end}")
            time.sleep(2)
            perform_drag_action(drag_start, drag_end)

        time.sleep(0.2)
    if not image_exists(frame1, starlight_image, starlight_threshold) and not image_exists(frame1, refresh_image, refresh_threshold):
                # 如果出现战斗，则进行回车战斗的操作
            if image_exists(frame1, template_paths_party, party_threshold):
                time.sleep(0.5)
                pyautogui.press('enter')
                print("进入战斗成功")
            #此处再次进行模拟p键来进行战斗
                time.sleep(3)
            if image_exists(frame1, winrate_image, winrate_threshold):
                print("成功发现胜率")
                time.sleep(0.5)
                pyautogui.click(278,823)
                time.sleep(0.5)
                pyautogui.press('p')
                pyautogui.press('enter')
            # 如果没有检测到 starlight.png，更新屏幕截图并等待
            if image_exists(frame1, skip_image, skip_threshold):
                print("发现跳过")
                time.sleep(0.5)
                pyautogui.click(1198,638)
                time.sleep(0.5)
                pyautogui.click(1198,400)
                time.sleep(1)
                pyautogui.click(1600,480)
            if image_exists(frame1, veryhigh_image, veryhigh_threshold) and image_exists(frame1, skip_image, skip_threshold) and not image_exists(frame1, commence_image, commence_threshold) and not image_exists(frame1, continue_image, continue_threshold):
                print("进行非常高的判断")
                template_paths1 =r'E:\python project\limbus dungeon\veryhigh.png'
                time.sleep(0.5)
                find_and_click_image(template_paths1)
                print("已经点击成功，退出该if")
            if image_exists(frame1, high_image, high_threshold) and not image_exists(frame1, veryhigh_image, veryhigh_threshold) and image_exists(frame1, skip_image, skip_threshold) and not image_exists(frame1, commence_image, commence_threshold) and not image_exists(frame1, continue_image, continue_threshold):
                template_paths2=r'E:\python project\limbus dungeon\high.png'
                time.sleep(0.5)
                find_and_click_image(template_paths2)
            if not image_exists(frame1, veryhigh_image, veryhigh_threshold) and not image_exists(frame1, high_image, high_threshold) and image_exists(frame1, skip_image, skip_threshold) and image_exists(frame1, normal_image, normal_threshold) and not image_exists(frame1, commence_image, commence_threshold) and not image_exists(frame1, continue_image, continue_threshold):
                template_paths3=r'E:\python project\limbus dungeon\normal.png'
                time.sleep(0.5)
                find_and_click_image(template_paths3)
            if not image_exists(frame1, veryhigh_image, veryhigh_threshold) and not image_exists(frame1, high_image,high_threshold) and image_exists(frame1, skip_image, skip_threshold) and not image_exists(frame1, normal_image,normal_threshold) and not image_exists(frame1, commence_image, commence_threshold) and not image_exists(frame1, continue_image,continue_threshold) and image_exists(frame1,low_image,low_threshold):
                template_paths11 = r'E:\python project\limbus dungeon\low.png'
                time.sleep(0.5)
                find_and_click_image(template_paths11)
            if not image_exists(frame1, veryhigh_image, veryhigh_threshold) and not image_exists(frame1, high_image,high_threshold) and image_exists(frame1, skip_image, skip_threshold) and not image_exists(frame1, normal_image,normal_threshold) and not image_exists(frame1, commence_image, commence_threshold) and not image_exists(frame1, continue_image,continue_threshold) and not image_exists(frame1, low_image, low_threshold) and image_exists(frame1, verylow_image,verylow_threshold):
                template_paths12 = r'E:\python project\limbus dungeon\verylow.png'
                time.sleep(0.5)
                find_and_click_image(template_paths12)
            if image_exists(frame1, continue_image, continue_threshold):
                template_paths4=[r'E:\python project\limbus dungeon\continue.png']
                time.sleep(0.5)
                find_and_click_icons(template_paths4)
            if image_exists(frame1, confirm_image, confirm_threshold):
                time.sleep(0.5)
                pyautogui.press('enter')
            if image_exists(frame1, proceed_image, proceed_threshold):
                template_paths5= [r'E:\python project\limbus dungeon\proceed.png']
                time.sleep(0.5)
                find_and_click_icons(template_paths5)
            if image_exists(frame1, commence_image, commence_threshold):
                template_paths6=[r'E:\python project\limbus dungeon\commence.png']
                time.sleep(0.5)
                find_and_click_icons(template_paths6)
            if image_exists(frame1, encounter_image, encounter_threshold):
                pyautogui.press('enter')
                time.sleep(0.5)
                pyautogui.press('enter')
            if image_exists(frame1, egogift_image,egogift_threshold):
                template_paths7=[r'E:\python project\limbus dungeon\egogift.png']
                time.sleep(0.5)
                find_and_click_icons(template_paths7)
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(1)
                if image_exists(frame1, confirm3_image, confirm3_threshold):
                    template_paths10 =[r'E:\python project\limbus dungeon\confirm3.png']
                    time.sleep(0.5)
                    find_and_click_icons(template_paths10)
                time.sleep(0.5)
                pyautogui.press('enter')
            if image_exists(frame1, commencebattle_image, commencebattle_threshold):
                template_paths9=[r'E:\python project\limbus dungeon\commencebattle.png']
                time.sleep(0.5)
                find_and_click_icons(template_paths9)
    time.sleep(1)
    if image_exists(frame1, exp_image, exp_threshold):
        break

time.sleep(2)
template_paths8=[r'E:\python project\limbus\confirm1.png',r'E:\python project\limbus\claimrewards.png',
                r'E:\python project\limbus\claim.png', r'E:\python project\limbus\confirm2.png',
                r'E:\python project\limbus\confirm3.png' ]
find_and_click_icons(template_paths8)
time.sleep(5)
pyautogui.click(2520,24,clicks=3)
time.sleep(1)
pyautogui.click(500,500)
print("一次镜牢成功运行")
