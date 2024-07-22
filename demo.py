import cv2
from ultralytics import YOLO
from tkinter import filedialog, simpledialog
import numpy as np
import mss
import os
import tkinter as tk
from tkinter import messagebox
import threading
import playsound
import time

# 配置全局变量
config = {
    "model_path": "",
    "width": 1024,
    "height": 768,
    "sound_path":""
}
# 配置界面
def setup_config():
    root = tk.Tk()
    root.title("配置")
    # 选择模型
    def choose_model():
        filename = filedialog.askopenfilename(title="选择模型文件", filetypes=[("Model Files", "*.pt")])
        if filename:
            config["model_path"] = filename
            model_label.config(text=f"模型: {os.path.basename(filename)}")
    # 设置分辨率
    def set_resolution():
        width = simpledialog.askinteger("宽度", "输入窗口宽度:", initialvalue=1024, minvalue=640, maxvalue=1920)
        height = simpledialog.askinteger("高度", "输入窗口高度:", initialvalue=768, minvalue=480, maxvalue=1080)
        if width and height:
            config["width"] = width
            config["height"] = height
            resolution_label.config(text=f"分辨率: {width}x{height}")
    # 启动按钮
    def start_program():
        root.destroy()
        main_program()
    def start_program2():
        root.destroy()
        main_program2()

    tk.Button(root, text="选择模型", command=choose_model).pack(pady=10)
    model_label = tk.Label(root, text="模型: 未选择")
    model_label.pack()
    tk.Button(root, text="设置分辨率", command=set_resolution).pack(pady=10)
    resolution_label = tk.Label(root, text=f"分辨率: {config['width']}x{config['height']}")
    resolution_label.pack()

    tk.Button(root, text="启动屏幕检测", command=start_program).pack(pady=20)
    tk.Button(root, text="启动远程摄像头", command=start_program2).pack(pady=20)
    root.mainloop()


# 全局变量，用于存储危险区域和控制绘制状态
danger_zone = []
drawing = False
draw_mode = False  # 绘制模式开关
temp_point = None  # 临时点用于绘制过程中
curve_points = []  # 存储绘制的曲线点
alarm_playing = False
lock = threading.Lock()  # 用于同步访问 alarm_playing 变量
# 主程序
def main_program():
    # 从配置中获取模型路径
    model_path = config['model_path']
    # 初始化YOLO模型
    model = YOLO(model_path)
    def calculate_bezier_points(points, num_points=100):
        points = np.array(points)
        N = len(points)
        t = np.linspace(0, 1, num_points)
        curve_points = np.zeros((num_points, 2))

        for i in range(num_points):
            point = np.zeros(2)
            for j in range(N):
                binom = np.math.factorial(N - 1) / (np.math.factorial(j) * np.math.factorial(N - 1 - j))
                point += binom * ((1 - t[i]) ** (N - 1 - j)) * (t[i] ** j) * points[j]
            curve_points[i] = point

        return curve_points.astype(int)

    # 鼠标回调函数，用于绘制贝塞尔曲线
    def draw_bezier_curve(event, x, y, flags, param):
        global curve_points, drawing, draw_mode, danger_zone

        if not draw_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            curve_points = [(x, y)]  # 开始一个新的曲线

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            curve_points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(curve_points) > 1:
                bezier_points = calculate_bezier_points(curve_points)
                danger_zone += bezier_points.tolist()

    # 启用/禁用绘制模式
    def toggle_draw_mode():
        global draw_mode
        draw_mode = not draw_mode
        mode = "启用" if draw_mode else "禁用"
        messagebox.showinfo("绘制模式", f"绘制模式已{mode}")

    # 撤销最后一个点
    #def undo_last_point():
    # if danger_zone:
        # danger_zone.pop()

    # 重置绘制的多边形
    def reset_polygon():
        global danger_zone
        danger_zone = []
    # 设置 Tkinter GUI
    root = tk.Tk()
    root.geometry("400x100")
    root.title("YOLOv8 监控")
    toggle_button = tk.Button(root, text="启用/禁用绘制模式", command=toggle_draw_mode)
    toggle_button.pack()
    reset_button = tk.Button(root, text="重置多边形", command=reset_polygon)
    reset_button.pack()
    # 检查检测框与危险区域的交集
    def is_intersecting(detection, danger_zone):
        detection_box = np.array([
            [detection[0], detection[1]],  # 左上角
            [detection[2], detection[1]],  # 右上角
            [detection[2], detection[3]],  # 右下角
            [detection[0], detection[3]]   # 左下角
        ])
        danger_poly = np.array(danger_zone)
        detection_poly = detection_box

        return cv2.rotatedRectangleIntersection(cv2.minAreaRect(danger_poly), cv2.minAreaRect(detection_poly))[0] != 0

    # 屏幕捕捉和 OpenCV 处理逻辑
    def play_alarm():
        try:
            playsound.playsound("alarm.mp3")
            print("报警声音播放完毕")
        except Exception as e:
            print(f"播放报警声音时出错: {e}")

    def opencv_processing():
        global alarm_playing  # 声明为全局变量
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1024, "height": 768}
            cv2.namedWindow("YOLOv8cam")
            cv2.setMouseCallback("YOLOv8cam", draw_bezier_curve)
            while True:
                sound_flag=False
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                results = model(frame)
                boxes = results[0].boxes.xyxy

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    if len(danger_zone) > 2 and is_intersecting((x1, y1, x2, y2), danger_zone):
                        sound_flag=True
                        print("报警：检测框与危险区域重叠！")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if danger_zone:
                    cv2.polylines(frame, [np.array(danger_zone)], isClosed=False, color=(0, 0, 255), thickness=2)

                cv2.imshow("YOLOv8cam", frame)
                if sound_flag:
                    with lock:
                        if not alarm_playing:
                            alarm_playing = True
                            print("启动报警声音线程")
                            threading.Thread(target=play_alarm).start()
                            threading.Thread(target=reset_alarm_status).start()
                if cv2.waitKey(1) == 27:  # 27 是 ESC 键
                    break
               
        cv2.destroyAllWindows()
    def reset_alarm_status():  
        time.sleep(300)  # 假设报警声音时长为5秒，根据实际时长调整
        global alarm_playing
        with lock:
            alarm_playing = False
    # 开始 OpenCV 线程
    thread = threading.Thread(target=opencv_processing)
    thread.start()

    # 开始 Tkinter 主事件循环
    root.mainloop()

    # 确保 OpenCV 线程也关闭
    thread.join()
def main_program2():
    # 从配置中获取模型路径
    model_path = config['model_path']
    # 初始化YOLO模型
    model = YOLO(model_path)
    def calculate_bezier_points(points, num_points=100):
        points = np.array(points)
        N = len(points)
        t = np.linspace(0, 1, num_points)
        curve_points = np.zeros((num_points, 2))

        for i in range(num_points):
            point = np.zeros(2)
            for j in range(N):
                binom = np.math.factorial(N - 1) / (np.math.factorial(j) * np.math.factorial(N - 1 - j))
                point += binom * ((1 - t[i]) ** (N - 1 - j)) * (t[i] ** j) * points[j]
            curve_points[i] = point

        return curve_points.astype(int)

    # 鼠标回调函数，用于绘制贝塞尔曲线
    def draw_bezier_curve(event, x, y, flags, param):
        global curve_points, drawing, draw_mode, danger_zone

        if not draw_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            curve_points = [(x, y)]  # 开始一个新的曲线

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            curve_points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(curve_points) > 1:
                bezier_points = calculate_bezier_points(curve_points)
                danger_zone += bezier_points.tolist()

    # 启用/禁用绘制模式
    def toggle_draw_mode():
        global draw_mode
        draw_mode = not draw_mode
        mode = "启用" if draw_mode else "禁用"
        messagebox.showinfo("绘制模式", f"绘制模式已{mode}")



    # 重置绘制的多边形
    def reset_polygon():
        global danger_zone
        danger_zone = []

    # 设置 Tkinter GUI
    root = tk.Tk()
    root.geometry("400x100")
    root.title("YOLOv8 监控")
    toggle_button = tk.Button(root, text="启用/禁用绘制模式", command=toggle_draw_mode)
    toggle_button.pack()
    reset_button = tk.Button(root, text="重置多边形", command=reset_polygon)
    reset_button.pack()
    # 检查检测框与危险区域的交集
    def is_intersecting(detection, danger_zone):
        detection_box = np.array([
            [detection[0], detection[1]],  # 左上角
            [detection[2], detection[1]],  # 右上角
            [detection[2], detection[3]],  # 右下角
            [detection[0], detection[3]]   # 左下角
        ])
        danger_poly = np.array(danger_zone)
        detection_poly = detection_box

        return cv2.rotatedRectangleIntersection(cv2.minAreaRect(danger_poly), cv2.minAreaRect(detection_poly))[0] != 0
    def opencv_processing():
        cap = cv2.VideoCapture(0)  # Open the camera

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        cv2.namedWindow("YOLOv8cam")
        cv2.setMouseCallback("YOLOv8cam", draw_bezier_curve)

        while True:
            res, frame = cap.read()
            if not res:
                print("无法读取摄像头画面")
                break

            results = model(frame)
            boxes = results[0].boxes.xyxy

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if len(danger_zone) > 2 and is_intersecting((x1, y1, x2, y2), danger_zone):
                    print("报警：检测框与危险区域重叠！")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if danger_zone:
                cv2.polylines(frame, [np.array(danger_zone)], isClosed=False, color=(0, 0, 255), thickness=2)

            cv2.imshow("YOLOv8cam", frame)

            if cv2.waitKey(1) == 27:  # 27 is the ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

    # 开始 OpenCV 线程
    thread = threading.Thread(target=opencv_processing)
    thread.start()

    # 开始 Tkinter 主事件循环
    root.mainloop()

    # 确保 OpenCV 线程也关闭
    thread.join()
if __name__ == "__main__":
    setup_config()