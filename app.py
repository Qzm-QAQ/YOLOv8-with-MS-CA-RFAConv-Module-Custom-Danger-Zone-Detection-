import cv2
from ultralytics import YOLO
from tkinter import filedialog, simpledialog
import numpy as np
import mss
import os
import tkinter as tk
from tkinter import messagebox
import playsound
import time
import platform
import logging
import threading  # 确保导入 threading 模块

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置全局变量
config = {
    "model_path": "",
    "width": 1024,
    "height": 768,
    "sound_path": ""
}

# 全局变量，用于存储危险区域和控制绘制状态
danger_zone = []
draw_mode = False
alarm_playing = False

# 配置界面
def setup_config():
    root = tk.Tk()
    root.title("配置")
    root.geometry("400x350")  # 增加高度以容纳更多按钮

    # 选择模型
    def choose_model():
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Model Files", "*.pt")]
        )
        if filename:
            config["model_path"] = filename
            model_label.config(text=f"模型: {os.path.basename(filename)}")
            logging.info(f"选择模型文件: {filename}")

    # 设置分辨率
    def set_resolution():
        width = simpledialog.askinteger(
            "宽度",
            "输入窗口宽度:",
            initialvalue=1024,
            minvalue=640,
            maxvalue=1920
        )
        height = simpledialog.askinteger(
            "高度",
            "输入窗口高度:",
            initialvalue=768,
            minvalue=480,
            maxvalue=1080
        )
        if width and height:
            config["width"] = width
            config["height"] = height
            resolution_label.config(text=f"分辨率: {width}x{height}")
            logging.info(f"设置分辨率: {width}x{height}")

    # 选择报警声音
    def choose_sound():
        filename = filedialog.askopenfilename(
            title="选择报警声音文件",
            filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("AAC Files", "*.aac")]
        )
        if filename:
            config["sound_path"] = filename
            sound_label.config(text=f"报警声音: {os.path.basename(filename)}")
            logging.info(f"选择报警声音文件: {filename}")

    # 启动屏幕检测
    def start_screen_detection():
        if not config["model_path"]:
            messagebox.showerror("错误", "请先选择模型文件！")
            return
        if not config["sound_path"]:
            messagebox.showerror("错误", "请先选择报警声音文件！")
            return
        root.destroy()
        main_program()

    # 启动摄像头检测
    def start_camera_detection():
        if not config["model_path"]:
            messagebox.showerror("错误", "请先选择模型文件！")
            return
        if not config["sound_path"]:
            messagebox.showerror("错误", "请先选择报警声音文件！")
            return
        root.destroy()
        main_program2()

    # UI 元素
    tk.Button(root, text="选择模型", command=choose_model).pack(pady=10)
    model_label = tk.Label(root, text="模型: 未选择")
    model_label.pack()

    tk.Button(root, text="设置分辨率", command=set_resolution).pack(pady=10)
    resolution_label = tk.Label(root, text=f"分辨率: {config['width']}x{config['height']}")
    resolution_label.pack()

    tk.Button(root, text="选择报警声音", command=choose_sound).pack(pady=10)
    sound_label = tk.Label(root, text="报警声音: 未选择")
    sound_label.pack()

    tk.Button(root, text="启动屏幕检测", command=start_screen_detection).pack(pady=20)
    tk.Button(root, text="启动摄像头检测", command=start_camera_detection).pack(pady=10)

    root.mainloop()

# 播放报警声音
def play_alarm():
    global alarm_playing
    try:
        if config["sound_path"]:
            playsound.playsound(config["sound_path"])
            logging.info("报警声音播放完毕")
        else:
            logging.warning("未配置报警声音路径。")
    except Exception as e:
        logging.error(f"播放报警声音时出错: {e}")
    alarm_playing = False

# 主程序 - 屏幕检测
def main_program():
    global alarm_playing

    # 从配置中获取模型路径
    model_path = config['model_path']
    # 初始化YOLO模型
    model = YOLO(model_path)
    logging.info(f"加载模型: {model_path}")

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
        global danger_zone, draw_mode, curve_points

        if not draw_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            curve_points = [(x, y)]
            logging.info("开始绘制危险区域")
            return

        if event == cv2.EVENT_MOUSEMOVE and curve_points:
            curve_points.append((x, y))

        if event == cv2.EVENT_LBUTTONUP and curve_points:
            if len(curve_points) > 1:
                bezier_points = calculate_bezier_points(curve_points)
                danger_zone.extend(bezier_points.tolist())
                logging.info("完成绘制危险区域")
            curve_points.clear()

    # 启用/禁用绘制模式
    def toggle_draw_mode():
        global draw_mode
        draw_mode = not draw_mode
        mode = "启用" if draw_mode else "禁用"
        messagebox.showinfo("绘制模式", f"绘制模式已{mode}")
        logging.info(f"绘制模式已{mode}")

    # 重置绘制的多边形
    def reset_polygon():
        global danger_zone
        danger_zone.clear()
        logging.info("重置危险区域")

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

        # 使用 cv2.intersectConvexConvex 进行准确的多边形交集
        try:
            intersection, _ = cv2.intersectConvexConvex(danger_poly, detection_poly)
            return intersection.size > 0
        except Exception as e:
            logging.error(f"检查交集时出错: {e}")
            return False

    # 屏幕捕捉和 OpenCV 处理逻辑
    def opencv_processing():
        global alarm_playing

        with mss.mss() as sct:
            # 动态获取主监视器
            monitors = sct.monitors
            if len(monitors) < 1:
                logging.error("未检测到任何监视器。")
                return
            primary_monitor = monitors[1]  # mss.monitors[0] 是所有监视器的字典
            monitor = {
                "top": primary_monitor["top"],
                "left": primary_monitor["left"],
                "width": config["width"],
                "height": config["height"]
            }

            cv2.namedWindow("YOLOv8cam", cv2.WINDOW_NORMAL)  # 设置窗口为可调整大小
            cv2.resizeWindow("YOLOv8cam", 640, 480)  # 设置窗口尺寸为640x480

            cv2.setMouseCallback("YOLOv8cam", draw_bezier_curve)

            while True:
                sound_flag = False
                try:
                    sct_img = sct.grab(monitor)
                except Exception as e:
                    logging.error(f"屏幕捕捉失败: {e}")
                    break

                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 可选：缩放帧以适应窗口
                scale_percent = 50  # 缩放百分比
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                results = model(frame)
                boxes = results[0].boxes.xyxy

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    if len(danger_zone) > 2 and is_intersecting((x1, y1, x2, y2), danger_zone):
                        sound_flag = True
                        logging.info("报警：检测框与危险区域重叠！")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if danger_zone:
                    cv2.polylines(frame, [np.array(danger_zone)], isClosed=True, color=(0, 0, 255), thickness=2)

                cv2.imshow("YOLOv8cam", frame)

                if sound_flag and not alarm_playing:
                    alarm_playing = True
                    logging.info("启动报警声音线程")
                    threading.Thread(target=play_alarm, daemon=True).start()

                if cv2.waitKey(1) == 27:  # 27 是 ESC 键
                    break

        cv2.destroyAllWindows()

# 主程序2 - 摄像头检测
def main_program2():
    global alarm_playing

    # 从配置中获取模型路径
    model_path = config['model_path']
    # 初始化YOLO模型
    model = YOLO(model_path)
    logging.info(f"加载模型: {model_path}")

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
        global danger_zone, draw_mode, curve_points

        if not draw_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            curve_points = [(x, y)]
            logging.info("开始绘制危险区域")
            return

        if event == cv2.EVENT_MOUSEMOVE and curve_points:
            curve_points.append((x, y))

        if event == cv2.EVENT_LBUTTONUP and curve_points:
            if len(curve_points) > 1:
                bezier_points = calculate_bezier_points(curve_points)
                danger_zone.extend(bezier_points.tolist())
                logging.info("完成绘制危险区域")
            curve_points.clear()

    # 启用/禁用绘制模式
    def toggle_draw_mode():
        global draw_mode
        draw_mode = not draw_mode
        mode = "启用" if draw_mode else "禁用"
        messagebox.showinfo("绘制模式", f"绘制模式已{mode}")
        logging.info(f"绘制模式已{mode}")

    # 重置绘制的多边形
    def reset_polygon():
        global danger_zone
        danger_zone.clear()
        logging.info("重置危险区域")

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

        # 使用 cv2.intersectConvexConvex 进行准确的多边形交集
        try:
            intersection, _ = cv2.intersectConvexConvex(danger_poly, detection_poly)
            return intersection.size > 0
        except Exception as e:
            logging.error(f"检查交集时出错: {e}")
            return False

    # 播放报警声音
    def play_alarm():
        global alarm_playing
        try:
            if config["sound_path"]:
                playsound.playsound(config["sound_path"])
                logging.info("报警声音播放完毕")
            else:
                logging.warning("未配置报警声音路径。")
        except Exception as e:
            logging.error(f"播放报警声音时出错: {e}")
        alarm_playing = False

    # 摄像头捕捉和 OpenCV 处理逻辑
    def opencv_processing():
        global alarm_playing

        cap = cv2.VideoCapture(1)  # 打开摄像头（索引已更改为1）

        if not cap.isOpened():
            logging.error("无法打开摄像头")
            return

        cv2.namedWindow("YOLOv8cam", cv2.WINDOW_NORMAL)  # 设置窗口为可调整大小
        cv2.resizeWindow("YOLOv8cam", 640, 480)  # 设置窗口尺寸为640x480

        cv2.setMouseCallback("YOLOv8cam", draw_bezier_curve)

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法读取摄像头画面")
                break

            # 可选：缩放帧以适应窗口
            scale_percent = 50  # 缩放百分比
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            results = model(frame)
            boxes = results[0].boxes.xyxy

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if len(danger_zone) > 2 and is_intersecting((x1, y1, x2, y2), danger_zone):
                    if not alarm_playing:
                        alarm_playing = True
                        logging.info("检测到危险区域重叠，启动报警声音")
                        threading.Thread(target=play_alarm, daemon=True).start()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if danger_zone:
                cv2.polylines(frame, [np.array(danger_zone)], isClosed=True, color=(0, 0, 255), thickness=2)

            cv2.imshow("YOLOv8cam", frame)

            if cv2.waitKey(1) == 27:  # 27 是 ESC 键
                break

        cap.release()
        cv2.destroyAllWindows()

    # 设置 Tkinter GUI
    root = tk.Tk()
    root.geometry("400x200")
    root.title("YOLOv8 监控")
    toggle_button = tk.Button(root, text="启用/禁用绘制模式", command=toggle_draw_mode)
    toggle_button.pack(pady=10)
    reset_button = tk.Button(root, text="重置多边形", command=reset_polygon)
    reset_button.pack(pady=10)

    # 启动 OpenCV 处理
    opencv_processing()

# 主程序2 - 摄像头检测
def main_program2():
    global alarm_playing

    # 从配置中获取模型路径
    model_path = config['model_path']
    # 初始化YOLO模型
    model = YOLO(model_path)
    logging.info(f"加载模型: {model_path}")

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
        global danger_zone, draw_mode, curve_points

        if not draw_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            curve_points = [(x, y)]
            logging.info("开始绘制危险区域")
            return

        if event == cv2.EVENT_MOUSEMOVE and curve_points:
            curve_points.append((x, y))

        if event == cv2.EVENT_LBUTTONUP and curve_points:
            if len(curve_points) > 1:
                bezier_points = calculate_bezier_points(curve_points)
                danger_zone.extend(bezier_points.tolist())
                logging.info("完成绘制危险区域")
            curve_points.clear()

    # 启用/禁用绘制模式
    def toggle_draw_mode():
        global draw_mode
        draw_mode = not draw_mode
        mode = "启用" if draw_mode else "禁用"
        messagebox.showinfo("绘制模式", f"绘制模式已{mode}")
        logging.info(f"绘制模式已{mode}")

    # 重置绘制的多边形
    def reset_polygon():
        global danger_zone
        danger_zone.clear()
        logging.info("重置危险区域")

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

        # 使用 cv2.intersectConvexConvex 进行准确的多边形交集
        try:
            intersection, _ = cv2.intersectConvexConvex(danger_poly, detection_poly)
            return intersection.size > 0
        except Exception as e:
            logging.error(f"检查交集时出错: {e}")
            return False

    # 播放报警声音
    def play_alarm():
        global alarm_playing
        try:
            if config["sound_path"]:
                playsound.playsound(config["sound_path"])
                logging.info("报警声音播放完毕")
            else:
                logging.warning("未配置报警声音路径。")
        except Exception as e:
            logging.error(f"播放报警声音时出错: {e}")
        alarm_playing = False

    # 摄像头捕捉和 OpenCV 处理逻辑
    def opencv_processing():
        global alarm_playing

        cap = cv2.VideoCapture(1)  # 打开摄像头（索引已更改为1）

        if not cap.isOpened():
            logging.error("无法打开摄像头")
            return

        cv2.namedWindow("YOLOv8cam", cv2.WINDOW_NORMAL)  # 设置窗口为可调整大小
        cv2.resizeWindow("YOLOv8cam", 640, 480)  # 设置窗口尺寸为640x480

        cv2.setMouseCallback("YOLOv8cam", draw_bezier_curve)

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法读取摄像头画面")
                break

            # 可选：缩放帧以适应窗口
            scale_percent = 50  # 缩放百分比
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            results = model(frame)
            boxes = results[0].boxes.xyxy

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if len(danger_zone) > 2 and is_intersecting((x1, y1, x2, y2), danger_zone):
                    if not alarm_playing:
                        alarm_playing = True
                        logging.info("检测到危险区域重叠，启动报警声音")
                        threading.Thread(target=play_alarm, daemon=True).start()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if danger_zone:
                cv2.polylines(frame, [np.array(danger_zone)], isClosed=True, color=(0, 0, 255), thickness=2)

            cv2.imshow("YOLOv8cam", frame)

            if cv2.waitKey(1) == 27:  # 27 是 ESC 键
                break

        cap.release()
        cv2.destroyAllWindows()

    # 设置 Tkinter GUI
    root = tk.Tk()
    root.geometry("400x200")
    root.title("YOLOv8 监控")
    toggle_button = tk.Button(root, text="启用/禁用绘制模式", command=toggle_draw_mode)
    toggle_button.pack(pady=10)
    reset_button = tk.Button(root, text="重置多边形", command=reset_polygon)
    reset_button.pack(pady=10)

    # 启动 OpenCV 处理
    opencv_processing()

if __name__ == "__main__":
    # 检查是否为 macOS 并指导用户授予必要权限
    if platform.system() == "Darwin":
        logging.info("请确保已为此应用授予屏幕录制权限。")
        logging.info("前往 系统偏好设置 -> 安全性与隐私 -> 隐私 -> 屏幕录制，并添加 Python。")
    setup_config()
