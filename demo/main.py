import cv2
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
import pyautogui
import os
import time
from typing import List, Dict, Optional
from app.model import ALPR


alpr = ALPR()


class Config:
    VIDEO_PATH = r"D:\VSCodeProjects\license_plate_regconition\media\video_test.mp4"
    OUTPUT_DIR = r'D:\VSCodeProjects\license_plate_regconition\demo\output'
    VIDEO_WIDTH_RATIO = 0.55
    VIDEO_HEIGHT_RATIO = 0.9
    FRAME_UPDATE_INTERVAL = 33
    VIDEO_UPDATE_INTERVAL = 20
    DETECTION_INTERVAL = 10
    VIDEO_FPS = 20.0
    FONT_FAMILY = "Courier"
    FONT_SIZE = 10

class PlateDetector:
    @staticmethod
    def detect_plate(frame: np.ndarray) -> str:
        try:
            texts = alpr.predict(frame)
            if not texts:
                return ''
            return str(texts)
        except:
            return ''

class VideoRecorder:
    def __init__(self, output_path: str, fps: float, frame_size: tuple):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer: Optional[cv2.VideoWriter] = None
        self._initialize_writer()
    
    def _initialize_writer(self) -> None:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
    
    def write_frame(self, frame: np.ndarray) -> None:
        if self.writer:
            self.writer.write(frame)
    
    def release(self) -> None:
        if self.writer:
            self.writer.release()
            self.writer = None

class Logger:
    def __init__(self, log_widget: tk.Text):
        self.log_widget = log_widget
        self.detected_plates: List[Dict[str, str]] = []
    
    def log_session_start(self) -> None:
        timestamp = self._get_timestamp()
        self._insert_log(f"Session started: {timestamp}\n\n")
    
    def log_detection(self, plate: str) -> None:
        timestamp = self._get_timestamp()
        self.detected_plates.append({'plate': plate, 'time': timestamp})
        self._insert_log(f"[{timestamp}] {plate}\n")
        self.log_widget.see("end")
    
    def log_session_end(self) -> None:
        timestamp = self._get_timestamp()
        self._insert_log(f"\nSession ended: {timestamp}\n")
        self._insert_log(f"Total detected: {len(self.detected_plates)}\n")
    
    def save_log_file(self, output_dir: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_dir, f'log_{timestamp}.txt')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(self.log_widget.get("1.0", "end-1c"))
        except Exception as e:
            print(f"Error saving log file: {e}")
    
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _insert_log(self, message: str) -> None:
        self.log_widget.insert("end", message)

class ALPRDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.frame_count = 0
        self.last_time = time.time()
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_recorder: Optional[VideoRecorder] = None
        self.logger: Optional[Logger] = None
        self._setup_window()
        self._setup_ui()
        self._initialize_components()
        self._start_session()
    
    def _setup_window(self) -> None:
        self.root.title("ALPR Demo")
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.close_app())
    
    def _setup_ui(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.video_width = int(screen_width * Config.VIDEO_WIDTH_RATIO)
        self.video_height = int(screen_height * Config.VIDEO_HEIGHT_RATIO)
        self.screen_size = (screen_width, screen_height)
        
        self.left_frame = tk.Frame(self.root, bg='black')
        self.right_frame = tk.Frame(self.root, bg='lightgray')
        self.left_frame.pack(side="left", fill="both", expand=True)
        self.right_frame.pack(side="right", fill="both")
        
        self.video_label = tk.Label(self.left_frame, bg='black')
        self.video_label.pack(expand=True)
        
        self.log_box = tk.Text(self.right_frame, wrap="word", font=(Config.FONT_FAMILY, Config.FONT_SIZE))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
    
    def _initialize_components(self) -> None:
        self.cap = cv2.VideoCapture(Config.VIDEO_PATH)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {Config.VIDEO_PATH}")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        output_filename = self._generate_output_filename()
        self.video_recorder = VideoRecorder(output_filename, Config.VIDEO_FPS, self.screen_size)
        self.logger = Logger(self.log_box)
    
    def _generate_output_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(Config.OUTPUT_DIR, f'demo_output_{timestamp}.mp4')
    
    def _start_session(self) -> None:
        if self.logger:
            self.logger.log_session_start()
        self.root.after(Config.FRAME_UPDATE_INTERVAL, self.update_frame)
    
    def update_frame(self) -> None:
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self._process_frame(frame)
            self._schedule_next_update()
        else:
            self._handle_video_end()
    
    def _process_frame(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        if self.frame_count % Config.DETECTION_INTERVAL == 0:
            self._detect_and_log_plate(frame)
        self._update_video_display(frame)
        
        current_time = time.time()
        if current_time - self.last_time > 0.05:
            self._record_screen()
            self.last_time = current_time
    
    def _detect_and_log_plate(self, frame: np.ndarray) -> None:
        if not self.logger:
            return
        plate = PlateDetector.detect_plate(frame)
        if plate:
            self.logger.log_detection(plate)
    
    def _update_video_display(self, frame: np.ndarray) -> None:
        frame_resized = cv2.resize(frame, (self.video_width, self.video_height), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        setattr(self.video_label, 'image', imgtk)
    
    def _record_screen(self) -> None:
        if not self.video_recorder:
            return
        try:
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            self.video_recorder.write_frame(screenshot_bgr)
        except Exception as e:
            print(f"Video recording error: {e}")
    
    def _schedule_next_update(self) -> None:
        self.root.after(Config.VIDEO_UPDATE_INTERVAL, self.update_frame)
    
    def _handle_video_end(self) -> None:
        if self.logger:
            self.logger.log_session_end()
            self.logger.save_log_file(Config.OUTPUT_DIR)
        self._cleanup_resources()
        self.root.after(3000, self.close_app)
    
    def _cleanup_resources(self) -> None:
        if self.video_recorder:
            self.video_recorder.release()
        if self.cap:
            self.cap.release()
    
    def close_app(self) -> None:
        try:
            self._cleanup_resources()
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            self.root.quit()
            self.root.destroy()

def main() -> None:
    try:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        root = tk.Tk()
        app = ALPRDemo(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()