# --- START OF FILE RK3588_MLP_System.py ---

import sys
import os
import cv2
import numpy as np
import time
import mediapipe as mp
import serial
import pyqtgraph as pg
import struct
import collections
import queue
import subprocess
import threading
from multiprocessing import Process, Queue, Value, Array

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QProgressBar, QFrame,
                             QMessageBox, QSizePolicy, QTextEdit, QDoubleSpinBox,
                             QGraphicsDropShadowEffect, QShortcut)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor

# 🌟 唯一指定 NPU 引擎，彻底告别 TensorFlow 和 ONNX
from rknnlite.api import RKNNLite

from gesture_runtime import ActionStateMachine, PhraseTranslator, SIMPLE_HAND_CONNECTIONS, dataset_split_dir, list_action_names, load_runtime_config, next_sequence_index, top1_with_margin

# ================= 🔧 全局动态路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CONFIG = load_runtime_config(BASE_DIR)

# 纯视觉特征数据集与模型路径
DATA_PATH = os.path.join(BASE_DIR, 'MP_Data_Fusion')
MODEL_PATH_RKNN = os.path.join(BASE_DIR, 'action_fusion.rknn')

# 本地断网 Qwen 大模型路径
LLM_DEMO_PATH = '/home/elf/examples/rkllm_api_demo/deploy/build/build_linux_aarch64_Release/llm_demo'
LLM_MODEL_PATH = os.path.join(BASE_DIR, 'Qwen2.5-1.5B-Instruct_w8a8.rkllm')

SEQUENCE_LENGTH = 30
VISUAL_SIZE = 102
VELOCITY_SIZE = 102
EMG_SIZE = 16
INPUT_SIZE = VISUAL_SIZE + VELOCITY_SIZE
SERIAL_BAUDRATE = 115200
# =======================================================

PRO_STYLE = """
QMainWindow { background-color: #0D0D12; }
QWidget { color: #E0E0E0; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 13px; }
QFrame#Panel { background-color: #16161E; border: 1px solid #2B2B36; border-radius: 8px; }
QFrame#CameraFrame { border: 2px solid #2B2B36; background-color: #000000; border-radius: 8px; }
QLabel#ResultText { font-size: 50px; font-weight: 900; color: #00E5FF; font-family: 'Consolas'; }
QFrame#Dashboard { background-color: #1A1A24; border: 1px solid #3A3A4A; border-radius: 6px; }
QLabel#DashText { font-family: 'Consolas'; font-size: 14px; font-weight: bold; color: #00FF9D; }
QLabel#DashAlert { font-family: 'Consolas'; font-size: 14px; font-weight: bold; color: #FF3366; }
QPushButton { background-color: #242430; color: #00E5FF; border: 1px solid #00E5FF; padding: 10px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #00E5FF; color: #000000; }
QPushButton:pressed { background-color: #00B3CC; }
QPushButton:disabled { color: #555555; border-color: #333333; background-color: #1A1A1A; }
QProgressBar { border: 1px solid #3A3A4A; border-radius: 4px; text-align: center; color: white; background-color: #000000; height: 18px; }
QProgressBar::chunk { background-color: #00E5FF; border-radius: 3px; }
QTextEdit, QLineEdit, QDoubleSpinBox { background-color: #0A0A0F; border: 1px solid #3A3A4A; color: #00FF9D; border-radius: 4px; padding: 5px; font-family: 'Consolas'; }
QTextEdit:focus, QLineEdit:focus, QDoubleSpinBox:focus { border: 1px solid #00E5FF; }
"""

def extract_visual_keypoints(results, hand_threshold):
    lh, rh = np.zeros(51), np.zeros(51)
    hand_detected_valid = False

    if results.multi_hand_landmarks:
        for idx, hand_res in enumerate(results.multi_hand_landmarks):
            multi_handedness = results.multi_handedness
            if multi_handedness and len(multi_handedness) > idx:
                if multi_handedness[idx].classification[0].score < hand_threshold: continue
                label = multi_handedness[idx].classification[0].label
                hand_detected_valid = True
            else: continue

            pts = np.array([[lm.x, lm.y] for lm in hand_res.landmark])
            base = pts[0]
            rel_pts = pts - base
            mcp_dist = np.linalg.norm(rel_pts[9])

            if mcp_dist > 1e-6: norm_pts = rel_pts / mcp_dist
            else: norm_pts = rel_pts
            features = list(norm_pts.flatten())

            def get_angle(p1, p2, p3):
                v1, v2 = p1 - p2, p3 - p2
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6: return 0.0
                return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)) / np.pi

            angles = [
                get_angle(pts[1], pts[2], pts[4]),
                get_angle(pts[5], pts[6], pts[8]),
                get_angle(pts[9], pts[10], pts[12]),
                get_angle(pts[13], pts[14], pts[16]),
                get_angle(pts[17], pts[18], pts[20])
            ]

            dists = [
                np.linalg.norm(rel_pts[4] - rel_pts[8]),
                np.linalg.norm(rel_pts[4] - rel_pts[12]),
                np.linalg.norm(rel_pts[4] - rel_pts[16]),
                np.linalg.norm(rel_pts[4] - rel_pts[20])
            ]
            if mcp_dist > 1e-6: dists = [d / mcp_dist for d in dists]

            features.extend(angles)
            features.extend(dists)

            if label == 'Left': lh = np.array(features)
            else: rh = np.array(features)

    return np.concatenate([lh, rh]), hand_detected_valid


def draw_simple_hand_skeleton(image, hand_landmarks):
    h, w = image.shape[:2]
    points = []
    for lm in hand_landmarks.landmark:
        points.append((int(lm.x * w), int(lm.y * h)))
    for a, b in SIMPLE_HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], (0, 229, 255), 2)
    for x, y in points:
        cv2.circle(image, (x, y), 3, (0, 255, 157), -1)


class SysMonitorWorker(QThread):
    sys_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            temp_str, npu_str = "--", "--"
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_str = f"{float(f.read().strip()) / 1000.0:.1f} °C"
            except: temp_str = "N/A"

            try:
                with open('/sys/kernel/debug/rknpu/load', 'r') as f:
                    for line in f.readlines():
                        if "NPU load" in line:
                            raw = line.split('NPU load:')[1].strip()
                            if raw.endswith(','): raw = raw[:-1]
                            npu_str = raw.replace('Core', 'C')
                            break
            except: npu_str = "N/A (Req Sudo)"

            self.sys_signal.emit(temp_str, npu_str)
            time.sleep(1)

    def stop(self):
        self.running = False; self.wait()


class EMGWorker(QThread):
    data_signal = pyqtSignal(np.ndarray, float)
    def __init__(self, shared_emg_array):
        super().__init__()
        self.running = True; self.ser = None
        self.simulation_mode = False; self.is_init = False
        self.HEADER, self.PACKET_SIZE = b'\x55\xAA', 34
        self.buffer = bytearray()
        self.moving_baseline, self.ema_envelope = np.zeros(EMG_SIZE), np.zeros(EMG_SIZE)
        self.shared_emg = shared_emg_array

    def auto_connect(self):
        try:
            self.ser = serial.Serial('COM15', SERIAL_BAUDRATE, timeout=0.01)
            self.simulation_mode = False
        except: self.simulation_mode = True

    def run(self):
        self.auto_connect()
        while self.running:
            if self.simulation_mode: time.sleep(0.02)
            else:
                if self.ser and self.ser.is_open:
                    try:
                        if self.ser.in_waiting: self.buffer.extend(self.ser.read(self.ser.in_waiting))
                        while len(self.buffer) >= self.PACKET_SIZE:
                            idx = self.buffer.find(self.HEADER)
                            if idx == -1: self.buffer = self.buffer[-2:]; break
                            if idx > 0: self.buffer = self.buffer[idx:]; continue
                            if len(self.buffer) < self.PACKET_SIZE: break
                            packet = self.buffer[:self.PACKET_SIZE]; self.buffer = self.buffer[self.PACKET_SIZE:]
                            try:
                                raw_data = np.array(struct.unpack('<16H', packet[2:]), dtype=float)
                                if not self.is_init:
                                    self.moving_baseline = raw_data.copy(); self.is_init = True; continue
                                self.moving_baseline = self.moving_baseline * 0.98 + raw_data * 0.02
                                rectified_data = np.abs(raw_data - self.moving_baseline)
                                for i in range(EMG_SIZE):
                                    if rectified_data[i] > self.ema_envelope[i]:
                                        self.ema_envelope[i] = rectified_data[i] * 0.4 + self.ema_envelope[i] * 0.6
                                    else:
                                        self.ema_envelope[i] = rectified_data[i] * 0.05 + self.ema_envelope[i] * 0.95
                                for i in range(EMG_SIZE): self.shared_emg[i] = self.ema_envelope[i]
                                self.data_signal.emit(self.ema_envelope, self.ema_envelope[0])
                            except: pass
                    except: time.sleep(1); self.auto_connect()
    def stop(self):
        self.running = False;
        if self.ser: self.ser.close()
        self.wait()


class VisionProcess(Process):
    def __init__(self, cmd_queue, out_queue, shared_ai_thresh, shared_hand_thresh):
        super().__init__()
        self.cmd_queue = cmd_queue
        self.out_queue = out_queue
        self.shared_ai_thresh = shared_ai_thresh
        self.shared_hand_thresh = shared_hand_thresh

    def run(self):
        try:
            os.system(f"taskset -p 0xf0 {os.getpid()}")
        except:
            pass

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        # 依然保持 model_complexity=0 (Lite模型)，保证流畅度
        hands = mp_hands.Hands(model_complexity=int(RUNTIME_CONFIG["mediapipe_model_complexity"]), min_detection_confidence=0.3, min_tracking_confidence=0.3)

        model, actions = None, []
        if os.path.exists(MODEL_PATH_RKNN):
            try:
                model = RKNNLite()
                if model.load_rknn(MODEL_PATH_RKNN) == 0:
                    model.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
                if os.path.exists(DATA_PATH):
                    actions = np.array(
                        list_action_names(DATA_PATH))
            except:
                pass

        cap = cv2.VideoCapture(21)
        if not cap.isOpened(): cap = (cv2.VideoCapture(22))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(RUNTIME_CONFIG["camera_width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RUNTIME_CONFIG["camera_height"]))

        mode, record_split, record_action, record_seq_idx = "PREDICT", "train", "", 0
        is_recording, frame_counter_rec = False, 0
        sequence, pred_history = [], collections.deque(maxlen=int(RUNTIME_CONFIG["prediction_smoothing"]))
        prev_visual_data = np.zeros(VISUAL_SIZE)
        prev_time, running = time.time(), True

        # 🌟 视觉容错与残影计数器 (保留)
        lost_tolerance = 0
        MAX_LOST = 5  # 允许画面最多连续丢失 5 帧而不中断识别

        while running and cap.isOpened():
            try:
                cmd = self.cmd_queue.get_nowait()
                if cmd[0] == "STOP":
                    running = False
                elif cmd[0] == "PREDICT":
                    mode = "PREDICT"
                    pred_history.clear()
                    sequence.clear()
                    if os.path.exists(MODEL_PATH_RKNN):
                        model = RKNNLite()
                        model.load_rknn(MODEL_PATH_RKNN)
                        model.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
                        actions = np.array(
                            list_action_names(DATA_PATH))
                elif cmd[0] == "RECORD_TRIGGER":
                    mode = "RECORD"
                    record_split, record_action, record_seq_idx = cmd[1], cmd[2], cmd[3]
                    is_recording, frame_counter_rec = True, 0
                    sequence.clear()
            except queue.Empty:
                pass

            ret, frame = cap.read()
            if not ret: break

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0)
            prev_time = curr_time

            display_image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            small_image = cv2.resize(display_image, (int(RUNTIME_CONFIG["process_width"]), int(RUNTIME_CONFIG["process_height"])))

            visual_data, velocity_data = np.zeros(VISUAL_SIZE), np.zeros(VELOCITY_SIZE)
            hand_detected_valid, status_msg = False, "TARGET LOST"

            # ==========================================
            # 🌟 彻底干掉奇偶抽帧！每一帧都让 MediaPipe 疯狂干活！
            results = hands.process(small_image)
            if results.multi_hand_landmarks:
                visual_data, hand_detected_valid = extract_visual_keypoints(results, self.shared_hand_thresh.value)
                if hand_detected_valid:
                    status_msg = "TRACKING"
                    if RUNTIME_CONFIG["draw_landmarks"]:
                        for hand_landmarks in results.multi_hand_landmarks:
                            if RUNTIME_CONFIG.get("draw_full_landmarks", False):
                                mp_drawing.draw_landmarks(display_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            else:
                                draw_simple_hand_skeleton(display_image, hand_landmarks)
                    velocity_data = visual_data - prev_visual_data
                    prev_visual_data = visual_data

                    lost_tolerance = 0  # 成功捕捉到目标，立刻清空容错计数！

            # 如果这一帧瞎了（比如手挥太快产生运动模糊）
            if not hand_detected_valid:
                lost_tolerance += 1
                if lost_tolerance < MAX_LOST and np.sum(prev_visual_data) > 0:
                    # 触发残影容错！假装手还停留在上一帧的位置
                    visual_data, velocity_data = prev_visual_data, np.zeros(VELOCITY_SIZE)
                    hand_detected_valid, status_msg = True, "TRACKING (RECOVER)"
                else:
                    prev_visual_data = np.zeros(VISUAL_SIZE)  # 彻底丢失
            # ==========================================

            fusion_data = np.concatenate([visual_data, velocity_data])
            action_res, prob_res, rec_progress, rec_done_msg = "--", 0.0, -1, None

            if mode == "RECORD" and is_recording:
                if hand_detected_valid:
                    save_root = dataset_split_dir(DATA_PATH, record_split)
                    save_path = os.path.join(save_root, record_action, str(record_seq_idx))
                    if frame_counter_rec == 0: os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, str(frame_counter_rec)), fusion_data)
                    frame_counter_rec += 1
                    rec_progress = frame_counter_rec
                    if frame_counter_rec == SEQUENCE_LENGTH:
                        is_recording, mode, rec_done_msg = False, "IDLE", f"REC_DONE: {record_seq_idx}"

            elif mode == "PREDICT" and model is not None:
                if hand_detected_valid:
                    sequence.append(fusion_data)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        outputs = model.inference(inputs=[np.expand_dims(sequence, axis=0).astype(np.float32)])
                        pred_history.append(outputs[0][0])
                        smooth_res = np.mean(pred_history, axis=0)
                        curr_idx, top_prob, margin = top1_with_margin(smooth_res)
                        if top_prob > self.shared_ai_thresh.value and margin >= float(RUNTIME_CONFIG["margin_threshold"]):
                            action_res = str(actions[curr_idx]) if len(actions) > curr_idx else "UNKNOWN"
                            prob_res = float(top_prob)
                else:
                    # 只有当容错值完全耗尽（超过5帧没看到手），才允许清空动作队列！
                    if lost_tolerance >= MAX_LOST:
                        sequence = []
                        pred_history.clear()

            if not self.out_queue.full():
                self.out_queue.put({'image': display_image, 'fps': fps, 'status': status_msg,
                                    'action': action_res, 'prob': prob_res, 'rec_progress': rec_progress,
                                    'rec_done': rec_done_msg})
        cap.release()


class UIBridge(QThread):
    image_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, float)
    rec_progress_signal = pyqtSignal(int)
    fps_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    def __init__(self, out_queue): super().__init__(); self.out_queue = out_queue; self.running = True
    def run(self):
        while self.running:
            try:
                data = self.out_queue.get(timeout=0.1)
                self.image_signal.emit(data['image']); self.fps_signal.emit(data['fps']); self.status_signal.emit(data['status'])
                self.result_signal.emit(data['action'], data['prob'])
                if data['rec_progress'] != -1: self.rec_progress_signal.emit(data['rec_progress'])
                if data['rec_done']: self.result_signal.emit(data['rec_done'], 1.0)
            except queue.Empty: pass
    def stop(self): self.running = False; self.wait()


class VoiceWorker(QThread):
    def __init__(self):
        super().__init__()
        self.text = ""; self.is_muted = False
        self.cache_dir = os.path.join(BASE_DIR, "voice_cache")
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)
    def speak(self, text):
        if not self.is_muted: self.text = text; self.start()
    def run(self):
        try:
            audio_file = os.path.join(self.cache_dir, f"{self.text}.mp3")
            if not os.path.exists(audio_file):
                subprocess.run(['edge-tts', '--voice', 'zh-CN-XiaoxiaoNeural', '--text', self.text, '--write-media', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(audio_file):
                subprocess.run(['mpg123', '-q', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass


class LocalLLMWorker(QThread):
    result_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.words, self.process, self.ready, self.is_enabled = [], None, False, True
        self.pending_key = ""
        self.translator = PhraseTranslator(BASE_DIR)
        self.start_engine()

    def start_engine(self):
        if not self.is_enabled:
            return
        if not os.path.exists(LLM_DEMO_PATH):
            self.log_signal.emit(f"LLM demo not found: {LLM_DEMO_PATH}")
            return
        try:
            self.log_signal.emit(">>> Starting local Qwen2.5 process...")
            self.process = subprocess.Popen(
                [LLM_DEMO_PATH, LLM_MODEL_PATH, "256", "1024"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._read_output, daemon=True).start()
        except Exception as e:
            self.log_signal.emit(f"LLM start failed: {e}")

    def stop_engine(self):
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
        self.ready = False
        self.log_signal.emit("Local LLM stopped")

    def toggle_engine(self):
        self.is_enabled = not self.is_enabled
        if self.is_enabled:
            self.start_engine()
        else:
            self.stop_engine()
        return self.is_enabled

    def _read_output(self):
        current = ""
        try:
            while self.process:
                char = self.process.stdout.read(1)
                if not char:
                    break
                current += char
                if current.endswith("user:") or current.endswith("user: ") or current.endswith("Human:"):
                    res = current[:-5].strip()
                    if "robot:" in res:
                        res = res.split("robot:")[-1].strip()
                    elif "Assistant:" in res:
                        res = res.split("Assistant:")[-1].strip()
                    if res and self.words:
                        self.translator.remember(self.words, res)
                        self.result_signal.emit(res)
                        self.words = []
                    self.ready = True
                    current = ""
        except Exception:
            pass

    def translate(self, words):
        self.words = words
        cached = self.translator.lookup(words)
        if cached:
            self.result_signal.emit(cached)
            self.words = []
            return

        fallback_text = self.translator.fallback(words)
        if not self.is_enabled or not self.process:
            self.log_signal.emit(f"Rule sentence: {fallback_text}")
            self.result_signal.emit(fallback_text)
            self.words = []
            return

        if self.ready:
            self.ready = False
            words_str = "|".join(self.words)
            prompt = (
                "You are a strict Chinese sign-language sentence formatter. "
                "Only reorder the given words and add necessary Chinese particles or punctuation. "
                "Do not add any meaning that is not present in the input. "
                f"Input words: [{words_str}]. Output one Chinese sentence only.\n"
            )
            try:
                self.process.stdin.write(prompt)
                self.process.stdin.flush()
            except Exception:
                self.log_signal.emit("LLM pipe failed; using rule sentence")
                self.result_signal.emit(fallback_text)
                self.words = []
        else:
            self.log_signal.emit(f"LLM busy; using rule sentence: {fallback_text}")
            self.result_signal.emit(fallback_text)
            self.words = []


class FusionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEUROLINK V8.0 - RK3588 纯 NPU 完全体")
        self.resize(1400, 850)
        self.setStyleSheet(PRO_STYLE)

        self.wave_buffer = collections.deque([2048] * 500, maxlen=500)
        self.cmd_queue = Queue(); self.out_queue = Queue(maxsize=2)
        self.shared_emg = Array('d', EMG_SIZE); self.shared_ai_thresh = Value('f', 0.80); self.shared_hand_thresh = Value('f', 0.50)

        self.sys_monitor = SysMonitorWorker()
        self.emg_worker = EMGWorker(self.shared_emg)
        self.voice_worker = VoiceWorker()
        self.llm_worker = LocalLLMWorker()
        self.ui_bridge = UIBridge(self.out_queue)

        self.vision_process = VisionProcess(self.cmd_queue, self.out_queue, self.shared_ai_thresh, self.shared_hand_thresh)

        self.word_buffer = []
        self.record_split = str(RUNTIME_CONFIG.get("record_split", "train")).lower()
        self.action_state = ActionStateMachine(RUNTIME_CONFIG["confirm_frames"], RUNTIME_CONFIG["cooldown_seconds"])
        self.last_action, self.locked_action, self.last_valid_time, self.action_confirm_count = "", "", time.time(), 0

        self.init_ui(); self.refresh_record_split_button(); self.connect_signals(); self.setup_shortcuts()
        self.emg_worker.start(); self.vision_process.start(); self.ui_bridge.start(); self.sys_monitor.start()

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QHBoxLayout(central); layout.setContentsMargins(15, 15, 15, 15); layout.setSpacing(15)

        nav = QFrame(); nav.setObjectName("Panel"); nav.setFixedWidth(280)
        nav_l = QVBoxLayout(nav); nav_l.setContentsMargins(20, 20, 20, 20)
        title = QLabel("NEUROLINK"); title.setObjectName("ResultText"); title.setStyleSheet("font-size: 32px; color: #FFFFFF;"); title.setAlignment(Qt.AlignCenter); nav_l.addWidget(title)
        subtitle = QLabel("V8.0 HARDWARE RUNTIME"); subtitle.setStyleSheet("color: #00E5FF; font-weight: bold; font-family: 'Consolas';"); subtitle.setAlignment(Qt.AlignCenter); nav_l.addWidget(subtitle)
        nav_l.addSpacing(30)

        self.btn_rec = self.create_btn("⏺️ 切换录制模式", self.go_to_record)
        self.btn_pred = self.create_btn("👁️ 切换推理模式", self.go_to_predict)
        self.btn_voice = self.create_btn("🔊 语音播报:开启", self.toggle_voice)
        self.btn_voice.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;")
        self.btn_ai = self.create_btn("🧠 离线大模型:开启", self.toggle_ai)
        self.btn_ai.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;")

        nav_l.addWidget(self.btn_rec); nav_l.addWidget(self.btn_pred); nav_l.addWidget(self.btn_voice); nav_l.addWidget(self.btn_ai); nav_l.addSpacing(30)

        self.box_rec = QFrame(); bl = QVBoxLayout(self.box_rec); bl.setContentsMargins(0, 0, 0, 0)
        self.txt_act = QLineEdit(); self.txt_act.setPlaceholderText(">> 输入动作标签...")
        self.btn_start = QPushButton("开始录制 (按L键)"); self.btn_start.setStyleSheet("background-color: #FF3366; color: white; border: none;"); self.btn_start.clicked.connect(self.trigger_record)
        self.prog_rec = QProgressBar(); self.prog_rec.setRange(0, SEQUENCE_LENGTH)
        bl.addWidget(QLabel("数据采集配置:")); bl.addWidget(self.txt_act); bl.addWidget(self.btn_start); bl.addWidget(self.prog_rec)
        nav_l.addWidget(self.box_rec); self.box_rec.hide(); nav_l.addStretch(); layout.addWidget(nav)

        cam = QFrame(); cam.setObjectName("CameraFrame"); cl = QVBoxLayout(cam); cl.setContentsMargins(0, 0, 0, 0)
        self.lbl_cam = QLabel("底层引擎初始化中..."); self.lbl_cam.setAlignment(Qt.AlignCenter); self.lbl_cam.setScaledContents(True); self.lbl_cam.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored); cl.addWidget(self.lbl_cam); layout.addWidget(cam, stretch=2)

        right = QFrame(); right.setObjectName("Panel"); right.setFixedWidth(360)
        rl = QVBoxLayout(right); rl.setContentsMargins(20, 20, 20, 20)
        rl.addWidget(QLabel("◆ NPU 结构特征识别"))
        self.lbl_res = QLabel("--"); self.lbl_res.setObjectName("ResultText"); self.lbl_res.setAlignment(Qt.AlignCenter)
        res_glow = QGraphicsDropShadowEffect(self); res_glow.setBlurRadius(20); res_glow.setColor(QColor("#00E5FF")); self.lbl_res.setGraphicsEffect(res_glow); rl.addWidget(self.lbl_res)
        self.lbl_conf = QLabel("CONFIDENCE: 0.0%"); self.lbl_conf.setStyleSheet("color: #8888AA; font-family: 'Consolas';"); self.lbl_conf.setAlignment(Qt.AlignCenter); rl.addWidget(self.lbl_conf)
        rl.addSpacing(15)

        dash = QFrame(); dash.setObjectName("Dashboard"); dl = QVBoxLayout(dash)
        dl.addWidget(QLabel("◆ 系统与硬件监控"))
        self.ui_lbl_mode = QLabel("MODE : PREDICT (NPU+LLM)"); self.ui_lbl_mode.setObjectName("DashText"); dl.addWidget(self.ui_lbl_mode)
        self.ui_lbl_fps = QLabel("FPS  : 0"); self.ui_lbl_fps.setObjectName("DashText"); dl.addWidget(self.ui_lbl_fps)
        self.ui_lbl_status = QLabel("STATE: STANDBY"); self.ui_lbl_status.setObjectName("DashText"); dl.addWidget(self.ui_lbl_status)
        self.ui_lbl_temp = QLabel("TEMP : -- °C"); self.ui_lbl_temp.setObjectName("DashText"); dl.addWidget(self.ui_lbl_temp)
        self.ui_lbl_npu = QLabel("NPU  : --"); self.ui_lbl_npu.setObjectName("DashText"); dl.addWidget(self.ui_lbl_npu)
        rl.addWidget(dash); rl.addSpacing(15)

        rl.addWidget(QLabel("◆ AI 动作识别阈值"))
        self.spin_threshold = QDoubleSpinBox(); self.spin_threshold.setRange(0.00, 1.00); self.spin_threshold.setValue(0.80); self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.valueChanged.connect(lambda v: setattr(self.shared_ai_thresh, 'value', v)); rl.addWidget(self.spin_threshold)

        rl.addWidget(QLabel("◆ 肌电波形监控"))
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setBackground('#000000'); self.plot_widget.setYRange(0, 4096); self.plot_widget.setFixedHeight(140)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('#00FF9D', width=1.5)); rl.addWidget(self.plot_widget)

        rl.addWidget(QLabel("◆ 系统运行日志"))
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True); self.txt_log.setFixedHeight(110); rl.addWidget(self.txt_log)
        layout.addWidget(right)

    def create_btn(self, t, f): b = QPushButton(t); b.clicked.connect(f); return b
    def setup_shortcuts(self): self.shortcut_rec = QShortcut(QKeySequence("L"), self); self.shortcut_rec.activated.connect(self.trigger_record_hotkey)

    def connect_signals(self):
        self.ui_bridge.image_signal.connect(self.update_cam); self.ui_bridge.result_signal.connect(self.handle_res)
        self.ui_bridge.rec_progress_signal.connect(self.prog_rec.setValue); self.ui_bridge.fps_signal.connect(self.update_fps)
        self.ui_bridge.status_signal.connect(self.update_status); self.emg_worker.data_signal.connect(self.update_plot)
        self.llm_worker.result_signal.connect(self.on_llm_result); self.llm_worker.log_signal.connect(self.log)
        self.sys_monitor.sys_signal.connect(self.update_sys_info)

    def update_cam(self, img): h, w, ch = img.shape; self.lbl_cam.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, ch * w, QImage.Format_RGB888)))
    def update_fps(self, fps): self.ui_lbl_fps.setText(f"FPS  : {fps}")
    def update_status(self, status):
        self.ui_lbl_status.setText(f"STATE: {status}")
        self.ui_lbl_status.setObjectName("DashAlert" if "LOST" in status else "DashText")
        self.ui_lbl_status.style().unpolish(self.ui_lbl_status); self.ui_lbl_status.style().polish(self.ui_lbl_status)
    def update_plot(self, full_data, single_val): self.wave_buffer.append(single_val); self.plot_curve.setData(list(self.wave_buffer))
    def update_sys_info(self, temp, npu):
        self.ui_lbl_temp.setText(f"TEMP : {temp}")
        if "°C" in temp:
            try:
                if float(temp.split(' ')[0]) >= 80.0: self.ui_lbl_temp.setStyleSheet("color: #FF3366;")
                else: self.ui_lbl_temp.setStyleSheet("color: #00FF9D;")
            except: pass
        self.ui_lbl_npu.setText(f"NPU  : {npu}")

    def refresh_record_split_button(self):
        split_text = f"DATA SPLIT : {self.record_split.upper()}"
        color = "#00FF9D" if self.record_split == "train" else "#FFD166"
        self.btn_split.setText(split_text)
        self.btn_split.setStyleSheet(f"background-color: #242430; color: {color}; border: 1px solid {color};")

    def trigger_record_hotkey(self):
        if self.box_rec.isVisible():
            self.trigger_record()

    def toggle_record_split(self):
        self.record_split = "test" if self.record_split == "train" else "train"
        self.refresh_record_split_button()
        self.log(f"Record split switched to: {self.record_split}")

    def trigger_record(self):
        name = self.txt_act.text().strip()
        if not name or not self.btn_start.isEnabled():
            return
        self.btn_start.setEnabled(False)
        split_root = dataset_split_dir(DATA_PATH, self.record_split)
        action_path = os.path.join(split_root, name)
        record_seq_idx = next_sequence_index(action_path)
        self.cmd_queue.put(["RECORD_TRIGGER", self.record_split, name, record_seq_idx])
        self.log(f"Start recording [{self.record_split}] sequence: {name}")


    def handle_res(self, act, prob):
        if "REC_DONE" in act:
            self.btn_start.setEnabled(True)
            self.prog_rec.setValue(0)
            self.log("Feature sequence saved")
            return

        now = time.time()
        if act == "--" or "UNKNOWN" in act or act == "IDLE":
            self.lbl_res.setText("--")
            self.action_state.update(None, now)
            if len(self.word_buffer) > 0 and (now - self.last_valid_time > float(RUNTIME_CONFIG["sentence_idle_seconds"])):
                self.log(f"Sentence rebuild: {self.word_buffer}")
                self.llm_worker.translate(self.word_buffer.copy())
                self.word_buffer.clear()
            return

        self.lbl_res.setText(act)
        self.lbl_conf.setText(f"CONFIDENCE: {prob * 100:.1f}%")
        confirmed = self.action_state.update(act, now)
        if confirmed and (not self.word_buffer or self.word_buffer[-1] != confirmed):
            self.word_buffer.append(confirmed)
            self.last_valid_time = now
            self.log(f"Word accepted: [{confirmed}]")

    def on_llm_result(self, sentence): self.log(f"🔊 AI 翻译: {sentence}"); self.voice_worker.speak(sentence)

    def toggle_voice(self):
        self.voice_worker.is_muted = not self.voice_worker.is_muted
        if self.voice_worker.is_muted: self.btn_voice.setText("🔇 语音播报: 关闭"); self.btn_voice.setStyleSheet("background-color: #242430; color: #FF3366; border: 1px solid #FF3366;")
        else: self.btn_voice.setText("🔊 语音播报: 开启"); self.btn_voice.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;")

    def toggle_ai(self):
        if self.llm_worker.toggle_engine():
            self.btn_ai.setText("🧠 离线大模型:开启"); self.btn_ai.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;"); self.ui_lbl_mode.setText("MODE : PREDICT (NPU+LLM)")
        else:
            self.btn_ai.setText("🧠 离线大模型:关闭"); self.btn_ai.setStyleSheet("background-color: #242430; color: #FF3366; border: 1px solid #FF3366;"); self.ui_lbl_mode.setText("MODE : PREDICT (NPU ONLY)")

    def go_to_record(self): self.cmd_queue.put(["RECORD"]); self.ui_lbl_mode.setText("MODE : RECORD DATA"); self.box_rec.show()
    def go_to_predict(self): self.cmd_queue.put(["PREDICT"]); self.ui_lbl_mode.setText("MODE : PREDICT"); self.box_rec.hide()
    def log(self, t): self.txt_log.append(f"> {t}"); self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
    def closeEvent(self, e):
        self.cmd_queue.put(["STOP"]); self.ui_bridge.stop(); self.emg_worker.stop(); self.sys_monitor.stop()
        if self.llm_worker.process: self.llm_worker.process.kill()
        e.accept()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
    app = QApplication(sys.argv)
    win = FusionWindow()
    win.show()
    sys.exit(app.exec_())
