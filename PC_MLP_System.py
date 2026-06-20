# --- START OF FILE onepice_system.py ---

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
import onnxruntime as ort
import queue
from multiprocessing import Process, Queue, Value, Array

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QProgressBar, QFrame,
                             QMessageBox, QSizePolicy, QTextEdit, QDoubleSpinBox,
                             QGraphicsDropShadowEffect, QShortcut)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D  # 🌟 引入了 Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tf2onnx
import dashscope

from gesture_runtime import (
    ActionStateMachine,
    PhraseTranslator,
    SIMPLE_HAND_CONNECTIONS,
    create_master_meta,
    class_count_report,
    dataset_split_dir,
    list_action_names,
    load_runtime_config,
    load_sequence_dataset,
    master_split_dir,
    next_sequence_index,
    stratified_split,
    top1_with_margin,
    normalized_entropy,
    write_json,
)

# 填入你的百炼 API Key7
#sk-b3d3f5bc13744420b02d28cff3507fba
dashscope.api_key = ""

# ================= 🔧 全局配置 =================
DATA_PATH = os.path.join('CNN_Gesture-master/MP_Data_Fusion')
MODEL_PATH_H5 = 'CNN_Gesture-master/action_fusion.h5'
MODEL_PATH_ONNX = 'CNN_Gesture-master/action_fusion.onnx'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CONFIG = load_runtime_config(BASE_DIR)
MASTER_ROOT = os.path.join(BASE_DIR, RUNTIME_CONFIG["master_root"])
BUILD_OUTPUT_ROOT = os.path.join(BASE_DIR, RUNTIME_CONFIG["dataset_output_root"])

SEQUENCE_LENGTH = 30
# 🌟 特征维度升级：单手 = 42(归一化坐标) + 5(弯曲角度) + 4(捏合距离) = 51。双手 = 102。
VISUAL_SIZE = 102
VELOCITY_SIZE = 102
EMG_SIZE = 16
# 🌟 彻底剥离 EMG 输入，专注于纯视觉特征的提纯
INPUT_SIZE = VISUAL_SIZE + VELOCITY_SIZE
SERIAL_BAUDRATE = 115200
# ===============================================

# --- 🎨 全新高级专业控制台样式 ---
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


# ================= 🧠 核心视觉算法 (特征工程升级) =================
def extract_visual_keypoints(results, hand_threshold):
    lh, rh = np.zeros(51), np.zeros(51)  # 新增角度与距离维度
    hand_detected_valid = False

    if results.multi_hand_landmarks:
        for idx, hand_res in enumerate(results.multi_hand_landmarks):
            multi_handedness = results.multi_handedness
            if multi_handedness and len(multi_handedness) > idx:
                if multi_handedness[idx].classification[0].score < hand_threshold: continue
                label = multi_handedness[idx].classification[0].label
                hand_detected_valid = True
            else:
                continue

            # 1. 提取基础坐标点
            pts = np.array([[lm.x, lm.y] for lm in hand_res.landmark])
            base = pts[0]
            rel_pts = pts - base
            mcp_dist = np.linalg.norm(rel_pts[9])  # 手腕到中指根部的距离（用作基准缩放）

            if mcp_dist > 1e-6:
                norm_pts = rel_pts / mcp_dist
            else:
                norm_pts = rel_pts
            features = list(norm_pts.flatten())  # 42 维基础坐标

            # 🌟 2. 注入灵魂：计算手指关节弯曲角度 (0~1归一化)
            def get_angle(p1, p2, p3):
                v1, v2 = p1 - p2, p3 - p2
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6: return 0.0
                return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)) / np.pi

            angles = [
                get_angle(pts[1], pts[2], pts[4]),  # 拇指弯曲度
                get_angle(pts[5], pts[6], pts[8]),  # 食指弯曲度
                get_angle(pts[9], pts[10], pts[12]),  # 中指弯曲度
                get_angle(pts[13], pts[14], pts[16]),  # 无名指弯曲度
                get_angle(pts[17], pts[18], pts[20])  # 小指弯曲度
            ]

            # 🌟 3. 注入灵魂：计算拇指与其他四指的相对距离 (捏合状态感知)
            dists = [
                np.linalg.norm(rel_pts[4] - rel_pts[8]),  # 拇指-食指
                np.linalg.norm(rel_pts[4] - rel_pts[12]),  # 拇指-中指
                np.linalg.norm(rel_pts[4] - rel_pts[16]),  # 拇指-无名指
                np.linalg.norm(rel_pts[4] - rel_pts[20])  # 拇指-小指
            ]
            if mcp_dist > 1e-6: dists = [d / mcp_dist for d in dists]  # 缩放不变性处理

            features.extend(angles)  # 扩充 5 维
            features.extend(dists)  # 扩充 4 维

            if label == 'Left':
                lh = np.array(features)
            else:
                rh = np.array(features)

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


# ================= 🧵 进程与线程工作区 =================

class EMGWorker(QThread):
    data_signal = pyqtSignal(np.ndarray, float)

    def __init__(self, shared_emg_array):
        super().__init__()
        self.running = True
        self.ser = None
        self.simulation_mode = False
        self.HEADER, self.PACKET_SIZE = b'\x55\xAA', 34
        self.buffer = bytearray()
        self.moving_baseline = np.zeros(EMG_SIZE)
        self.ema_envelope = np.zeros(EMG_SIZE)
        self.is_init = False
        self.shared_emg = shared_emg_array

    def auto_connect(self):
        try:
            self.ser = serial.Serial('COM15', SERIAL_BAUDRATE, timeout=0.01)
            print(">>> 肌电串口连接成功 (已转为仅监听模式)")
            self.simulation_mode = False
        except Exception:
            self.simulation_mode = True

    def run(self):
        self.auto_connect()
        while self.running:
            if self.simulation_mode:
                time.sleep(0.02)
            else:
                if self.ser and self.ser.is_open:
                    try:
                        if self.ser.in_waiting:
                            self.buffer.extend(self.ser.read(self.ser.in_waiting))
                        while len(self.buffer) >= self.PACKET_SIZE:
                            header_index = self.buffer.find(self.HEADER)
                            if header_index == -1:
                                self.buffer = self.buffer[-2:]
                                break
                            if header_index > 0:
                                self.buffer = self.buffer[header_index:]
                                continue
                            if len(self.buffer) < self.PACKET_SIZE: break

                            packet = self.buffer[:self.PACKET_SIZE]
                            self.buffer = self.buffer[self.PACKET_SIZE:]
                            try:
                                data_tuple = struct.unpack('<16H', packet[2:])
                                raw_data = np.array(data_tuple, dtype=float)

                                if not self.is_init:
                                    self.moving_baseline = raw_data.copy()
                                    self.is_init = True
                                    continue

                                self.moving_baseline = self.moving_baseline * 0.98 + raw_data * 0.02
                                rectified_data = np.abs(raw_data - self.moving_baseline)

                                for i in range(EMG_SIZE):
                                    if rectified_data[i] > self.ema_envelope[i]:
                                        self.ema_envelope[i] = rectified_data[i] * 0.4 + self.ema_envelope[i] * 0.6
                                    else:
                                        self.ema_envelope[i] = rectified_data[i] * 0.05 + self.ema_envelope[i] * 0.95

                                for i in range(EMG_SIZE): self.shared_emg[i] = self.ema_envelope[i]
                                self.data_signal.emit(self.ema_envelope, self.ema_envelope[0])
                            except Exception:
                                pass
                    except Exception:
                        time.sleep(1)
                        self.auto_connect()

    def stop(self):
        self.running = False
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
        #手部检测和追踪的置信度均默认设为了0.5

        hands = mp_hands.Hands(model_complexity=int(RUNTIME_CONFIG["mediapipe_model_complexity"]), min_detection_confidence=0.5, min_tracking_confidence=0.5)

        model, actions = None, []
        if os.path.exists(MODEL_PATH_ONNX):
            try:
                model = ort.InferenceSession(MODEL_PATH_ONNX)
                input_name = model.get_inputs()[0].name
                if os.path.exists(DATA_PATH):
                    actions = list_action_names(DATA_PATH)
            except Exception as e:
                print(f"ONNX 加载失败: {e}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        mode, record_split, record_action, record_seq_idx = "PREDICT", "train", "", 0
        is_recording, frame_counter_rec = False, 0
        record_video_writer, record_video_path, record_sample_path = None, "", ""
        sequence = []
        prev_visual_data = np.zeros(VISUAL_SIZE)
        pred_history = collections.deque(maxlen=int(RUNTIME_CONFIG["prediction_smoothing"]))
        prev_time = time.time()
        running = True

        while running and cap.isOpened():
            try:
                cmd = self.cmd_queue.get_nowait()
                if cmd[0] == "STOP":
                    running = False
                elif cmd[0] == "PREDICT":
                    mode = "PREDICT"
                    pred_history.clear()
                    if os.path.exists(MODEL_PATH_ONNX):
                        model = ort.InferenceSession(MODEL_PATH_ONNX)
                        input_name = model.get_inputs()[0].name
                        actions = list_action_names(DATA_PATH)
                elif cmd[0] == "RECORD_TRIGGER":
                    mode = "RECORD"
                    record_split, record_action, record_seq_idx = cmd[1], cmd[2], cmd[3]
                    is_recording, frame_counter_rec = True, 0
                    record_sample_path = os.path.join(master_split_dir(MASTER_ROOT, record_split), record_action, str(record_seq_idx))
                    os.makedirs(record_sample_path, exist_ok=True)
                    record_video_path = os.path.join(record_sample_path, "capture.mp4")
                    record_video_writer = None
            except queue.Empty:
                pass

            ret, frame = cap.read()
            if not ret: break

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0)
            prev_time = curr_time

            frame_flipped = cv2.flip(frame, 1)
            display_image = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            small_image = cv2.resize(display_image, (int(RUNTIME_CONFIG["process_width"]), int(RUNTIME_CONFIG["process_height"])))

            results = hands.process(small_image)
            visual_data = np.zeros(VISUAL_SIZE)
            velocity_data = np.zeros(VELOCITY_SIZE)
            hand_detected_valid = False
            status_msg = "TARGET LOST"

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

            if not hand_detected_valid: prev_visual_data = np.zeros(VISUAL_SIZE)

            # 🌟 纯视觉融合 (剔除 EMG)
            fusion_data = np.concatenate([visual_data, velocity_data])

            action_res, prob_res = "--", 0.0
            rec_progress, rec_done_msg = -1, None

            if mode == "RECORD" and is_recording:
                if hand_detected_valid:
                    if frame_counter_rec == 0:
                        if record_video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            record_video_writer = cv2.VideoWriter(record_video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    if record_video_writer is not None:
                        record_video_writer.write(frame_flipped)
                    frame_counter_rec += 1
                    rec_progress = frame_counter_rec
                    if frame_counter_rec == SEQUENCE_LENGTH:
                        if record_video_writer is not None:
                            record_video_writer.release()
                            record_video_writer = None
                        meta = create_master_meta(record_action, record_split, "PC", record_video_path, frame_counter_rec, 20.0)
                        write_json(os.path.join(record_sample_path, "meta.json"), meta)
                        is_recording, mode = False, "IDLE"
                        rec_done_msg = f"REC_DONE: {record_seq_idx}"

            elif mode == "PREDICT" and model is not None:
                if hand_detected_valid:
                    sequence.append(fusion_data)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                        outputs = model.run(None, {input_name: input_data})
                        pred_history.append(outputs[0][0])
                        smooth_res = np.mean(pred_history, axis=0)
                        curr_idx, top_prob, margin = top1_with_margin(smooth_res)
                        entropy = normalized_entropy(smooth_res)
                        if (
                            top_prob > self.shared_ai_thresh.value
                            and margin >= float(RUNTIME_CONFIG["margin_threshold"])
                            and entropy <= float(RUNTIME_CONFIG.get("entropy_threshold", 0.72))
                        ):
                            action_res = str(actions[curr_idx]) if len(actions) > curr_idx else "UNKNOWN"
                            prob_res = float(top_prob)
                else:
                    sequence = []
                    pred_history.clear()

            if not self.out_queue.full():
                self.out_queue.put({
                    'image': display_image, 'fps': fps, 'status': status_msg,
                    'action': action_res, 'prob': prob_res,
                    'rec_progress': rec_progress, 'rec_done': rec_done_msg
                })
        cap.release()
        if record_video_writer is not None:
            record_video_writer.release()


class UIBridge(QThread):
    image_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, float)
    rec_progress_signal = pyqtSignal(int)
    fps_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)

    def __init__(self, out_queue):
        super().__init__()
        self.out_queue = out_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.out_queue.get(timeout=0.1)
                self.image_signal.emit(data['image'])
                self.fps_signal.emit(data['fps'])
                self.status_signal.emit(data['status'])
                self.result_signal.emit(data['action'], data['prob'])

                if data['rec_progress'] != -1: self.rec_progress_signal.emit(data['rec_progress'])
                if data['rec_done']: self.result_signal.emit(data['rec_done'], 1.0)
            except queue.Empty:
                pass

    def stop(self):
        self.running = False
        self.wait()


class VoiceWorker(QThread):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.is_muted = False

    def speak(self, text):
        if not self.is_muted:
            self.text = text
            self.start()

    def run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(self.text)
            engine.runAndWait()
        except:
            pass


class LLMWorker(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.words = []
        self.translator = PhraseTranslator(BASE_DIR)

    def translate(self, words):
        self.words = words
        self.start()

    def run(self):
        if not self.words:
            return
        cached = self.translator.lookup(self.words)
        if cached:
            self.result_signal.emit(cached)
            return
        fallback_text = self.translator.fallback(self.words)
        words_str = "|".join(self.words)
        prompt = (
            "You are a strict Chinese sign-language sentence formatter. "
            "Only reorder the given words and add necessary Chinese particles or punctuation. "
            "Do not add any meaning that is not present in the input. "
            f"Input words: [{words_str}]. Output one Chinese sentence only."
        )
        try:
            response = dashscope.Generation.call(model='qwen-turbo', prompt=prompt, temperature=0.01, top_p=0.1)
            if response.status_code == 200:
                sentence = response.output.text.strip()
                self.translator.remember(self.words, sentence)
                self.result_signal.emit(sentence)
            else:
                self.result_signal.emit(fallback_text)
        except Exception:
            self.result_signal.emit(fallback_text)


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    started_signal = pyqtSignal(int)

    def build_model(self, num_classes):
        model = Sequential(name="gesture_tcn_light")
        model.add(Conv1D(96, 3, padding="same", activation="relu", input_shape=(SEQUENCE_LENGTH, INPUT_SIZE)))
        model.add(BatchNormalization())
        model.add(Conv1D(96, 3, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(128, 5, padding="same", activation="relu"))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.20))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        return model

    def run(self):
        try:
            self.had_error = False
            max_epochs = 150
            self.started_signal.emit(max_epochs)
            self.log_signal.emit(">>> Loading gesture dataset...")
            preferred_root = BUILD_OUTPUT_ROOT if os.path.exists(BUILD_OUTPUT_ROOT) else DATA_PATH
            train_data_path = dataset_split_dir(preferred_root, "train")
            data_path = train_data_path if os.path.exists(train_data_path) else preferred_root
            if not os.path.exists(data_path):
                self.log_signal.emit(f">>> Dataset path missing: {DATA_PATH}")
                self.finished_signal.emit()
                return

            X, labels, actions, bad = load_sequence_dataset(data_path, SEQUENCE_LENGTH, INPUT_SIZE)
            if len(actions) < 2 or len(X) < 2:
                self.log_signal.emit(">>> Need at least 2 classes and 2 valid sequences.")
                self.finished_signal.emit()
                return

            self.log_signal.emit(f">>> Valid sequences: {len(X)} | classes: {len(actions)}")
            self.log_signal.emit(f">>> Per-class counts: {class_count_report(labels, actions)}")
            if bad:
                self.log_signal.emit(f">>> Skipped invalid sequences: {len(bad)}")

            train_idx, val_idx, test_idx = stratified_split(labels, train_ratio=0.70, val_ratio=0.15, seed=42)
            y = to_categorical(labels, num_classes=len(actions)).astype(np.float32)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = self.build_model(len(actions))

            class ProgressCallback(Callback):
                def __init__(self, worker):
                    self.worker = worker
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    self.worker.progress_signal.emit(epoch + 1)
                    acc = logs.get("categorical_accuracy", 0.0)
                    val_acc = logs.get("val_categorical_accuracy", 0.0)
                    self.worker.log_signal.emit(f">>> epoch {epoch + 1}: train={acc:.3f}, val={val_acc:.3f}")

            callbacks = [
                ProgressCallback(self),
                EarlyStopping(monitor="val_categorical_accuracy", patience=25, restore_best_weights=True, mode="max"),
                ModelCheckpoint(MODEL_PATH_H5, monitor="val_categorical_accuracy", save_best_only=True, mode="max"),
            ]

            self.log_signal.emit(">>> Training light Conv1D temporal model...")
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if len(X_val) else None,
                epochs=max_epochs,
                batch_size=16,
                verbose=0,
                callbacks=callbacks,
            )

            if os.path.exists(MODEL_PATH_H5):
                model = load_model(MODEL_PATH_H5)
            else:
                model.save(MODEL_PATH_H5)

            if len(X_test):
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                self.log_signal.emit(f">>> Held-out test accuracy: {acc:.3f} loss={loss:.3f}")
            else:
                self.log_signal.emit(">>> No held-out test split available; collect more data per class.")

            self.log_signal.emit(">>> Exporting ONNX for RK3588 conversion...")
            spec = (tf.TensorSpec((None, SEQUENCE_LENGTH, INPUT_SIZE), tf.float32, name="mlp_input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
            with open(MODEL_PATH_ONNX, "wb") as f:
                f.write(onnx_model.SerializeToString())
            self.log_signal.emit(f">>> ONNX saved: {MODEL_PATH_ONNX}")
        except Exception as e:
            self.had_error = True
            self.error_signal.emit(str(e))
            self.log_signal.emit(f">>> Training failed: {e}")
        self.finished_signal.emit()


# ================= 🖥️ 主窗口 =================
class FusionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEUROLINK V8.0 - RK3588 NPU 极速架构")
        self.resize(1400, 850)
        self.setStyleSheet(PRO_STYLE)

        self.wave_buffer = collections.deque([2048] * 500, maxlen=500)
        self.cmd_queue = Queue()
        self.out_queue = Queue(maxsize=2)
        self.shared_emg = Array('d', EMG_SIZE)
        self.shared_ai_thresh = Value('f', float(RUNTIME_CONFIG["confidence_threshold"]))
        self.shared_hand_thresh = Value('f', 0.50)

        self.emg_worker = EMGWorker(self.shared_emg)
        self.train_worker = TrainWorker()
        self.voice_worker = VoiceWorker()
        self.llm_worker = LLMWorker()
        self.ui_bridge = UIBridge(self.out_queue)

        self.vision_process = VisionProcess(
            self.cmd_queue, self.out_queue,
            self.shared_ai_thresh, self.shared_hand_thresh
        )

        self.word_buffer = []
        self.action_state = ActionStateMachine(RUNTIME_CONFIG["confirm_frames"], RUNTIME_CONFIG["cooldown_seconds"])
        self.last_action, self.locked_action = "", ""
        self.last_valid_time, self.last_speak_time, self.action_confirm_count = time.time(), 0, 0
        self.record_split = str(RUNTIME_CONFIG.get("record_split", "train")).lower()

        self.init_ui()
        self.refresh_record_split_button()
        self.connect_signals()
        self.setup_shortcuts()

        self.emg_worker.start()
        self.vision_process.start()
        self.ui_bridge.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        nav = QFrame()
        nav.setObjectName("Panel")
        nav.setFixedWidth(280)
        nav_l = QVBoxLayout(nav)
        nav_l.setContentsMargins(20, 20, 20, 20)

        title = QLabel("NEUROLINK")
        title.setObjectName("ResultText")
        title.setStyleSheet("font-size: 32px; color: #FFFFFF;")
        title.setAlignment(Qt.AlignCenter)
        nav_l.addWidget(title)

        subtitle = QLabel("V8.0 NPU-MLP ENGINE")
        subtitle.setStyleSheet("color: #00E5FF; font-weight: bold; font-family: 'Consolas';")
        subtitle.setAlignment(Qt.AlignCenter)
        nav_l.addWidget(subtitle)
        nav_l.addSpacing(30)

        self.btn_rec = self.create_btn("⏺️ 切换录制模式", self.go_to_record)
        self.btn_train = self.create_btn("⚡ 切换超速训练", self.go_to_train)
        self.btn_pred = self.create_btn("👁️ 切换推理模式", self.go_to_predict)

        self.btn_voice = self.create_btn("🔊 语音播报:开启", self.toggle_voice)
        self.btn_voice.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;")

        nav_l.addWidget(self.btn_rec)
        nav_l.addWidget(self.btn_train)
        nav_l.addWidget(self.btn_pred)
        nav_l.addWidget(self.btn_voice)
        nav_l.addSpacing(30)

        self.box_rec = QFrame()
        bl = QVBoxLayout(self.box_rec)
        bl.setContentsMargins(0, 0, 0, 0)
        self.txt_act = QLineEdit()
        self.txt_act.setPlaceholderText(">> 输入动作标签...")
        self.btn_start = QPushButton("开始录制 (按L键)")
        self.btn_start.setStyleSheet("background-color: #FF3366; color: white; border: none;")
        self.btn_start.clicked.connect(self.trigger_record)
        self.btn_split = QPushButton()
        self.btn_split.clicked.connect(self.toggle_record_split)
        self.prog_rec = QProgressBar()
        self.prog_rec.setRange(0, SEQUENCE_LENGTH)
        bl.addWidget(QLabel("数据采集配置 (包含结构特征):"))
        bl.addWidget(self.txt_act)
        bl.addWidget(self.btn_split)
        bl.addWidget(self.btn_start)
        bl.addWidget(self.prog_rec)
        nav_l.addWidget(self.box_rec)
        self.box_rec.hide()

        self.box_train = QFrame()
        bt = QVBoxLayout(self.box_train)
        bt.setContentsMargins(0, 0, 0, 0)
        self.btn_run = QPushButton(">> 启动 MLP 极速训练")
        self.btn_run.setStyleSheet("background-color: #00FF9D; color: black; border: none;")
        self.btn_run.clicked.connect(self.start_training)
        self.prog_train = QProgressBar()
        self.prog_train.setRange(0, 150)
        bt.addWidget(self.btn_run)
        bt.addWidget(self.prog_train)
        nav_l.addWidget(self.box_train)
        self.box_train.hide()
        nav_l.addStretch()
        layout.addWidget(nav)

        cam = QFrame()
        cam.setObjectName("CameraFrame")
        cl = QVBoxLayout(cam)
        cl.setContentsMargins(0, 0, 0, 0)
        self.lbl_cam = QLabel("底层引擎初始化中...")
        self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setScaledContents(True)
        self.lbl_cam.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        cl.addWidget(self.lbl_cam)
        layout.addWidget(cam, stretch=2)

        right = QFrame()
        right.setObjectName("Panel")
        right.setFixedWidth(360)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 20, 20, 20)

        rl.addWidget(QLabel("◆ NPU 结构特征识别"))
        self.lbl_res = QLabel("--")
        self.lbl_res.setObjectName("ResultText")
        self.lbl_res.setAlignment(Qt.AlignCenter)

        res_glow = QGraphicsDropShadowEffect(self)
        res_glow.setBlurRadius(20)
        res_glow.setColor(QColor("#00E5FF"))
        self.lbl_res.setGraphicsEffect(res_glow)
        rl.addWidget(self.lbl_res)

        self.lbl_conf = QLabel("CONFIDENCE: 0.0%")
        self.lbl_conf.setStyleSheet("color: #8888AA; font-family: 'Consolas';")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        rl.addWidget(self.lbl_conf)
        rl.addSpacing(15)

        dash = QFrame()
        dash.setObjectName("Dashboard")
        dl = QVBoxLayout(dash)
        dl.addWidget(QLabel("◆ 系统运行监控"))
        self.ui_lbl_mode = QLabel("MODE : PREDICT")
        self.ui_lbl_mode.setObjectName("DashText")
        dl.addWidget(self.ui_lbl_mode)
        self.ui_lbl_fps = QLabel("FPS  : 0")
        self.ui_lbl_fps.setObjectName("DashText")
        dl.addWidget(self.ui_lbl_fps)
        self.ui_lbl_status = QLabel("STATE: STANDBY")
        self.ui_lbl_status.setObjectName("DashText")
        dl.addWidget(self.ui_lbl_status)
        rl.addWidget(dash)
        rl.addSpacing(15)

        rl.addWidget(QLabel("◆ AI 动作识别阈值"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.00, 1.00)
        self.spin_threshold.setValue(float(RUNTIME_CONFIG["confidence_threshold"]))
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.valueChanged.connect(lambda v: setattr(self.shared_ai_thresh, 'value', v))
        rl.addWidget(self.spin_threshold)

        rl.addWidget(QLabel("◆ 肌电监控 (已从 AI 输入剥离)"))
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#000000')
        self.plot_widget.setYRange(0, 4096)
        self.plot_widget.setFixedHeight(140)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('#00FF9D', width=1.5))
        rl.addWidget(self.plot_widget)

        rl.addWidget(QLabel("◆ 控制台终端日志"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(110)
        rl.addWidget(self.txt_log)
        layout.addWidget(right)

    def create_btn(self, t, f):
        b = QPushButton(t)
        b.clicked.connect(f)
        return b

    def setup_shortcuts(self):
        self.shortcut_rec = QShortcut(QKeySequence("L"), self)
        self.shortcut_rec.activated.connect(self.trigger_record_hotkey)

    def connect_signals(self):
        self.ui_bridge.image_signal.connect(self.update_cam)
        self.ui_bridge.result_signal.connect(self.handle_res)
        self.ui_bridge.rec_progress_signal.connect(self.prog_rec.setValue)
        self.ui_bridge.fps_signal.connect(self.update_fps)
        self.ui_bridge.status_signal.connect(self.update_status)
        self.train_worker.log_signal.connect(self.log)
        self.train_worker.progress_signal.connect(self.prog_train.setValue)
        self.train_worker.started_signal.connect(self.on_training_started)
        self.train_worker.finished_signal.connect(self.on_training_finished)
        self.train_worker.error_signal.connect(self.on_training_error)
        self.emg_worker.data_signal.connect(self.update_plot)
        self.llm_worker.result_signal.connect(self.on_llm_result)

    def update_cam(self, img):
        h, w, ch = img.shape
        self.lbl_cam.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, ch * w, QImage.Format_RGB888)))

    def update_fps(self, fps):
        self.ui_lbl_fps.setText(f"FPS  : {fps}")

    def update_status(self, status):
        self.ui_lbl_status.setText(f"STATE: {status}")
        self.ui_lbl_status.setObjectName("DashAlert" if "LOST" in status else "DashText")
        self.ui_lbl_status.style().unpolish(self.ui_lbl_status)
        self.ui_lbl_status.style().polish(self.ui_lbl_status)

    def update_plot(self, full_data, single_val):
        self.wave_buffer.append(single_val)
        self.plot_curve.setData(list(self.wave_buffer))

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


    def start_training(self):
        if self.train_worker.isRunning():
            self.log("Training is already running")
            return
        self.btn_run.setEnabled(False)
        self.prog_train.setValue(0)
        self.log("Training started")
        self.train_worker.start()

    def on_training_started(self, max_epochs):
        self.prog_train.setRange(0, int(max_epochs))

    def on_training_finished(self):
        self.btn_run.setEnabled(True)
        if not getattr(self.train_worker, "had_error", False):
            self.log("Training finished")
            QMessageBox.information(self, "SYS", "MLP training and ONNX export complete")

    def on_training_error(self, message):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "TRAIN ERROR", message)


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
            if len(self.word_buffer) > 0:
                min_words = int(RUNTIME_CONFIG.get("sentence_min_words", 2))
                idle_seconds = float(RUNTIME_CONFIG["sentence_idle_seconds"])
                if len(self.word_buffer) < min_words:
                    idle_seconds = float(RUNTIME_CONFIG.get("single_word_idle_seconds", idle_seconds))
                if now - self.last_valid_time > idle_seconds:
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

    def on_llm_result(self, sentence):
        self.log(f"🔊 AI 造句输出: {sentence}")
        self.voice_worker.speak(sentence)

    def toggle_voice(self):
        self.voice_worker.is_muted = not self.voice_worker.is_muted
        if self.voice_worker.is_muted:
            self.btn_voice.setText("🔇 语音播报: 关闭")
            self.btn_voice.setStyleSheet("background-color: #242430; color: #FF3366; border: 1px solid #FF3366;")
        else:
            self.btn_voice.setText("🔊 语音播报: 开启")
            self.btn_voice.setStyleSheet("background-color: #242430; color: #00FF9D; border: 1px solid #00FF9D;")

    def go_to_record(self):
        self.cmd_queue.put(["RECORD"])
        self.ui_lbl_mode.setText("MODE : RECORD DATA")
        self.box_rec.show()
        self.box_train.hide()

    def go_to_train(self):
        self.cmd_queue.put(["IDLE"])
        self.ui_lbl_mode.setText("MODE : NEURAL TRAIN")
        self.box_rec.hide()
        self.box_train.show()

    def go_to_predict(self):
        self.cmd_queue.put(["PREDICT"])
        self.ui_lbl_mode.setText("MODE : PREDICT (ONNX)")
        self.box_rec.hide()
        self.box_train.hide()

    def log(self, t):
        self.txt_log.append(f"> {t}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def closeEvent(self, e):
        self.cmd_queue.put(["STOP"])
        self.ui_bridge.stop()
        self.emg_worker.stop()
        self.train_worker.wait()
        e.accept()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
    app = QApplication(sys.argv)
    win = FusionWindow()
    win.show()
    sys.exit(app.exec_())
