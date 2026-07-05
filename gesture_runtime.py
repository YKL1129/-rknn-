import json
import os
import time
from collections import Counter

import numpy as np


DEFAULT_RUNTIME_CONFIG = {
    "confidence_threshold": 0.80,
    "margin_threshold": 0.15,
    "entropy_threshold": 1.55,
    "confirm_frames": 4,
    "cooldown_seconds": 0.70,
    "sentence_idle_seconds": 1.20,
    "single_word_idle_seconds": 2.80,
    "sentence_min_words": 2,
    "sentence_merge_window_seconds": 1.00,
    "mediapipe_model_complexity": 0,
    "camera_width": 640,
    "camera_height": 480,
    "process_width": 320,
    "process_height": 240,
    "draw_landmarks": True,
    "draw_full_landmarks": False,
    "prediction_smoothing": 5,
    "sequence_length": 30,
    "input_size": 204,
    "record_split": "train",
    "master_root": "Gesture_Master",
    "dataset_output_root": "MP_Data_Build",
    "feature_spec": "fusion_v1_204",
    "emg_transport": "auto",
    "serial_port": "",
    "spi_device": "/dev/spidev4.0",
    "spi_mode": 0,
    "spi_speed_hz": 1000000,
    "spi_bits_per_word": 8,
    "spi_packet_size": 34,
    "spi_poll_interval_ms": 5,
    "voice_backend_preference": "pyttsx3",
    "voice_player_preference": "ffplay",
    "voice_edge_tts_voice": "zh-CN-XiaoxiaoNeural",
    "voice_output_mode": "recognized_word",
    "voice_cooldown_seconds": 1.5,
}


def load_runtime_config(base_dir):
    path = os.path.join(base_dir, "runtime_config.json")
    config = DEFAULT_RUNTIME_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            for key, value in user_config.items():
                if key in config:
                    config[key] = value
        except Exception:
            pass
    return config


def dataset_split_dir(data_path, split_name):
    split_name = str(split_name).strip().lower()
    if split_name in ("train", "test", "val"):
        return os.path.join(data_path, split_name)
    return data_path


def master_split_dir(master_root, split_name):
    split_name = str(split_name).strip().lower()
    if split_name not in ("train", "test", "val"):
        split_name = "train"
    return os.path.join(master_root, split_name)


def list_action_names(data_path):
    action_root = dataset_split_dir(data_path, "train")
    if not os.path.exists(action_root):
        action_root = data_path
    if not os.path.exists(action_root):
        return np.array([])
    reserved_dirs = {"train", "test", "val"}
    return np.array(sorted([
        f for f in os.listdir(action_root)
        if os.path.isdir(os.path.join(action_root, f)) and f not in reserved_dirs
    ]))


def next_sequence_index(action_root):
    if not os.path.exists(action_root):
        os.makedirs(action_root, exist_ok=True)
        return 0
    dirs = [d for d in os.listdir(action_root) if os.path.isdir(os.path.join(action_root, d))]
    nums = [int(d) for d in dirs if str(d).isdigit()]
    return max(nums) + 1 if nums else 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path, default=None):
    default = {} if default is None else default
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_master_meta(label, split, source_device, video_path, frame_count=0, capture_fps=0.0):
    return {
        "标签": str(label),
        "分区": str(split),
        "来源设备": str(source_device),
        "视频路径": str(video_path),
        "采集帧率": float(capture_fps),
        "总帧数": int(frame_count),
        "录制时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def load_manifest(manifest_path):
    data = read_json(manifest_path, default={"样本列表": []})
    if "样本列表" not in data or not isinstance(data["样本列表"], list):
        data = {"样本列表": []}
    return data


def append_manifest_entry(manifest_path, entry):
    data = load_manifest(manifest_path)
    data["样本列表"].append(entry)
    write_json(manifest_path, data)


def build_manifest_entry(sample_id, label, split, source_type, source_path, output_path, feature_spec, status, reason=""):
    return {
        "样本编号": str(sample_id),
        "标签": str(label),
        "分区": str(split),
        "来源类型": str(source_type),
        "来源路径": str(source_path),
        "输出路径": str(output_path),
        "特征方案": str(feature_spec),
        "状态": str(status),
        "失败原因": str(reason),
    }


def top1_with_margin(probs):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return -1, 0.0, 0.0
    order = np.argsort(probs)
    best = int(order[-1])
    best_prob = float(probs[best])
    second_prob = float(probs[order[-2]]) if probs.size > 1 else 0.0
    return best, best_prob, best_prob - second_prob


def normalized_entropy(probs):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size <= 1:
        return 0.0
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(float(probs.size)))


SIMPLE_HAND_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


class ActionStateMachine:
    def __init__(self, confirm_frames=4, cooldown_seconds=0.70):
        self.confirm_frames = int(confirm_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self.state = "IDLE"
        self.candidate = None
        self.count = 0
        self.cooldown_until = 0.0
        self.last_confirmed = None

    def reset(self):
        self.state = "IDLE"
        self.candidate = None
        self.count = 0

    def update(self, action, now=None):
        now = time.time() if now is None else now
        if action in (None, "", "--", "IDLE"):
            if self.state != "COOLDOWN":
                self.reset()
            elif now >= self.cooldown_until:
                self.reset()
            return None

        if self.state == "COOLDOWN":
            if now < self.cooldown_until:
                return None
            self.reset()

        if self.candidate == action:
            self.count += 1
        else:
            self.candidate = action
            self.count = 1
            self.state = "CANDIDATE"

        if self.count >= self.confirm_frames:
            self.state = "COOLDOWN"
            self.cooldown_until = now + self.cooldown_seconds
            self.last_confirmed = action
            self.candidate = None
            self.count = 0
            return action
        return None


def normalize_words(words):
    result = []
    for word in words:
        word = str(word).strip()
        if not word or word == "IDLE" or word == "--":
            continue
        if result and result[-1] == word:
            continue
        result.append(word)
    return result


class PhraseTranslator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.template_path = os.path.join(base_dir, "phrase_templates.json")
        self.cache_path = os.path.join(base_dir, "phrase_cache.json")
        self.templates = self._load_json(self.template_path, [])
        self.cache = self._load_json(self.cache_path, {})

    def _load_json(self, path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def key(self, words):
        return "|".join(normalize_words(words))

    def fallback(self, words):
        return "".join(normalize_words(words))

    def lookup(self, words):
        words = normalize_words(words)
        if not words:
            return ""
        key = self.key(words)
        if key in self.cache:
            return self.cache[key]
        for item in self.templates:
            if item.get("words") == words:
                sentence = item.get("sentence", "")
                if sentence:
                    self.cache[key] = sentence
                    self._save_cache()
                    return sentence
        return ""

    def remember(self, words, sentence):
        words = normalize_words(words)
        sentence = str(sentence).strip()
        if words and sentence:
            self.cache[self.key(words)] = sentence
            self._save_cache()


def load_sequence_dataset(data_path, sequence_length=30, input_size=204, split=None):
    root_path = dataset_split_dir(data_path, split) if split else data_path
    if not os.path.exists(root_path):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64), np.array([]), []

    reserved_dirs = {"train", "test", "val"}
    actions = np.array(sorted([
        f for f in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, f)) and f not in reserved_dirs
    ]))
    sequences, labels, bad = [], [], []
    label_map = {label: idx for idx, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(root_path, action)
        seq_dirs = sorted(
            [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))],
            key=lambda x: int(x) if x.isdigit() else x
        )
        for seq in seq_dirs:
            window, valid = [], True
            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_path, seq, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    valid = False
                    break
                try:
                    frame_data = np.load(frame_path).astype(np.float32)
                    if frame_data.shape != (input_size,):
                        valid = False
                        break
                    window.append(frame_data)
                except Exception:
                    valid = False
                    break
            if valid and len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                bad.append(os.path.join(action, seq))

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), actions, bad


def stratified_split(labels, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n > 1:
            n_train = max(1, n - 1)
            n_val = 0 if n == 2 else 1
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def class_count_report(labels, actions):
    counts = Counter(labels.tolist())
    return ", ".join(f"{actions[i]}:{counts.get(i, 0)}" for i in range(len(actions)))
