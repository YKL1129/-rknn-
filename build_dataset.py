import argparse
import os
import shutil

import cv2
import mediapipe as mp
import numpy as np

from gesture_runtime import (
    append_manifest_entry,
    build_manifest_entry,
    dataset_split_dir,
    ensure_dir,
    list_action_names,
    load_manifest,
    load_runtime_config,
    master_split_dir,
    read_json,
    write_json,
)


SEQUENCE_LENGTH = 30
VISUAL_SIZE = 102
VELOCITY_SIZE = 102
INPUT_SIZE = VISUAL_SIZE + VELOCITY_SIZE
RUNTIME_CONFIG = load_runtime_config(os.path.dirname(os.path.abspath(__file__)))

ZH = {
    "desc": "\u6839\u636e\u89c6\u9891\u6bcd\u7248\u6784\u5efa\u8bad\u7ec3\u96c6\u548c\u6d4b\u8bd5\u96c6\u7279\u5f81\u6570\u636e",
    "master_root": "\u6bcd\u7248\u76ee\u5f55",
    "legacy_root": "\u65e7\u6570\u636e\u76ee\u5f55",
    "output_root": "\u8f93\u51fa\u76ee\u5f55",
    "feature_spec": "\u7279\u5f81\u65b9\u6848",
    "split": "\u6570\u636e\u5206\u533a",
    "rebuild": "\u91cd\u5efa\u5168\u90e8",
    "incremental": "\u4ec5\u589e\u91cf",
    "hand_threshold": "\u624b\u90e8\u9608\u503c",
    "scan_master": "\u6b63\u5728\u626b\u63cf\u6bcd\u7248\u76ee\u5f55",
    "found_samples": "\u5171\u53d1\u73b0\u6bcd\u7248\u6837\u672c",
    "register_legacy": "\u6b63\u5728\u767b\u8bb0\u65e7\u7279\u5f81\u6570\u636e",
    "rebuild_all": "\u6b63\u5728\u91cd\u5efa\u5168\u90e8\u8f93\u51fa\u76ee\u5f55",
    "success": "\u6210\u529f",
    "failed": "\u5931\u8d25",
    "complete": "\u6784\u5efa\u5b8c\u6210",
    "success_count": "\u6210\u529f\u6837\u672c\u6570",
    "fail_count": "\u5931\u8d25\u6837\u672c\u6570",
    "label_table": "\u5f53\u524d\u7c7b\u522b\u8868",
    "output_dir": "\u8f93\u51fa\u76ee\u5f55",
    "manifest": "\u6e05\u5355\u6587\u4ef6",
    "skip_exists": "\u8df3\u8fc7",
    "missing_video": "\u7f3a\u5c11 capture.mp4",
    "open_video_failed": "\u65e0\u6cd5\u6253\u5f00\u89c6\u9891",
    "insufficient_frames": "\u5e27\u6570\u4e0d\u8db3\uff0c\u81f3\u5c11\u9700\u8981",
    "sample_ok": "\u6210\u529f",
    "sample_fail": "\u5931\u8d25",
}


def zh(msg):
    return msg.encode("utf-8").decode("utf-8")


def log(message):
    print(message)


def extract_visual_keypoints(results, hand_threshold):
    lh, rh = np.zeros(51), np.zeros(51)
    hand_detected_valid = False
    if results.multi_hand_landmarks:
        for idx, hand_res in enumerate(results.multi_hand_landmarks):
            multi_handedness = results.multi_handedness
            if multi_handedness and len(multi_handedness) > idx:
                if multi_handedness[idx].classification[0].score < hand_threshold:
                    continue
                label = multi_handedness[idx].classification[0].label
                hand_detected_valid = True
            else:
                continue

            pts = np.array([[lm.x, lm.y] for lm in hand_res.landmark])
            base = pts[0]
            rel_pts = pts - base
            mcp_dist = np.linalg.norm(rel_pts[9])
            norm_pts = rel_pts / mcp_dist if mcp_dist > 1e-6 else rel_pts
            features = list(norm_pts.flatten())

            def get_angle(p1, p2, p3):
                v1, v2 = p1 - p2, p3 - p2
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    return 0.0
                return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)) / np.pi

            angles = [
                get_angle(pts[1], pts[2], pts[4]),
                get_angle(pts[5], pts[6], pts[8]),
                get_angle(pts[9], pts[10], pts[12]),
                get_angle(pts[13], pts[14], pts[16]),
                get_angle(pts[17], pts[18], pts[20]),
            ]
            dists = [
                np.linalg.norm(rel_pts[4] - rel_pts[8]),
                np.linalg.norm(rel_pts[4] - rel_pts[12]),
                np.linalg.norm(rel_pts[4] - rel_pts[16]),
                np.linalg.norm(rel_pts[4] - rel_pts[20]),
            ]
            if mcp_dist > 1e-6:
                dists = [d / mcp_dist for d in dists]
            features.extend(angles)
            features.extend(dists)

            if label == "Left":
                lh = np.array(features)
            else:
                rh = np.array(features)
    return np.concatenate([lh, rh]), hand_detected_valid


def extract_sequence_from_video(video_path, hand_threshold):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=int(RUNTIME_CONFIG["mediapipe_model_complexity"]),
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0.0, zh(ZH["open_video_failed"])

    prev_visual_data = np.zeros(VISUAL_SIZE)
    sequence = []
    total_frames = 0
    capture_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    target_size = (int(RUNTIME_CONFIG["process_width"]), int(RUNTIME_CONFIG["process_height"]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_image = cv2.resize(image, target_size)
        results = hands.process(small_image)
        visual_data = np.zeros(VISUAL_SIZE)
        velocity_data = np.zeros(VELOCITY_SIZE)
        hand_detected_valid = False

        if results.multi_hand_landmarks:
            visual_data, hand_detected_valid = extract_visual_keypoints(results, hand_threshold)
            if hand_detected_valid:
                velocity_data = visual_data - prev_visual_data
                prev_visual_data = visual_data
        if not hand_detected_valid:
            prev_visual_data = np.zeros(VISUAL_SIZE)

        fusion_data = np.concatenate([visual_data, velocity_data]).astype(np.float32)
        sequence.append(fusion_data)

    cap.release()
    hands.close()

    if len(sequence) < SEQUENCE_LENGTH:
        return [], total_frames, capture_fps, f"{zh(ZH['insufficient_frames'])} {SEQUENCE_LENGTH} \u5e27"
    return sequence[:SEQUENCE_LENGTH], total_frames, capture_fps, ""


def process_master_sample(sample_dir, output_root, feature_spec, hand_threshold, manifest_path):
    meta_path = os.path.join(sample_dir, "meta.json")
    meta = read_json(meta_path, default={})
    label = meta.get("\u6807\u7b7e") or os.path.basename(os.path.dirname(sample_dir))
    split = meta.get("\u5206\u533a") or os.path.basename(os.path.dirname(os.path.dirname(sample_dir)))
    sample_id = os.path.basename(sample_dir)
    video_path = os.path.join(sample_dir, "capture.mp4")

    if not os.path.exists(video_path):
        entry = build_manifest_entry(sample_id, label, split, "master_video", sample_dir, "", feature_spec, zh(ZH["failed"]), zh(ZH["missing_video"]))
        append_manifest_entry(manifest_path, entry)
        log(f"{zh(ZH['sample_fail'])}: {sample_id} - {zh(ZH['missing_video'])}")
        return False

    sequence, frame_count, capture_fps, error = extract_sequence_from_video(video_path, hand_threshold)
    meta["\u603b\u5e27\u6570"] = frame_count
    meta["\u91c7\u96c6\u5e27\u7387"] = capture_fps
    write_json(meta_path, meta)

    if error:
        entry = build_manifest_entry(sample_id, label, split, "master_video", sample_dir, "", feature_spec, zh(ZH["failed"]), error)
        append_manifest_entry(manifest_path, entry)
        log(f"{zh(ZH['sample_fail'])}: {sample_id} - {error}")
        return False

    save_root = dataset_split_dir(output_root, split)
    save_path = os.path.join(save_root, label, sample_id)
    ensure_dir(save_path)
    for idx, frame_data in enumerate(sequence):
        np.save(os.path.join(save_path, f"{idx}.npy"), frame_data)

    entry = build_manifest_entry(sample_id, label, split, "master_video", sample_dir, save_path, feature_spec, zh(ZH["success"]), "")
    append_manifest_entry(manifest_path, entry)
    log(f"{zh(ZH['sample_ok'])}: {split}/{label}/{sample_id}")
    return True


def register_legacy_features(legacy_root, output_root, feature_spec, manifest_path):
    if not legacy_root or not os.path.exists(legacy_root):
        return
    for split in ("train", "test"):
        split_root = dataset_split_dir(legacy_root, split)
        if not os.path.exists(split_root):
            continue
        for label in sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))]):
            label_root = os.path.join(split_root, label)
            for seq_id in sorted([d for d in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, d))], key=lambda x: int(x) if str(x).isdigit() else x):
                source_path = os.path.join(label_root, seq_id)
                target_path = os.path.join(dataset_split_dir(output_root, split), label, seq_id)
                if os.path.exists(target_path):
                    continue
                ensure_dir(os.path.dirname(target_path))
                shutil.copytree(source_path, target_path)
                entry = build_manifest_entry(seq_id, label, split, "legacy_feature", source_path, target_path, feature_spec, zh(ZH["success"]), "")
                append_manifest_entry(manifest_path, entry)


def scan_master_samples(master_root, split_filter):
    samples = []
    for split in ("train", "test"):
        if split_filter not in ("all", split):
            continue
        split_root = master_split_dir(master_root, split)
        if not os.path.exists(split_root):
            continue
        for label in sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))]):
            label_root = os.path.join(split_root, label)
            for sample_id in sorted([d for d in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, d))], key=lambda x: int(x) if str(x).isdigit() else x):
                samples.append(os.path.join(label_root, sample_id))
    return samples


def create_parser():
    parser = argparse.ArgumentParser(description=zh(ZH["desc"]))
    parser.add_argument("--母版目录", "--master-root", dest="master_root", default=RUNTIME_CONFIG["master_root"], help=zh("\u89c6\u9891\u6bcd\u7248\u6839\u76ee\u5f55"))
    parser.add_argument("--旧数据目录", "--legacy-root", dest="legacy_root", default=os.path.join("CNN_Gesture-master", "MP_Data_Fusion"), help=zh("\u65e7\u7248 204 \u7ef4\u7279\u5f81\u76ee\u5f55"))
    parser.add_argument("--输出目录", "--output-root", dest="output_root", default=RUNTIME_CONFIG["dataset_output_root"], help=zh("\u6784\u5efa\u540e\u7684\u8bad\u7ec3/\u6d4b\u8bd5\u96c6\u8f93\u51fa\u76ee\u5f55"))
    parser.add_argument("--特征方案", "--feature-spec", dest="feature_spec", default=RUNTIME_CONFIG["feature_spec"], help=zh("\u5f53\u524d\u6784\u5efa\u4f7f\u7528\u7684\u7279\u5f81\u65b9\u6848\u540d"))
    parser.add_argument("--数据分区", "--split", dest="split", choices=["train", "test", "all"], default="all", help=zh("\u53ea\u6784\u5efa\u8bad\u7ec3\u96c6\u3001\u6d4b\u8bd5\u96c6\u6216\u5168\u90e8"))
    parser.add_argument("--重建全部", "--rebuild", dest="rebuild", action="store_true", help=zh("\u91cd\u65b0\u6784\u5efa\u5e76\u6e05\u7a7a\u5df2\u6709\u8f93\u51fa"))
    parser.add_argument("--仅增量", "--incremental", dest="incremental", action="store_true", help=zh("\u53ea\u5904\u7406\u65b0\u6837\u672c\uff0c\u8df3\u8fc7\u5df2\u6709\u8f93\u51fa"))
    parser.add_argument("--手部阈值", dest="hand_threshold", type=float, default=0.50, help=zh("\u624b\u90e8\u6709\u6548\u6027\u9608\u503c"))
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    master_root = args.master_root
    output_root = args.output_root
    feature_spec = args.feature_spec
    manifest_path = os.path.join(output_root, "manifest.json")

    if args.rebuild and os.path.exists(output_root):
        log(zh(ZH["rebuild_all"]))
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    if args.rebuild:
        write_json(manifest_path, {"样本列表": []})
    else:
        load_manifest(manifest_path)

    log(f"{zh(ZH['scan_master'])}: {master_root}")
    samples = scan_master_samples(master_root, args.split)
    log(f"{zh(ZH['found_samples'])}: {len(samples)}")

    if args.legacy_root:
        log(zh(ZH["register_legacy"]))
        register_legacy_features(args.legacy_root, output_root, feature_spec, manifest_path)

    success_count = 0
    fail_count = 0
    for sample_dir in samples:
        sample_id = os.path.basename(sample_dir)
        meta = read_json(os.path.join(sample_dir, "meta.json"), default={})
        split = meta.get("\u5206\u533a") or os.path.basename(os.path.dirname(os.path.dirname(sample_dir)))
        label = meta.get("\u6807\u7b7e") or os.path.basename(os.path.dirname(sample_dir))
        output_path = os.path.join(dataset_split_dir(output_root, split), label, sample_id)
        if args.incremental and os.path.exists(output_path):
            log(f"{zh(ZH['skip_exists'])}: {split}/{label}/{sample_id}")
            continue
        ok = process_master_sample(sample_dir, output_root, feature_spec, args.hand_threshold, manifest_path)
        success_count += int(ok)
        fail_count += int(not ok)

    actions = list_action_names(output_root)
    log("")
    log(zh(ZH["complete"]))
    log(f"{zh(ZH['success_count'])}: {success_count}")
    log(f"{zh(ZH['fail_count'])}: {fail_count}")
    log(f"{zh(ZH['label_table'])}: {list(actions)}")
    log(f"{zh(ZH['output_dir'])}: {output_root}")
    log(f"{zh(ZH['manifest'])}: {manifest_path}")


if __name__ == "__main__":
    main()
