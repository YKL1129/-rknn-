import argparse
import os

import numpy as np

from gesture_runtime import load_sequence_dataset, list_action_names, read_json, stratified_split


SEQUENCE_LENGTH = 30
INPUT_SIZE = 204


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def load_fixed_test_dataset(data_path, sequence_length, input_size):
    manifest_path = os.path.join(data_path, "manifest.json")
    manifest = read_json(manifest_path, default={})
    train_actions = []
    if manifest.get("样本列表"):
        train_actions = sorted(list({
            item.get("标签")
            for item in manifest["样本列表"]
            if item.get("分区") == "train" and item.get("状态") == "成功" and item.get("标签")
        }))
        train_actions = np.array(train_actions)
    if len(train_actions) == 0:
        train_actions = list_action_names(data_path)
    if len(train_actions) == 0:
        raise SystemExit(f"No training classes found in {os.path.join(data_path, 'train')} or {data_path}")

    test_root = os.path.join(data_path, "test")
    if not os.path.exists(test_root):
        raise SystemExit(f"No fixed test-set directory found in {test_root}")

    label_map = {label: idx for idx, label in enumerate(train_actions)}
    sequences, labels, bad = [], [], []

    for action in sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]):
        if action not in label_map:
            bad.append(os.path.join(action, "__unknown_class__"))
            continue
        action_path = os.path.join(test_root, action)
        seq_dirs = sorted(
            [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))],
            key=lambda x: int(x) if str(x).isdigit() else x
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

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), train_actions, bad


def predict_h5(model_path, X):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    return model.predict(X, verbose=0)


def predict_onnx(model_path, X):
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    probs = sess.run(None, {input_name: X.astype(np.float32)})[0]
    return probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate gesture model on held-out sequence data.")
    parser.add_argument("--data", default=os.path.join("CNN_Gesture-master", "MP_Data_Fusion"))
    parser.add_argument("--model", default=os.path.join("CNN_Gesture-master", "action_fusion.onnx"))
    parser.add_argument("--format", choices=["onnx", "h5"], default="onnx")
    parser.add_argument("--split", choices=["test", "all", "holdout"], default="test")
    args = parser.parse_args()

    if args.split == "test":
        X_eval, y_eval, actions, bad = load_fixed_test_dataset(args.data, SEQUENCE_LENGTH, INPUT_SIZE)
        if len(X_eval) == 0:
            raise SystemExit(f"No fixed test-set sequences found in {os.path.join(args.data, 'test')}")
    else:
        X, labels, actions, bad = load_sequence_dataset(args.data, SEQUENCE_LENGTH, INPUT_SIZE)
        if len(X) == 0:
            raise SystemExit(f"No valid sequences found in {args.data}")

    if args.split == "holdout":
        _, _, test_idx = stratified_split(labels, train_ratio=0.70, val_ratio=0.15, seed=42)
        if len(test_idx) == 0:
            raise SystemExit("No held-out test samples. Collect more sequences per class or use --split all.")
        X_eval = X[test_idx]
        y_eval = labels[test_idx]
    elif args.split == "all":
        X_eval = X
        y_eval = labels

    probs = predict_onnx(args.model, X_eval) if args.format == "onnx" else predict_h5(args.model, X_eval)
    if probs.ndim != 2:
        raise SystemExit(f"Unexpected model output shape: {probs.shape}")
    if probs.shape[1] != len(actions):
        test_root = os.path.join(args.data, "test")
        test_actions = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]) if os.path.exists(test_root) else []
        raise SystemExit(
            "Class mismatch: model outputs "
            f"{probs.shape[1]} classes, but the current dataset label table has {len(actions)} classes. "
            f"Train labels: {list(actions)}. "
            f"Test labels: {test_actions}. "
            "Check your train/test folders and make sure the trained model matches the current class set."
        )
    y_pred = np.argmax(probs, axis=1)
    overall = float(np.mean(y_pred == y_eval))
    matrix = confusion_matrix(y_eval, y_pred, len(actions))

    print(f"Samples: {len(X_eval)}")
    print(f"Overall accuracy: {overall:.4f}")
    print(f"Skipped invalid sequences: {len(bad)}")
    print()
    print("Per-class accuracy:")
    for idx, action in enumerate(actions):
        mask = y_eval == idx
        if not np.any(mask):
            print(f"  {action}: n=0 acc=N/A")
            continue
        acc = float(np.mean(y_pred[mask] == y_eval[mask]))
        print(f"  {action}: n={int(mask.sum())} acc={acc:.4f}")

    print()
    print("Confusion matrix rows=true cols=pred:")
    print("labels:", " | ".join(str(a) for a in actions))
    for idx, row in enumerate(matrix):
        print(f"{actions[idx]}:", " ".join(str(int(v)) for v in row))


if __name__ == "__main__":
    main()
