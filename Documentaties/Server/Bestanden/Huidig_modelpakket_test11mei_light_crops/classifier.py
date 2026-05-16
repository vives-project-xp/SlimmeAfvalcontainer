import numpy as np
import cv2
import json
import os
import onnxruntime as ort

class TwoStageGarbageClassifier:
    def __init__(self, model_dir=".", stage1_threshold=0.60, stage2_threshold=0.55, onnx_threads=4):
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold

        metadata_path = os.path.join(model_dir, "two_stage_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.input_size = int(meta.get("input_size", 224))
        self.stage1_classes = meta["stage1_classes"]
        self.stage2_classes = meta["stage2_overige_classes"]
        self.main_label_for_stage2 = meta.get("main_label_for_stage2", "Overige")
        self.default_fallback = meta.get("default_fallback", "Restafval")

        stage1_onnx = os.path.join(model_dir, "stage1_main.onnx")
        stage2_onnx = os.path.join(model_dir, "stage2_overige.onnx")

        # Load ONNX models
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, int(onnx_threads))
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.stage1_session = ort.InferenceSession(stage1_onnx, sess_options, providers=["CPUExecutionProvider"])
        self.stage1_input_name = self.stage1_session.get_inputs()[0].name

        self.stage2_session = ort.InferenceSession(stage2_onnx, sess_options, providers=["CPUExecutionProvider"])
        self.stage2_input_name = self.stage2_session.get_inputs()[0].name

    def preprocess(self, crop):
        img = cv2.resize(crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img

    def classify(self, crop):
        if crop.size == 0:
            return "unknown", 0.0

        inp = self.preprocess(crop)

        # Stage 1
        outputs1 = self.stage1_session.run(None, {self.stage1_input_name: inp})
        logits1 = outputs1[0][0]
        exp1 = np.exp(logits1 - np.max(logits1))
        probs1 = exp1 / exp1.sum()

        s1_idx = int(np.argmax(probs1))
        s1_conf = float(probs1[s1_idx])
        stage1_label = self.stage1_classes[s1_idx]

        if s1_conf < self.stage1_threshold:
            return self.default_fallback, s1_conf

        if stage1_label != self.main_label_for_stage2:
            return stage1_label, s1_conf

        # Stage 2
        outputs2 = self.stage2_session.run(None, {self.stage2_input_name: inp})
        logits2 = outputs2[0][0]
        exp2 = np.exp(logits2 - np.max(logits2))
        probs2 = exp2 / exp2.sum()

        s2_idx = int(np.argmax(probs2))
        s2_conf = float(probs2[s2_idx])
        sub_label = self.stage2_classes[s2_idx]

        if s2_conf < self.stage2_threshold:
            return self.main_label_for_stage2, s2_conf
        else:
            return f"{self.main_label_for_stage2}/{sub_label}", s2_conf
