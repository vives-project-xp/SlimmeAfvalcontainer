import cv2
import numpy as np
import onnxruntime as ort

class YOLODetector:
    def __init__(
        self,
        model_path,
        conf_threshold=0.4,
        iou_threshold=0.35,
        min_box_area_ratio=0.002,
        max_box_area_ratio=0.85,
        max_aspect_ratio=8.0,
        max_detections=6,
        num_threads=4,
        inter_op_threads=1,
    ):
        sess_options = ort.SessionOptions()
        if num_threads is not None and int(num_threads) > 0:
            sess_options.intra_op_num_threads = int(num_threads)
        if inter_op_threads is not None and int(inter_op_threads) > 0:
            sess_options.inter_op_num_threads = int(inter_op_threads)
        # Keep execution deterministic and avoid thread-affinity surprises on VMs.
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.max_detections = int(max_detections)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.model_h = self.input_shape[2]
        self.model_w = self.input_shape[3]

    def preprocess(self, frame):
        """Resize and normalize frame for YOLO input."""
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.resize(frame, (self.model_w, self.model_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC to CHW
        img = np.expand_dims(img, 0)          # add batch dim
        return img

    def postprocess(self, outputs):
        """Parse raw YOLO output into [x1, y1, x2, y2, conf, class_id] rows."""
        raw = outputs[0][0]                   # shape: [4 + num_classes, num_detections]
        raw = np.transpose(raw)               # transpose to [num_detections, 4 + num_classes]
        boxes_xywh = raw[:, :4]
        scores = raw[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx, cy, w, h to x1, y1, x2, y2 and scale to original frame size
        x_scale = self.orig_w / self.model_w
        y_scale = self.orig_h / self.model_h
        img_area = float(self.orig_w * self.orig_h)

        results = []
        for (cx, cy, w, h), conf, cls in zip(boxes_xywh, confidences, class_ids):
            x1 = int((cx - w / 2) * x_scale)
            y1 = int((cy - h / 2) * y_scale)
            x2 = int((cx + w / 2) * x_scale)
            y2 = int((cy + h / 2) * y_scale)
            x1 = max(0, min(self.orig_w - 1, x1))
            y1 = max(0, min(self.orig_h - 1, y1))
            x2 = max(0, min(self.orig_w - 1, x2))
            y2 = max(0, min(self.orig_h - 1, y2))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            area_ratio = (bw * bh) / max(1.0, img_area)
            if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
                continue

            aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect_ratio > self.max_aspect_ratio:
                continue

            results.append([x1, y1, x2, y2, float(conf), int(cls)])

        # NMS to remove overlapping boxes for the same object
        if results:
            boxes = [[r[0], r[1], r[2]-r[0], r[3]-r[1]] for r in results]
            scores = [r[4] for r in results]
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if indices is None or len(indices) == 0:
                return []

            if isinstance(indices, tuple):
                kept = [int(i) for i in indices]
            else:
                kept = np.array(indices).reshape(-1).astype(int).tolist()
            kept.sort(key=lambda idx: results[idx][4], reverse=True)
            kept = kept[:self.max_detections]
            results = [results[i] for i in kept]

        return results

    def detect(self, frame):
        inp = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs)
