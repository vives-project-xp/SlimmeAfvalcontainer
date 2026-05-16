import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TwoStageSmartBinClassifier:
    def __init__(
        self,
        model_dir="models/two_stage",
        stage1_threshold=0.60,
        stage2_threshold=0.55,
    ):
        self.model_dir = model_dir
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metadata_path = os.path.join(model_dir, "two_stage_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata niet gevonden: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.input_size = int(meta.get("input_size", 224))
        self.stage1_classes = meta["stage1_classes"]
        self.stage2_classes = meta["stage2_overige_classes"]
        self.main_label_for_stage2 = meta.get("main_label_for_stage2", "Overige")
        self.default_fallback = meta.get("default_fallback", "Restafval")

        stage1_weights = os.path.join(model_dir, meta["stage1_model"])
        stage2_weights = os.path.join(model_dir, meta["stage2_model"])
        if not os.path.exists(stage1_weights):
            raise FileNotFoundError(f"Stage 1 gewichten niet gevonden: {stage1_weights}")
        if not os.path.exists(stage2_weights):
            raise FileNotFoundError(f"Stage 2 gewichten niet gevonden: {stage2_weights}")

        self.stage1_model = self._build_model(len(self.stage1_classes), stage1_weights)
        self.stage2_model = self._build_model(len(self.stage2_classes), stage2_weights)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, num_classes, weight_path):
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            stage1_logits = self.stage1_model(x)
            stage1_probs = F.softmax(stage1_logits, dim=1)[0]
            s1_conf, s1_idx = torch.max(stage1_probs, dim=0)

        s1_conf = float(s1_conf.item())
        s1_idx = int(s1_idx.item())
        stage1_label = self.stage1_classes[s1_idx]

        result = {
            "stage1_prediction": stage1_label,
            "stage1_confidence": s1_conf,
            "stage2_prediction": None,
            "stage2_confidence": None,
            "is_uncertain": False,
            "final_class": None,
            "reason": "",
        }

        if s1_conf < self.stage1_threshold:
            result["is_uncertain"] = True
            result["final_class"] = self.default_fallback
            result["reason"] = (
                f"Stage 1 te onzeker ({s1_conf*100:.1f}% < {self.stage1_threshold*100:.1f}%), "
                f"fallback naar {self.default_fallback}"
            )
            return result

        if stage1_label != self.main_label_for_stage2:
            result["final_class"] = stage1_label
            result["reason"] = f"Stage 1 beslist direct: {stage1_label} ({s1_conf*100:.1f}%)"
            return result

        with torch.no_grad():
            stage2_logits = self.stage2_model(x)
            stage2_probs = F.softmax(stage2_logits, dim=1)[0]
            s2_conf, s2_idx = torch.max(stage2_probs, dim=0)

        s2_conf = float(s2_conf.item())
        s2_idx = int(s2_idx.item())
        sub_label = self.stage2_classes[s2_idx]

        result["stage2_prediction"] = f"{self.main_label_for_stage2}/{sub_label}"
        result["stage2_confidence"] = s2_conf

        if s2_conf < self.stage2_threshold:
            result["is_uncertain"] = True
            result["final_class"] = self.main_label_for_stage2
            result["reason"] = (
                f"Stage 2 te onzeker ({s2_conf*100:.1f}% < {self.stage2_threshold*100:.1f}%), "
                f"terug naar hoofdklasse {self.main_label_for_stage2}"
            )
        else:
            result["final_class"] = f"{self.main_label_for_stage2}/{sub_label}"
            result["reason"] = f"2-staps match: {result['final_class']} ({s2_conf*100:.1f}%)"

        return result


def main():
    if len(sys.argv) < 2:
        print("Gebruik: python pi_inference_two_stage.py <image_path> [stage1_threshold] [stage2_threshold]")
        return

    image_path = sys.argv[1]
    stage1_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
    stage2_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55

    clf = TwoStageSmartBinClassifier(
        model_dir="models/two_stage",
        stage1_threshold=stage1_threshold,
        stage2_threshold=stage2_threshold,
    )
    out = clf.predict(image_path)

    print("\n" + "=" * 60)
    print(f"Afbeelding: {image_path}")
    print("=" * 60)
    print(f"Stage 1: {out['stage1_prediction']} ({out['stage1_confidence']*100:.1f}%)")
    if out["stage2_prediction"] is not None:
        print(f"Stage 2: {out['stage2_prediction']} ({out['stage2_confidence']*100:.1f}%)")
    print(f"Finale klasse: {out['final_class']}")
    print(f"Status: {out['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
