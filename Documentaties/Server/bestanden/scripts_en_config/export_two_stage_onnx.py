import json
import os

import onnx
import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet(num_classes: int, weight_path: str, device: torch.device):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_with_external_data(model, dummy_input, onnx_path: str):
    data_path = onnx_path + ".data"

    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    if os.path.exists(data_path):
        os.remove(data_path)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    return onnx_path, data_path


def main():
    model_dir = "/root/smart_bin_project/models/two_stage_crops"
    metadata_path = os.path.join(model_dir, "two_stage_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata niet gevonden: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    input_size = int(meta.get("input_size", 224))
    stage1_classes = meta["stage1_classes"]
    stage2_classes = meta["stage2_overige_classes"]

    stage1_weights = os.path.join(model_dir, meta["stage1_model"])
    stage2_weights = os.path.join(model_dir, meta["stage2_model"])
    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"Stage 1 gewichten niet gevonden: {stage1_weights}")
    if not os.path.exists(stage2_weights):
        raise FileNotFoundError(f"Stage 2 gewichten niet gevonden: {stage2_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    stage1_model = build_mobilenet(len(stage1_classes), stage1_weights, device)
    stage2_model = build_mobilenet(len(stage2_classes), stage2_weights, device)

    stage1_onnx = os.path.join(model_dir, "stage1_main.onnx")
    stage2_onnx = os.path.join(model_dir, "stage2_overige.onnx")

    out1 = export_with_external_data(stage1_model, dummy_input, stage1_onnx)
    out2 = export_with_external_data(stage2_model, dummy_input, stage2_onnx)

    print("Export klaar:")
    print(f"  Stage 1 ONNX : {out1[0]}")
    print(f"  Stage 1 DATA : {out1[1]}")
    print(f"  Stage 2 ONNX : {out2[0]}")
    print(f"  Stage 2 DATA : {out2[1]}")


if __name__ == "__main__":
    main()
