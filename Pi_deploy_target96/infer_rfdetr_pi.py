#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image
import supervision as sv

from rfdetr import RFDETRBase


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR inferentie op Raspberry Pi (CPU)")
    parser.add_argument("--model", default="model_best_ema_target96.pth", help="Pad naar .pth checkpoint")
    parser.add_argument("--image", required=True, help="Pad naar input afbeelding")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", default="prediction.jpg", help="Pad voor geannoteerde output")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model niet gevonden: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Afbeelding niet gevonden: {image_path}")

    model = RFDETRBase(pretrain_weights=str(model_path), device="cpu")
    detections = model.predict(str(image_path), threshold=args.threshold)

    image = sv.read_image(str(image_path))
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    labels = []
    for conf, class_id in zip(detections.confidence, detections.class_id):
        class_name = model.class_names.get(int(class_id), str(class_id))
        labels.append(f"{class_name} {float(conf):.2f}")

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    Image.fromarray(annotated).save(output_path)

    print(f"Detecties: {len(detections)}")
    print(f"Input:    {image_path}")
    print(f"Output:   {output_path}")


if __name__ == "__main__":
    main()
