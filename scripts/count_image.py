import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from object_counter.counter import Detection
from object_counter.counter import count_by_label, format_counts
from object_counter.detector import YoloOnnxDetector, draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conta objetos em uma imagem.")
    parser.add_argument("image", type=Path, help="Caminho da imagem para analisar.")
    parser.add_argument(
        "--mode",
        choices=["auto", "yolo", "dark", "light", "contrast", "rice", "beans", "parts"],
        default="auto",
        help="Modo de contagem: auto, yolo, dark, light, contrast, rice, beans ou parts.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="Area minima usada nos modos por contorno. Se omitida, usa o padrao do preset.",
    )
    parser.add_argument(
        "--parts-profile",
        choices=["auto", "precise", "sensitive", "small"],
        default="auto",
        help="Ajuste do modo parts: auto, precise, sensitive ou small.",
    )
    parser.add_argument(
        "--parts-type",
        choices=["all", "parafuso", "porca", "arruela", "fixacao"],
        default="all",
        help="Filtra o modo parts por tipo: all, parafuso, porca, arruela ou fixacao.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures/result.jpg"),
        help="Caminho para salvar a imagem com caixas desenhadas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(str(args.image))

    if frame is None:
        raise SystemExit(f"Nao foi possivel abrir a imagem: {args.image}")

    detections = detect_objects(frame, args.mode, args.min_area, args.parts_profile, args.parts_type)
    counts, total = count_by_label(detections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(frame, detections)
    cv2.imwrite(str(args.output), annotated)

    print(format_counts(counts, total))
    print(f"Imagem anotada salva em: {args.output}")


def detect_objects(
    frame,
    mode: str,
    min_area: int | None,
    parts_profile: str,
    parts_type: str,
) -> list[Detection]:
    if mode in {"dark", "beans"}:
        area = min_area or (800 if mode == "dark" else 250)
        return detect_dark_objects(frame, area, label="feijao" if mode == "beans" else "objeto escuro")

    if mode in {"light", "rice"}:
        area = min_area or (80 if mode == "light" else 40)
        return detect_light_objects(frame, area, label="arroz" if mode == "rice" else "objeto claro")

    if mode in {"contrast", "parts"}:
        area = min_area if min_area is not None else (None if mode == "parts" else 120)
        detections = detect_contrast_objects(
            frame,
            area,
            label="peca" if mode == "parts" else "objeto",
            profile=parts_profile,
        )
        if mode == "parts" and parts_type != "all":
            return [detection for detection in detections if detection.label == parts_type]

        return detections

    if mode == "yolo":
        return detect_with_yolo(frame)

    detections = detect_with_yolo(frame)
    if detections:
        return detections

    return detect_dark_objects(frame, min_area or 800)


def detect_with_yolo(frame) -> list[Detection]:
    onnx_model = PROJECT_ROOT / "models" / "yolov8n.onnx"
    pt_model = PROJECT_ROOT / "yolov8n.pt"

    if onnx_model.exists():
        detector = YoloOnnxDetector(onnx_model)
        return detector.detect(frame)

    if pt_model.exists():
        return detect_with_ultralytics(frame, pt_model)

    raise SystemExit(
        "Nenhum modelo encontrado.\n"
        "Esperado um destes arquivos:\n"
        f"- {onnx_model}\n"
        f"- {pt_model}\n\n"
        "Como voce ja baixou o yolov8n.pt antes, confirme se ele esta na pasta raiz do projeto."
    )


def detect_dark_objects(frame, min_area: int, label: str = "objeto escuro") -> list[Detection]:
    from object_counter.blob_counter import detect_dark_objects as detect_blobs

    return detect_blobs(frame, min_area=min_area, label=label)


def detect_light_objects(frame, min_area: int, label: str) -> list[Detection]:
    from object_counter.blob_counter import detect_light_objects as detect_blobs

    return detect_blobs(frame, min_area=min_area, label=label)


def detect_contrast_objects(frame, min_area: int, label: str) -> list[Detection]:
    from object_counter.blob_counter import detect_contrast_objects as detect_blobs

    return detect_blobs(frame, min_area=min_area, label=label)


def detect_with_ultralytics(frame, model_path: Path) -> list[Detection]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise SystemExit(
            "O arquivo ONNX nao existe e o pacote ultralytics nao esta instalado.\n"
            "Rode: python -m pip install ultralytics"
        ) from error

    model = YOLO(str(model_path))
    result = model.predict(frame, verbose=False)[0]
    names = result.names
    detections: list[Detection] = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            )
        )

    return detections


if __name__ == "__main__":
    main()
