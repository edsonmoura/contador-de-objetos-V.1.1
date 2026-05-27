from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


def count_by_label(detections: list[Detection]) -> tuple[dict[str, int], int]:
    counts = Counter(detection.label for detection in detections)
    ordered_counts = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return ordered_counts, sum(ordered_counts.values())


def format_counts(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return "Nenhum objeto detectado."

    lines = [f"Total: {total}", ""]
    lines.extend(f"{label}: {quantity}" for label, quantity in counts.items())
    return "\n".join(lines)

