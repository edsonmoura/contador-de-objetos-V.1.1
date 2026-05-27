import cv2
import numpy as np

from object_counter.counter import Detection


HARDWARE_PROFILES = {
    "auto": {
        "min_area": 350,
        "diff_scale": 0.75,
        "close_ratio": 0.014,
        "open_ratio": 0.004,
        "min_fill": 0.10,
        "max_aspect": 8.0,
    },
    "precise": {
        "min_area": 650,
        "diff_scale": 0.95,
        "close_ratio": 0.012,
        "open_ratio": 0.005,
        "min_fill": 0.15,
        "max_aspect": 7.0,
    },
    "sensitive": {
        "min_area": 220,
        "diff_scale": 0.55,
        "close_ratio": 0.018,
        "open_ratio": 0.003,
        "min_fill": 0.08,
        "max_aspect": 9.0,
    },
    "small": {
        "min_area": 120,
        "diff_scale": 0.50,
        "close_ratio": 0.016,
        "open_ratio": 0.003,
        "min_fill": 0.07,
        "max_aspect": 9.0,
    },
}


def detect_dark_objects(
    frame_bgr: np.ndarray,
    min_area: int = 800,
    label: str = "objeto escuro",
) -> list[Detection]:
    return detect_blobs_by_threshold(
        frame_bgr,
        threshold_mode=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        min_area=min_area,
        label=label,
    )


def detect_light_objects(
    frame_bgr: np.ndarray,
    min_area: int = 80,
    label: str = "objeto claro",
) -> list[Detection]:
    return detect_blobs_by_threshold(
        frame_bgr,
        threshold_mode=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        min_area=min_area,
        label=label,
    )


def detect_contrast_objects(
    frame_bgr: np.ndarray,
    min_area: int | None = None,
    label: str = "objeto",
    profile: str = "auto",
) -> list[Detection]:
    if label == "peca":
        return detect_hardware_components(frame_bgr, min_area=min_area, profile=profile)

    area = min_area or 120
    dark = detect_dark_objects(frame_bgr, min_area=area, label=label)
    light = detect_light_objects(frame_bgr, min_area=area, label=label)
    return merge_overlapping_detections(dark + light)


def detect_hardware_components(
    frame_bgr: np.ndarray,
    min_area: int | None = None,
    profile: str = "auto",
) -> list[Detection]:
    settings = HARDWARE_PROFILES.get(profile, HARDWARE_PROFILES["auto"])
    area_limit = min_area or int(settings["min_area"])
    mask, raw_mask = build_hardware_mask(frame_bgr, settings)
    return detections_from_hardware_mask(
        frame_bgr,
        mask,
        raw_mask,
        min_area=area_limit,
        min_fill=float(settings["min_fill"]),
        max_aspect=float(settings["max_aspect"]),
    )


def detect_objects_against_background(
    frame_bgr: np.ndarray,
    min_area: int = 500,
    label: str = "peca",
) -> list[Detection]:
    height, width = frame_bgr.shape[:2]
    background_bgr = estimate_background_color(frame_bgr)

    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    background_lab = cv2.cvtColor(np.uint8([[background_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
    distance = np.linalg.norm(frame_lab - background_lab, axis=2)
    distance = np.clip(distance, 0, 255).astype(np.uint8)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)

    _threshold_value, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_size = max(7, make_odd(int(min(height, width) * 0.018)))
    open_size = max(3, make_odd(int(min(height, width) * 0.006)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    image_area = height * width
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > image_area * 0.55:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area_value = box_width * box_height
        if box_area_value <= 0:
            continue

        fill_ratio = area / box_area_value
        aspect_ratio = max(box_width / box_height, box_height / box_width)
        if fill_ratio < 0.12 or aspect_ratio > 8:
            continue

        detections.append(
            Detection(
                label=label,
                confidence=1.0,
                box=(x, y, box_width, box_height),
            )
        )

    return sorted(detections, key=lambda detection: (detection.box[1], detection.box[0]))


def build_hardware_mask(frame_bgr: np.ndarray, settings: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    min_dimension = min(height, width)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    background_kernel_size = make_odd(max(31, min(151, int(min_dimension * 0.13))))
    local_background = cv2.medianBlur(blurred, background_kernel_size)
    difference = cv2.absdiff(blurred, local_background)

    edges = cv2.Canny(blurred, 40, 120)
    difference = cv2.max(difference, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))

    otsu_threshold, _mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = max(12, int(otsu_threshold * float(settings["diff_scale"])))
    raw_mask = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)[1]

    close_size = max(5, make_odd(int(min_dimension * float(settings["close_ratio"]))))
    open_size = max(3, make_odd(int(min_dimension * float(settings["open_ratio"]))))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = fill_external_contours(mask)
    return mask, raw_mask


def fill_external_contours(mask: np.ndarray) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def detections_from_hardware_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    raw_mask: np.ndarray,
    min_area: int,
    min_fill: float,
    max_aspect: float,
) -> list[Detection]:
    height, width = frame_bgr.shape[:2]
    image_area = height * width
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > image_area * 0.20:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue

        box_area_value = box_width * box_height
        fill_ratio = area / box_area_value
        aspect_ratio = max(box_width / box_height, box_height / box_width)
        if fill_ratio < min_fill or aspect_ratio > max_aspect:
            continue

        label = classify_hardware_component(frame_bgr, raw_mask, contour, (x, y, box_width, box_height))
        detections.append(
            Detection(
                label=label,
                confidence=1.0,
                box=clip_box((x, y, box_width, box_height), width, height),
            )
        )

    merged = merge_overlapping_detections(detections)
    return sorted(merged, key=lambda detection: (detection.box[1], detection.box[0]))


def classify_hardware_component(
    frame_bgr: np.ndarray,
    raw_mask: np.ndarray,
    contour: np.ndarray,
    box: tuple[int, int, int, int],
) -> str:
    x, y, width, height = box
    rect_width, rect_height = cv2.minAreaRect(contour)[1]
    rotated_aspect_ratio = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
    axis_aspect_ratio = max(width / height, height / width)
    if max(rotated_aspect_ratio, axis_aspect_ratio) >= 1.45:
        return "parafuso"

    contour_area = max(1.0, cv2.contourArea(contour))
    perimeter = max(1.0, cv2.arcLength(contour, True))
    circularity = min(1.0, 4 * np.pi * contour_area / (perimeter * perimeter))
    solidity = estimate_solidity(contour)
    vertex_count = estimate_vertex_count(contour)
    hole_ratio = estimate_hole_ratio(raw_mask, box)
    edge_density = estimate_edge_density(frame_bgr[y : y + height, x : x + width])

    if hole_ratio >= 0.11 and circularity >= 0.55 and vertex_count > 8:
        return "arruela"

    if hole_ratio >= 0.04 or vertex_count <= 8 or solidity < 0.88 or edge_density >= 0.22:
        return "porca"

    return "fixacao"


def estimate_solidity(contour: np.ndarray) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0

    return float(cv2.contourArea(contour) / hull_area)


def estimate_vertex_count(contour: np.ndarray) -> int:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0

    approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    return int(len(approximation))


def estimate_hole_ratio(raw_mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, width, height = box
    crop = raw_mask[y : y + height, x : x + width]
    if crop.size == 0:
        return 0.0

    center_margin_x = max(1, int(width * 0.18))
    center_margin_y = max(1, int(height * 0.18))
    center = crop[center_margin_y : height - center_margin_y, center_margin_x : width - center_margin_x]
    if center.size == 0:
        return 0.0

    background_like = cv2.bitwise_not(center)
    contours, _hierarchy = cv2.findContours(background_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest_hole = max(cv2.contourArea(contour) for contour in contours)
    return float(largest_hole / max(1, width * height))


def estimate_edge_density(frame_bgr: np.ndarray) -> float:
    if frame_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    return float(np.count_nonzero(edges) / edges.size)


def clip_box(box: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = box
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    right = max(x + 1, min(x + width, image_width))
    bottom = max(y + 1, min(y + height, image_height))
    return x, y, right - x, bottom - y


def estimate_background_color(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    border = max(4, int(min(height, width) * 0.04))
    samples = np.concatenate(
        [
            frame_bgr[:border, :, :].reshape(-1, 3),
            frame_bgr[-border:, :, :].reshape(-1, 3),
            frame_bgr[:, :border, :].reshape(-1, 3),
            frame_bgr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0).astype(np.uint8)


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def detect_blobs_by_threshold(
    frame_bgr: np.ndarray,
    threshold_mode: int,
    min_area: int,
    label: str,
) -> list[Detection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        threshold_mode,
    )

    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(
                label=label,
                confidence=1.0,
                box=(x, y, width, height),
            )
        )

    return sorted(detections, key=lambda detection: (detection.box[1], detection.box[0]))


def merge_overlapping_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []

    for detection in sorted(detections, key=lambda item: box_area(item.box), reverse=True):
        if any(intersection_over_union(detection.box, other.box) > 0.35 for other in kept):
            continue
        kept.append(detection)

    return sorted(kept, key=lambda item: (item.box[1], item.box[0]))


def box_area(box: tuple[int, int, int, int]) -> int:
    return box[2] * box[3]


def intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second

    x_left = max(first_x, second_x)
    y_top = max(first_y, second_y)
    x_right = min(first_x + first_width, second_x + second_width)
    y_bottom = min(first_y + first_height, second_y + second_height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union
