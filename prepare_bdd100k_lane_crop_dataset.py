import os
import json
import cv2
import numpy as np
from collections import Counter


# =========================
# 경로 설정
# =========================

BDD_ROOT = r"C:\Users\olcha\Safepark\bdd100k"
PROJECT_ROOT = r"C:\Users\olcha\Safepark\SafePark-DS"

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset_lane_crop")

CLASS_NAMES = [
    "yellow_double",
    "yellow_single",
    "white_dotted",
    "white_solid",
    "none"
]


# =========================
# 기본 함수
# =========================

def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def ensure_dirs():
    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            os.makedirs(
                os.path.join(OUTPUT_ROOT, split, class_name),
                exist_ok=True
            )


def get_image_dir(split):
    candidates = [
        os.path.join(BDD_ROOT, split, "images"),
        os.path.join(BDD_ROOT, "images", "100k", split),
        os.path.join(BDD_ROOT, "images", split),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"{split} 이미지 폴더를 찾을 수 없습니다.")


def get_label_json(split):
    candidates = [
        os.path.join(BDD_ROOT, split, "annotations", f"bdd100k_labels_images_{split}.json"),
        os.path.join(BDD_ROOT, "labels", f"bdd100k_labels_images_{split}.json"),
        os.path.join(BDD_ROOT, "labels", "lane", f"lane_{split}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"{split} 라벨 JSON을 찾을 수 없습니다.")


def get_lane_class(label):
    """
    BDD100K lane label 1개를 우리 클래스명으로 변환.
    """

    category = normalize_text(label.get("category"))

    if category != "lane":
        return None

    attrs = label.get("attributes", {})

    lane_type = normalize_text(
        attrs.get("laneType")
        or attrs.get("lane_type")
        or label.get("laneType")
        or ""
    )

    lane_style = normalize_text(
        attrs.get("laneStyle")
        or attrs.get("lane_style")
        or attrs.get("continuity")
        or attrs.get("style")
        or label.get("laneStyle")
        or ""
    )

    # road curb, crosswalk, other는 제외
    if lane_type in ["road curb", "crosswalk", "single other", "double other"]:
        return None

    if lane_type == "double yellow":
        return "yellow_double"

    if lane_type == "single yellow":
        return "yellow_single"

    if lane_type == "single white":
        if "dash" in lane_style or "dotted" in lane_style or "broken" in lane_style:
            return "white_dotted"
        return "white_solid"

    if lane_type == "double white":
        # double white는 일단 white_solid로 묶음
        return "white_solid"

    return None


def collect_vertices(label):
    """
    label 안의 poly2d vertices를 모두 모은다.
    """
    points = []

    for poly in label.get("poly2d", []):
        vertices = poly.get("vertices", [])

        for p in vertices:
            if len(p) >= 2:
                x, y = p[0], p[1]
                points.append([float(x), float(y)])

    return points


def crop_lane_roi(img, points, margin=80, output_size=224):
    """
    poly2d 좌표 주변 bounding box를 만들고 crop.
    """
    if not points:
        return None

    h, w = img.shape[:2]

    pts = np.array(points, dtype=np.float32)

    x_min = int(np.min(pts[:, 0]) - margin)
    y_min = int(np.min(pts[:, 1]) - margin)
    x_max = int(np.max(pts[:, 0]) + margin)
    y_max = int(np.max(pts[:, 1]) + margin)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    # 너무 작은 crop은 제외
    if crop_w < 40 or crop_h < 40:
        return None

    crop = img[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (output_size, output_size))

    return crop


def save_crop(crop, split, class_name, image_base_name, lane_id, counter):
    save_dir = os.path.join(OUTPUT_ROOT, split, class_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{image_base_name}_lane{lane_id}_{counter}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, crop)

    return save_path


# =========================
# none 데이터 생성
# =========================

def make_none_crop(img, lane_boxes, output_size=224, max_trials=30):
    """
    차선이 없는 영역에서 random crop을 만들어 none 클래스로 사용.
    단순 버전: 이미지 상단/좌우 등에서 랜덤 crop.
    """

    h, w = img.shape[:2]

    crop_size = min(h, w) // 3
    crop_size = max(160, min(crop_size, 320))

    for _ in range(max_trials):
        x1 = np.random.randint(0, max(1, w - crop_size))
        y1 = np.random.randint(0, max(1, h - crop_size))

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # lane box와 너무 겹치면 제외
        overlap = False
        for bx1, by1, bx2, by2 in lane_boxes:
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                crop_area = crop_size * crop_size

                if inter_area / crop_area > 0.10:
                    overlap = True
                    break

        if overlap:
            continue

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (output_size, output_size))
        return crop

    return None


def get_lane_box(points, margin=80, img_shape=None):
    if not points:
        return None

    pts = np.array(points, dtype=np.float32)

    x1 = int(np.min(pts[:, 0]) - margin)
    y1 = int(np.min(pts[:, 1]) - margin)
    x2 = int(np.max(pts[:, 0]) + margin)
    y2 = int(np.max(pts[:, 1]) + margin)

    if img_shape is not None:
        h, w = img_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

    return x1, y1, x2, y2


# =========================
# 변환 실행
# =========================

def convert_split(split, max_per_class):
    image_dir = get_image_dir(split)
    label_json = get_label_json(split)

    print()
    print(f"[{split}] 이미지 폴더:", image_dir)
    print(f"[{split}] 라벨 파일:", label_json)

    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_counts = Counter()
    missing_images = 0
    total_lane_crops = 0

    for item_idx, item in enumerate(data):
        image_name = (
            item.get("name")
            or item.get("image")
            or item.get("file_name")
            or item.get("filename")
        )

        if not image_name:
            continue

        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            missing_images += 1
            continue

        img = cv2.imread(image_path)

        if img is None:
            missing_images += 1
            continue

        image_base_name = os.path.splitext(os.path.basename(image_name))[0]

        lane_boxes = []
        labels = item.get("labels", [])

        # 먼저 실제 차선 crop 저장
        for label in labels:
            class_name = get_lane_class(label)

            if class_name is None:
                continue

            if class_counts[class_name] >= max_per_class:
                continue

            points = collect_vertices(label)

            if not points:
                continue

            lane_box = get_lane_box(points, margin=80, img_shape=img.shape)

            if lane_box is not None:
                lane_boxes.append(lane_box)

            crop = crop_lane_roi(
                img=img,
                points=points,
                margin=80,
                output_size=224
            )

            if crop is None:
                continue

            lane_id = label.get("id", total_lane_crops)

            save_crop(
                crop=crop,
                split=split,
                class_name=class_name,
                image_base_name=image_base_name,
                lane_id=lane_id,
                counter=class_counts[class_name]
            )

            class_counts[class_name] += 1
            total_lane_crops += 1

        # none crop 생성
        if class_counts["none"] < max_per_class:
            none_crop = make_none_crop(
                img=img,
                lane_boxes=lane_boxes,
                output_size=224
            )

            if none_crop is not None:
                save_crop(
                    crop=none_crop,
                    split=split,
                    class_name="none",
                    image_base_name=image_base_name,
                    lane_id="none",
                    counter=class_counts["none"]
                )
                class_counts["none"] += 1

        if (item_idx + 1) % 1000 == 0:
            print(f"{item_idx + 1}개 이미지 처리 중...")
            for class_name in CLASS_NAMES:
                print(f" - {class_name}: {class_counts[class_name]}")

        # 모든 클래스가 max_per_class 이상이면 중단
        if all(class_counts[c] >= max_per_class for c in CLASS_NAMES):
            break

    print()
    print(f"[{split}] crop 변환 완료")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {class_counts[class_name]}장")

    if missing_images > 0:
        print("이미지 누락:", missing_images)


def main():
    print("BDD100K 차선 ROI crop 데이터셋 생성 시작")
    print("BDD_ROOT:", BDD_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)

    ensure_dirs()

    convert_split("train", max_per_class=3000)
    convert_split("val", max_per_class=800)

    print()
    print("완료.")
    print("생성 위치:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()