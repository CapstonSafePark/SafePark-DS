import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from lib.config import cfg
from lib.models import get_net


device = torch.device("cpu")
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

model=get_net(cfg)
checkpoint=torch.load("weights/End-to-end.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model=model.to(device)
model.eval()

class LaneCropClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        ) 
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

CROP_CLASSIFIER_PATH = r"C:\Users\olcha\Safepark\SafePark-DS\weights\lane_crop_type_classifier.pth"
crop_checkpoint = torch.load(CROP_CLASSIFIER_PATH, map_location=device)
crop_class_names = crop_checkpoint["class_names"]
crop_classifier = LaneCropClassifier(
    num_classes=len(crop_class_names)
)
crop_classifier.load_state_dict(
    crop_checkpoint["model_state_dict"]
)
crop_classifier = crop_classifier.to(device)
crop_classifier.eval()

def make_crop_tensor(crop_bgr, size=224):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (size, size))

    crop_rgb = crop_rgb.astype("float32") / 255.0
    crop_rgb = np.transpose(crop_rgb, (2, 0, 1))

    tensor = torch.from_numpy(crop_rgb).float()
    return tensor

def extract_lane_candidate_crops(img_bgr, lane_mask, min_area=120, margin=80):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lane_mask.astype("uint8"),
        connectivity=8
    )

    h, w = lane_mask.shape
    crops = []

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        bw = stats[label_idx, cv2.CC_STAT_WIDTH]
        bh = stats[label_idx, cv2.CC_STAT_HEIGHT]

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)

        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if crop.shape[0] < 40 or crop.shape[1] < 40:
            continue

        crops.append({
            "crop": crop,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "area": int(area)
        })

    return crops

def classify_lane_crop(crop_bgr):
    x = make_crop_tensor(crop_bgr, size=224)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = crop_classifier(x)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()

    line_type = crop_class_names[pred_idx]

    return line_type, confidence

def decide_final_line_type(candidate_results):
    if not candidate_results:
        return "none", 0.0

    # confidence 낮은 것은 제외
    valid = [
        r for r in candidate_results
        if r["confidence"] >= 0.55 and r["line_type"] != "none"
    ]

    if not valid:
        return "none", 0.0

    # 우선순위: 위험/중요 차선 먼저
    priority = [
        "yellow_double",
        "white_dotted",
        "yellow_single",
        "white_solid"
    ]

    for target in priority:
        target_results = [
            r for r in valid
            if r["line_type"] == target
        ]

        if target_results:
            best = max(target_results, key=lambda x: x["confidence"])
            return target, best["confidence"]

    best = max(valid, key=lambda x: x["confidence"])
    return best["line_type"], best["confidence"]

def run_lane_detection(image_path):
    img_det = cv2.imread(image_path)

    if img_det is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    img_rgb = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    img = transform(img_resized).to(device).float()
    img = img.unsqueeze(0)

    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(img)

    ll_seg_mask = ll_seg_out.argmax(dim=1)
    mask = ll_seg_mask.squeeze().cpu().numpy().astype("uint8")

    h, w = img_det.shape[:2]
    mask_resized = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    h, w = mask_resized.shape

    roi_mask = np.zeros_like(mask_resized)

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    y1 = 0
    y2 = h

    roi_mask[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]

    lane_crops = extract_lane_candidate_crops(
        img_bgr=img_det,
        lane_mask=roi_mask,
        min_area=120,
        margin=80
    )

    candidate_results = []

    for item in lane_crops:
        crop_type, crop_confidence = classify_lane_crop(item["crop"])

        candidate_results.append({
            "line_type": crop_type,
            "confidence": float(crop_confidence),
            "box": item["box"],
            "area": item["area"]
        })

    line_type, confidence = decide_final_line_type(candidate_results)

    if confidence < 0.65:
        line_type = "unknown"

    result = {
        "line_type": line_type,
        "confidence": float(confidence),
        "candidate_count": len(candidate_results),
        "candidates": candidate_results[:10]
    }

    return result
        
   
