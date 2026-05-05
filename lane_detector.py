import cv2
import numpy as np
import torch
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

def run_lane_detection(image_path):
    img_det = cv2.imread(image_path)

    if img_det is None:
        return{"error"}
    img_rgb = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img = transform(img_resized).to(device).float()
    img = img.unsqueeze(0)
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(img)
    ll_seg_mask = ll_seg_out.argmax(dim=1) #0을 배경, 1을 차선으로 설정
    mask = ll_seg_mask.squeeze().cpu().numpy().astype('uint8') #실사용할 mask 데이터로 변환
    h, w = img_det.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    #ROI 설정
    h, w = mask_resized.shape
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)
    y1 = int(h * 0.30)
    y2 = int(h * 0.95)

    roi_mask = mask_resized[y1:y2, x1:x2]
    lane_pixels = img_det[mask_resized == 1] #차선 위치인 1만 선택

    #차선 색상 분류
    roi_img = img_det[y1:y2, x1:x2]

    hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    #노란색 차선
    yellow_color_mask = (
        (hsv_roi[:, :, 0] >= 10) &
        (hsv_roi[:, :, 0] <= 45) &
        (hsv_roi[:, :, 1] >= 25) &
        (hsv_roi[:, :, 2] >= 80)
    )

    #흰색 차선
    white_color_mask = (
        (hsv_roi[:, :, 1] <= 60) &
        (hsv_roi[:, :, 2] >= 140)
    )

    yellow_lane_mask = ((roi_mask == 1) & yellow_color_mask).astype('uint8')
    white_lane_mask = ((roi_mask == 1) & white_color_mask).astype('uint8')

    kernel = np.ones((3, 3), np.uint8)

    yellow_lane_mask = cv2.dilate(yellow_lane_mask, kernel, iterations = 1)
    white_lane_mask = cv2.dilate(white_lane_mask, kernel, iterations = 1)

    #차선 개수 판단
    def count_lines_by_mask(binary_mask, area_threshold = 200):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask.astype('uint8'),
            connectivity = 8
        )

        line_count = 0

        areas = []

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            #이상값으로 추정되는 노이즈는 제외
            if area > area_threshold:
                areas.append(area)
        
        areas.sort(reverse=True)

        return len(areas), areas

    def is_dotted_line(binary_mask, area_threshold = 200):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask.astype('uint8'),
            connectivity = 8
        )

        dotted_candidates = 0
        solid_candidates = 0

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]

            if area < area_threshold:
                continue
                    
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]

            component = binary_mask[y:y+h, x:x+w]

            row_sum = np.sum(component, axis=1)
            line_rows = row_sum > 0
            empty_ratio = np.sum(~line_rows) / len(line_rows)

            if empty_ratio > 0.45:
                dotted_candidates += 1
            else:
                solid_candidates += 1

        return dotted_candidates > solid_candidates and dotted_candidates > 0
            

    yellow_count, yellow_areas = count_lines_by_mask(yellow_lane_mask, area_threshold = 100)
    yellow_count_for_type = min(yellow_count, 2)
    valid_yellow_areas = [int(a) for a in yellow_areas if int(a) > 150]
    valid_yellow_count = len(valid_yellow_areas)
    valid_yellow_count_for_type = min(valid_yellow_count, 2)
    white_count, white_areas = count_lines_by_mask(white_lane_mask, area_threshold = 200)

    white_is_dotted = is_dotted_line(white_lane_mask)

    yellow_area_sum = sum(valid_yellow_areas)
    white_area_sum = sum([int(a) for a in white_areas])

    white_is_dotted_by_count = white_count >= 4
    white_is_dotted_final = white_is_dotted or white_is_dotted_by_count

    if valid_yellow_count_for_type >= 2 or (yellow_count >= 2 and yellow_area_sum > 350):
        line_type = "yellow_double"
    elif white_count >= 2 and white_is_dotted_final and white_area_sum >= yellow_area_sum:
        line_type = "white_dotted"
    elif valid_yellow_count_for_type == 1 and yellow_area_sum > 8000:
        line_type = "yellow_single"
    elif white_count >= 1 and white_is_dotted_final:
        line_type = "white_dotted"
    else:
        line_type = "none"
        
    result = {
        "line_type": line_type,
        "yellow_count": int(yellow_count),
        "valid_yellow_count": int(valid_yellow_count),
        "white_count": int(white_count),
        "total_count": int(valid_yellow_count + white_count),
        "white_is_dotted": bool(white_is_dotted_final),
        "white_is_dotted_by_count": bool(white_is_dotted_by_count),
        "yellow_area_sum": int(yellow_area_sum),
        "white_area_sum": int(white_area_sum)
    }
    return result
