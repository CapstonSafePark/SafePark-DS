import easyocr
import re

reader = easyocr.Reader(['ko', 'en'], gpu = False)

def extract_text_from_image(image_path):
    results = reader.readtext(image_path, detail = 0)
    ocr_text = "\n".join(results)
    return ocr_text

def parse_parking_fee_policy(ocr_text):
    text = ocr_text.replace(",", "")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    basic_minutes = 30
    basic_fee = None
    extra_unit_minutes = 10
    extra_unit_fee = None
    daily_max_fee = None

    #기본요금 검색
    for i, line in enumerate(lines):
        if "기본" in line and "요금" in line:
            minute_match = re.search(r"(\d+)\s*분", line)
            if minute_match:
                basic_minutes = int(minute_match.group(1))

            fee_match = re.search(r"(\d+)\s*원", line)
            if fee_match:
                basic_fee = int(fee_match.group(1))
            elif i + 1 < len(lines):
                next_fee_match = re.search(r"(\d+)\s*원", lines[i + 1])
                if next_fee_match:
                    basic_fee = int(next_fee_match.group(1))
    
    #추가요금 검색
    for i, line in enumerate(lines):
        if "초과" in line or "추가" in line or "분당" in line:
            minute_match = re.search(r"(\d+)\s*분", line)
            if minute_match:
                extra_unit_minutes = int(minute_match.group(1))

            fee_match = re.search(r"(\d+)\s*원", line)
            if fee_match:
                extra_unit_fee = int(fee_match.group(1))
            elif i + 1 < len(lines):
                next_fee_match = re.search(r"(\d+)\s*원", lines[i + 1])
                if next_fee_match:
                    extra_unit_fee = int(next_fee_match.group(1))

    #1일 최대요금 검색
    for i, line in enumerate(lines):
        if "1일" in line or "종일" in line:
            for near_line in lines[i:i + 5]:
                fee_match = re.search(r"(\d+)\s*원", near_line)
                if fee_match:
                    daily_max_fee = int(fee_match.group(1))
                    break

    if basic_fee is None:
        basic_fee = 500

    if extra_unit_fee is None:
        extra_unit_fee = 300


    return {
        "basic_minutes": basic_minutes,
        "basic_fee": basic_fee,
        "extra_unit_minutes": extra_unit_minutes,
        "extra_unit_fee": extra_unit_fee,
        "daily_max_fee": daily_max_fee
    }

def calculate_parking_fee(image_path, duration_minutes):
    try:
        ocr_text = extract_text_from_image(image_path)
    except Exception as e:
        return {
            "error": "OCR failed",
            "message": str(e),
            "image_path": image_path
        }

    policy = parse_parking_fee_policy(ocr_text)

    basic_minutes = policy["basic_minutes"]
    basic_fee = policy["basic_fee"]
    extra_unit_minutes = policy["extra_unit_minutes"]
    extra_unit_fee = policy["extra_unit_fee"]
    daily_max_fee = policy["daily_max_fee"]

    if duration_minutes <= basic_minutes:
        fee = basic_fee
        breakdown = f"기본 {basic_minutes}분 {basic_fee}원"
    else:
        extra_minutes = duration_minutes - basic_minutes

        extra_units = (extra_minutes + extra_unit_minutes - 1) // extra_unit_minutes
        extra_fee = extra_units * extra_unit_fee

        fee = basic_fee + extra_fee

        breakdown = (
            f"기본 {basic_minutes}분 {basic_fee}원 + "
            f"추가 {extra_minutes}분"
            f"({extra_units}회 * {extra_unit_fee}원) = {extra_fee}원"
        )

    if daily_max_fee is not None and fee > daily_max_fee:
        fee = daily_max_fee
        breakdown += f" → 1일 최대요금 {daily_max_fee}원 적용"
    
    return {
        "fee": fee,
        "breakdown": breakdown,
        "policy": policy,
        "ocr_text": ocr_text,
        "image_path": image_path
    }