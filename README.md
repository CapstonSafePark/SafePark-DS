# SafePark-DS

## 실행 방법

1. 필요한 라이브러리 설치

pip install flask torch torchvision opencv-python numpy easyocr

2. 모델 파일 준비
weight 폴더를 생성 후 아래 파일 삽입

weights/End-to-end.pth

3. 서버 실행
python app.py

실행 후 아래 주소로 api 요청 

http://127.0.0.1:5000/ds/line-detect

응답 예시 
{
  "line_type": "yellow_double",
  "yellow_count": 2,
  "valid_yellow_count": 2,
  "white_count": 3,
  "total_count": 5,
  "white_is_dotted": false
}

4. weight파일은 카카오톡으로 공유 해놓겠습니다.

5. 요금 계산 API
요청 주소
POST http://127.0.0.1:5000/ds/parking_fee

요청 방식
form-data 형식으로 전송합니다.
KEY   Type    설명
image   File    주차 요금 안내판 이미지
duration_minutes    Text    주차 시간(분)

요청 예시
curl.exe -X POST http://127.0.0.1:5000/ds/parking_fee `
  -F "image=@C:\Users\olcha\Safepark\YOLOP\pictures\test.jpg" `
  -F "duration_minutes=90"

응답 예시
{
  "fee": 9000,
  "breakdown": "기본 30분 3000원 + 추가 60분(6회 * 1000원) = 6000원",
  "policy": {
    "basic_minutes": 30,
    "basic_fee": 3000,
    "extra_unit_minutes": 10,
    "extra_unit_fee": 1000,ㄴ
    "daily_max_fee": 45000
  },
  "ocr_text": "이용시간 및 요금안내\n기본요금(30분)\n3,000원\n초과시 10분당\n1,000원",
  "image_path": "uploads\\test.jpg"
}

