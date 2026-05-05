# SafePark-DS

## 실행 방법

1. 필요한 라이브러리 설치

pip install flask torch torchvision opencv-python numpy

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
