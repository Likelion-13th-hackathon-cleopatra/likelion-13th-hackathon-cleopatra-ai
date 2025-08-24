# --- 1. 베이스 이미지 (Python slim 버전 권장)
FROM python:3.11-slim

# --- 2. 작업 디렉토리
WORKDIR /app

# --- 3. 필요한 파일 복사
COPY requirements.txt ./

# --- 4. 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. 앱 소스 복사
COPY . .

# --- 6. 런타임에 필요한 폴더 생성 (빈 상태)
RUN mkdir -p data outputs

# --- 7. FastAPI 실행 (uvicorn)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
