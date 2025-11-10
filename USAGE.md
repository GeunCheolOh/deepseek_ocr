# DeepSeek OCR 사용 가이드

## 빠른 시작

### 1. 애플리케이션 실행

```bash
cd /Users/ogeuncheol/Documents/project/deepseek_ocr
source venv/bin/activate
streamlit run app.py
```

자동으로 브라우저가 열리며 `http://localhost:8501`로 접속됩니다.

### 2. 이미지 업로드

1. 웹 인터페이스에서 "이미지 파일을 업로드하세요" 버튼 클릭
2. JPG, PNG, BMP, WEBP 형식의 이미지 선택
3. 이미지가 왼쪽 화면에 표시됩니다

### 3. OCR 모드 선택

#### 기본 OCR
- 이미지의 모든 텍스트를 추출합니다
- 간단한 문서나 스크린샷에 적합

#### 마크다운 변환
- 문서를 구조화된 마크다운 형식으로 변환
- 표, 제목, 리스트 등의 구조를 유지
- 논문, 보고서, 복잡한 문서에 적합

#### 사용자 정의
- 원하는 프롬프트를 직접 입력
- 예시:
  - "Extract all text from this image."
  - "Summarize the main points in this document."
  - "Extract only the email addresses."

### 4. 텍스트 추출

"텍스트 추출" 버튼을 클릭하면:
- 모델이 이미지를 분석합니다
  - MPS (Apple Silicon): 10-30초
  - CUDA (NVIDIA GPU): 5-15초
  - CPU: 1-3분
- 추출된 텍스트가 화면에 표시됩니다
- "결과 다운로드" 버튼으로 텍스트 파일로 저장 가능

## 성능 최적화 팁

### 하드웨어 가속
애플리케이션 시작 시 사용 중인 디바이스가 표시됩니다:
- **MPS (Apple Silicon GPU)**: M1/M2/M3 Mac에서 자동 활성화, 가장 빠름
- **CUDA (NVIDIA GPU)**: NVIDIA GPU 사용 시 자동 활성화
- **CPU**: 다른 하드웨어에서 사용, 가장 느림

### 이미지 해상도
- 권장: 1024px ~ 2048px
- 너무 큰 이미지는 처리 시간이 오래 걸립니다
- 너무 작은 이미지는 정확도가 떨어질 수 있습니다

### 문서 타입별 권장 설정

**영수증/명함**
- 모드: 기본 OCR
- 해상도: 작게 유지

**논문/보고서**
- 모드: 마크다운 변환
- 해상도: 높게 유지

**표가 많은 문서**
- 모드: 마크다운 변환
- 프롬프트: "Convert to markdown, preserve table structure."

## 문제 해결

### 메모리 부족 오류

`app.py` 파일을 수정하여 이미지 크기를 줄이세요:

```python
res = model.infer(
    ...
    base_size=512,      # 1024 -> 512로 변경
    image_size=320,     # 640 -> 320으로 변경
    ...
)
```

### 처리가 너무 느림

- 이미지 해상도를 낮추세요
- macOS CPU 모드에서는 2-5분 정도 걸릴 수 있습니다
- 첫 실행은 모델 로딩으로 더 오래 걸립니다

### 모델 다운로드 실패

```bash
# 캐시 삭제 후 재시도
rm -rf ~/.cache/huggingface
```

## 애플리케이션 종료

터미널에서 `Ctrl + C`를 눌러 종료합니다.

## 지원되는 언어

DeepSeek-OCR은 다국어를 지원합니다:
- 한국어, 영어, 중국어, 일본어 등
- 혼합된 언어도 처리 가능

## 추가 정보

- 공식 모델: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- 논문: https://arxiv.org/abs/2510.18234

