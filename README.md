# DeepSeek OCR 웹 애플리케이션

DeepSeek AI의 DeepSeek-OCR 모델을 사용하여 이미지에서 텍스트를 추출하고 문서를 마크다운으로 변환하는 Streamlit 기반 웹 애플리케이션입니다.

공식 모델: [DeepSeek-OCR on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## 기능

- 이미지에서 텍스트 추출 (Free OCR)
- 문서를 마크다운 형식으로 변환
- 사용자 정의 프롬프트 지원
- 추출된 텍스트 다운로드

## 설치 방법

### 1. 가상 환경 생성 및 활성화 (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 실행 방법

```bash
source venv/bin/activate  # 가상 환경이 활성화되지 않은 경우
streamlit run app.py
```

브라우저에서 자동으로 열리지 않는 경우 `http://localhost:8501`로 접속하세요.

## 시스템 요구사항

- Python 3.10 이상
- 최소 16GB RAM (32GB 권장)
- macOS (Apple Silicon 또는 Intel)
- 약 6GB의 디스크 공간 (모델 다운로드용)

## 하드웨어 가속

현재 CPU 모드로 작동합니다:

- **CPU**: 안정적으로 작동 (느리지만 확실함)
- **NVIDIA GPU**: CUDA 지원 (별도 설정 필요)
- **Apple Silicon (M1/M2/M3)**: MPS는 현재 모델과 호환 문제로 비활성화됨

## 주의사항

- 첫 실행 시 모델 다운로드로 인해 시간이 소요됩니다 (약 6GB)
- CPU 모드에서 작동하므로 처리 속도가 느립니다 (이미지당 2-5분)
- 고해상도 이미지는 처리 시간이 더 오래 걸립니다
- 첫 OCR 실행 시 모델 초기화로 시간이 추가로 소요됩니다

## OCR 모드

### 기본 OCR
이미지의 모든 텍스트를 추출합니다.

### 마크다운 변환
문서를 구조화된 마크다운 형식으로 변환합니다.

### 사용자 정의
원하는 프롬프트를 직접 입력하여 특정 작업을 수행할 수 있습니다.

## 문제 해결

### CUDA 오류 (macOS)
첫 실행 시 자동으로 CPU 모드로 패치됩니다. 만약 "Torch not compiled with CUDA enabled" 오류가 발생하면:

```python
# fix_cuda.py 실행
import re
file_path = "/Users/ogeuncheol/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR/*/modeling_deepseekocr.py"
# .cuda() 제거 및 autocast cpu로 변경
```

또는 캐시를 삭제하고 재실행하세요:
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR
```

### 메모리 부족 오류
`app.py`에서 `base_size`와 `image_size` 값을 줄이세요:
```python
base_size=512,  # 1024에서 512로
image_size=320   # 640에서 320으로
```

### 모델 로드 실패
인터넷 연결을 확인하고 Hugging Face에서 모델 다운로드가 가능한지 확인하세요.

