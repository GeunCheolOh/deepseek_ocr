#!/bin/bash

echo "======================================"
echo "DeepSeek OCR macOS 설치 스크립트"
echo "======================================"

# 가상 환경 생성
if [ ! -d "venv" ]; then
    echo "가상 환경 생성 중..."
    python3 -m venv venv
fi

# 가상 환경 활성화
source venv/bin/activate

# pip 업그레이드
echo "pip 업그레이드 중..."
pip install --upgrade pip

# 패키지 설치
echo "필수 패키지 설치 중..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "설치 완료!"
echo "======================================"
echo ""
echo "사용 방법:"
echo "  1. 가상 환경 활성화: source venv/bin/activate"
echo "  2. 애플리케이션 실행: streamlit run app.py"
echo "  3. 브라우저에서 http://localhost:8501 접속"
echo ""
echo "주의: 첫 실행 시 모델 다운로드 및 CPU 패치가 자동으로 진행됩니다."
echo ""

