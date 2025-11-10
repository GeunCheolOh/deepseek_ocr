# ============================================================
# CELL: 노트북 개요
# ============================================================
"""
# Google Colab 기반 DeepSeek-OCR 사용법 가이드

이 스크립트는 Google Colab 환경을 기준으로 DeepSeek-OCR 모델을 다루는
주피터 노트북 스타일의 실습을 제공합니다.

**주요 목표**
- 허깅페이스에서 DeepSeek-OCR 리소스 다운로드
- 텍스트 입력 기반 추론 예제 실행
- 이미지 입력 기반 VQA 워크플로우 구성
- DeepSeek-OCR 고유 태스크와 프롬프트 확인
- Gradio로 영수증 분석 애플리케이션 서빙

`DEEPSEEK_EXECUTION_MODE` 환경 변수를 활용하여
`mock`(기본) 또는 `full` 실행 모드를 선택할 수 있습니다.
"""


# ============================================================
# CELL: 환경 설정 및 임포트
# ============================================================
import os
import sys
import json
import pathlib
import textwrap
from typing import Optional, Tuple

RUN_MODE = os.getenv("DEEPSEEK_EXECUTION_MODE", "mock").lower()
BASE_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "deepseek_assets"
WORK_DIR.mkdir(exist_ok=True)


# ============================================================
# CELL: 외부 모듈 로딩
# ============================================================
try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    HfApi = snapshot_download = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = AutoTokenizer = None

try:
    import torch
except ImportError:
    torch = None

try:
    import gradio as gr
except ImportError:
    gr = None

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CELL: 유틸리티 함수 정의
# ============================================================
def log_section(title: str) -> None:
    border = "=" * 20
    print(f"\n{border} {title} {border}")


def ensure_dependency(name: str, available: bool) -> None:
    if RUN_MODE == "full" and not available:
        raise ImportError(
            f"{name} 모듈이 필요합니다. Colab에서 `pip install {name}` 명령으로 설치하세요."
        )


def describe_mode() -> None:
    log_section("실행 모드 확인")
    print(f"실행 모드: {RUN_MODE}")
    if RUN_MODE == "mock":
        print("Mock 모드: 모델 다운로드와 추론을 축약하여 빠르게 검증합니다.")
        print("실습 시에는 `DEEPSEEK_EXECUTION_MODE=full` 로 설정하세요.")
    else:
        print("Full 모드: 실제 모델 다운로드와 추론을 수행합니다.")


# ============================================================
# CELL: 패키지 설치 안내
# ============================================================
"""
# 필수 패키지 설치

Colab에서 다음 명령을 실행하여 필요한 패키지를 설치하세요.

```bash
!pip install torch transformers huggingface_hub gradio pillow
```

이 스크립트는 Mock 모드 기본값을 통해 빠른 검증이 가능하며,
Full 모드에서는 위 패키지들이 반드시 설치되어 있어야 합니다.
"""


# ============================================================
# CELL: 모델 다운로드
# ============================================================
def download_deepseek_model(model_id: str = "deepseek-ai/DeepSeek-OCR") -> pathlib.Path:
    log_section("모델 다운로드")
    if RUN_MODE == "mock":
        dummy_dir = WORK_DIR / "mock_model"
        dummy_dir.mkdir(exist_ok=True)
        (dummy_dir / "config.json").write_text(json.dumps({"model_id": model_id}))
        print("Mock 모드: 실제 모델 대신 구성 파일만 생성했습니다.")
        return dummy_dir

    ensure_dependency("huggingface_hub", HfApi is not None)
    ensure_dependency("torch", torch is not None)
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        api = HfApi()
        api.set_access_token(token)
        print("허깅페이스 토큰을 사용하여 인증했습니다.")
    cache_dir = WORK_DIR / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_files_only=False,
        revision="main",
    )
    print(f"모델이 다운로드되었습니다: {local_dir}")
    
    from patch_model import patch_deepseek_for_cpu
    if not torch.cuda.is_available():
        try:
            patch_deepseek_for_cpu()
        except Exception as e:
            print(f"패치 중 오류: {e}")
    
    return pathlib.Path(local_dir)


# ============================================================
# CELL: 모델 래퍼 정의
# ============================================================
class MockDeepSeekOCR:
    def __init__(self, model_path: pathlib.Path):
        self.model_path = model_path

    def infer(self, tokenizer, prompt: str, **kwargs) -> str:
        summary = textwrap.shorten(prompt.replace("\n", " "), width=120)
        return f"[Mock 출력] 입력 프롬프트 미리보기: {summary}"


def load_deepseek(model_path: pathlib.Path) -> Tuple[object, object]:
    log_section("모델 로딩")
    if RUN_MODE == "mock":
        print("Mock 모드: 토크나이저 없이 기본 응답을 생성합니다.")
        return MockDeepSeekOCR(model_path), None

    ensure_dependency("transformers", AutoModel is not None and AutoTokenizer is not None)
    ensure_dependency("torch", torch is not None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    model = model.eval().to(device=device, dtype=dtype)
    print(f"모델을 {device}({dtype})에 로드했습니다.")
    return model, tokenizer


# ============================================================
# CELL: 텍스트 입력 추론
# ============================================================
def demo_text_inference(model, tokenizer, image_path: Optional[pathlib.Path] = None) -> None:
    log_section("텍스트 기반 추론")
    prompt = "<image>\nFree OCR."
    image_file = str(image_path) if image_path else None
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=str(WORK_DIR / "text_only_output"),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False,
        eval_mode=True,
    )
    print("추론 결과:")
    print(result)


# ============================================================
# CELL: 샘플 이미지 생성
# ============================================================
def create_sample_image() -> pathlib.Path:
    image_path = WORK_DIR / "sample_receipt.png"
    img = Image.new("RGB", (640, 400), color="white")
    drawer = ImageDraw.Draw(img)
    lines = [
        "Store: ChatMart",
        "Date: 2025-01-01",
        "Item        Qty   Price",
        "Coffee       2    $8.00",
        "Bagel        1    $3.50",
        "Total: $11.50",
    ]
    y = 40
    for line in lines:
        drawer.text((40, y), line, fill="black")
        y += 50
    img.save(image_path)
    print(f"샘플 이미지를 생성했습니다: {image_path}")
    return image_path


# ============================================================
# CELL: 이미지 VQA 추론
# ============================================================
def demo_image_vqa(model, tokenizer, image_path: pathlib.Path) -> None:
    log_section("이미지 기반 VQA")
    prompt = "<image>\n<|grounding|>Summarize the receipt."
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path=str(WORK_DIR / "image_output"),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False,
        eval_mode=True,
    )
    print("VQA 결과:")
    print(result)


# ============================================================
# CELL: 고유 태스크 안내
# ============================================================
"""
# DeepSeek-OCR 고유 태스크 정리

DeepSeek-OCR는 다양한 프롬프트 태그를 제공하여 문서 구조화,
도표 분석, 특정 구역 인식 등을 지원합니다.

자세한 프롬프트 목록과 사용 예시는 공식 리포지터리를 참고하세요.
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
"""


# ============================================================
# CELL: 태스크 프롬프트 샘플 생성
# ============================================================
def list_special_prompts() -> None:
    log_section("지원 태스크 프롬프트")
    prompts = {
        "기본 OCR": "<image>\\nFree OCR.",
        "마크다운 변환": "<image>\\n<|grounding|>Convert the document to markdown.",
        "테이블 추출": "<image>\\n<|grounding|>Extract table cells with coordinates.",
        "도형 설명": "<image>\\nDescribe this figure in Korean.",
        "참조 기반 찾기": "<image>\\nLocate <|ref|>Invoice ID<|/ref|> in the document.",
    }
    for name, prompt in prompts.items():
        print(f"- {name}: {prompt}")


# ============================================================
# CELL: Gradio 앱 소개
# ============================================================
"""
# Gradio 영수증 분석 애플리케이션

텍스트 입력과 이미지 업로드를 받아 영수증 요약과
질의응답을 수행하는 Gradio 인터페이스를 구성합니다.

Mock 모드에서는 모델 출력을 가상으로 생성하여 동작을 검증합니다.
"""


# ============================================================
# CELL: Gradio 앱 구현
# ============================================================
def build_gradio_interface(model, tokenizer):
    def analyze_receipt(image: Optional[Image.Image], question: str) -> str:
        if image is None:
            return "이미지를 업로드하세요."
        temp_path = WORK_DIR / "gradio_input.png"
        image.convert("RGB").save(temp_path)
        prompt = f"<image>\\n{question.strip() or 'Summarize the receipt.'}"
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(temp_path),
            output_path=str(WORK_DIR / "gradio_output"),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )
        temp_path.unlink(missing_ok=True)
        return result

    if gr is None:
        if RUN_MODE == "full":
            ensure_dependency("gradio", False)
        else:
            log_section("Gradio 대체 인터페이스")
            print("Mock 모드: gradio가 설치되지 않아 간단한 대체 인터페이스를 사용합니다.")
        
        class SimpleInterface:
            def predict(self, image, question):
                return analyze_receipt(image, question)
        return SimpleInterface()

    return gr.Interface(
        fn=analyze_receipt,
        inputs=[
            gr.Image(type="pil", label="영수증 이미지 업로드"),
            gr.Textbox(label="질문", placeholder="예: 총 금액이 얼마인지 알려줘."),
        ],
        outputs=gr.Textbox(label="모델 응답"),
        title="DeepSeek-OCR 영수증 분석",
        description="텍스트 + 이미지 기반 영수증 분석 데모",
    )


# ============================================================
# CELL: Gradio 동작 검증
# ============================================================
def test_gradio_app(interface, sample_image: pathlib.Path) -> None:
    log_section("Gradio 검증")
    if interface is None:
        print("오류: interface가 None입니다.")
        return
    
    if hasattr(interface, 'predict') and callable(interface.predict):
        test_result = interface.predict(Image.open(sample_image), "총 금액이 얼마야?")
        print("Gradio 예측 결과:")
        print(test_result)
    else:
        print("Gradio 인터페이스가 생성되었습니다.")
        print("브라우저에서 테스트하려면:")
        print("  interface.launch()")
        print("명령을 실행하세요.")


# ============================================================
# CELL: 메인 실행
# ============================================================
def main() -> None:
    describe_mode()
    model_dir = download_deepseek_model()
    model, tokenizer = load_deepseek(model_dir)
    sample_image = create_sample_image()
    demo_text_inference(model, tokenizer, sample_image)
    demo_image_vqa(model, tokenizer, sample_image)
    list_special_prompts()
    interface = build_gradio_interface(model, tokenizer)
    test_gradio_app(interface, sample_image)
    log_section("실습 완료")
    print("모든 단계를 성공적으로 실행했습니다.")


if __name__ == "__main__":
    main()


