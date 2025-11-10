import os
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
import tempfile
import traceback
from dotenv import load_dotenv
from patch_model import patch_deepseek_for_cpu

load_dotenv()

MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")

if not torch.cuda.is_available():
    try:
        patch_deepseek_for_cpu()
    except Exception as e:
        print(f"패치 중 오류: {e}")

st.set_page_config(
    page_title="DeepSeek OCR",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def download_model():
    """모델을 먼저 다운로드합니다."""
    try:
        st.info(f"모델 다운로드 중: {MODEL_PATH}")
        
        # 토크나이저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # 모델 다운로드
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        st.success("모델 다운로드 완료!")
        return model, tokenizer
    except Exception as e:
        st.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

@st.cache_resource
def load_model():
    """다운로드된 모델을 디바이스에 로드합니다."""
    try:
        # 먼저 모델 다운로드
        model, tokenizer = download_model()
        if model is None or tokenizer is None:
            return None, None
        
        st.info("모델을 디바이스에 로드 중...")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "CUDA (NVIDIA GPU)"
            dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            dtype = torch.float32
        
        st.info(f"사용 디바이스: {device_name} (dtype: {dtype})")
        
        # 모델을 디바이스에 로드
        model = model.eval().to(device=device, dtype=dtype)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

def extract_text_from_image(model, tokenizer, image, prompt_type, custom_prompt=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name, format='JPEG')
            image_file = tmp_file.name
        
        if prompt_type == "free_ocr":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "
        
        with tempfile.TemporaryDirectory() as output_dir:
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
                eval_mode=True
            )
        
        try:
            os.unlink(image_file)
        except:
            pass
        
        if res is None:
            return "오류: 모델이 결과를 반환하지 않았습니다."
        
        return res
    except Exception as e:
        error_msg = f"추론 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        st.error(error_msg)
        return error_msg

def main():
    st.title("DeepSeek OCR Demo Page")
    st.markdown("이미지를 업로드하여 텍스트를 추출하거나 마크다운으로 변환하세요.")
    
    with st.sidebar:
        st.header("설정")
        st.markdown("""
        ### 사용 방법
        1. 이미지 파일을 업로드하세요
        2. OCR 모드를 선택하세요
        3. '텍스트 추출' 버튼을 클릭하세요
        
        ### OCR 모드
        - **기본 OCR**: 이미지의 모든 텍스트 추출
        - **마크다운 변환**: 문서를 마크다운 형식으로 변환
        - **사용자 정의**: 원하는 프롬프트 입력
        """)
        
        st.markdown("---")
        st.markdown("Powered by [DeepSeek OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)")
    
    with st.spinner("모델 다운로드 및 로딩 중... 처음 실행 시 시간이 걸릴 수 있습니다."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("모델을 로드할 수 없습니다. 위의 오류 메시지를 확인하세요.")
        return
    
    st.success("모델이 성공적으로 로드되었습니다!")
    
    uploaded_file = st.file_uploader(
        "이미지 파일을 업로드하세요",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("업로드된 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("OCR 설정")
            
            task_option = st.radio(
                "OCR 모드 선택:",
                ["기본 OCR", "마크다운 변환", "사용자 정의"]
            )
            
            custom_prompt = None
            if task_option == "기본 OCR":
                prompt_type = "free_ocr"
                st.info("이미지의 모든 텍스트를 추출합니다.")
            elif task_option == "마크다운 변환":
                prompt_type = "markdown"
                st.info("문서를 마크다운 형식으로 변환합니다.")
            else:
                prompt_type = "custom"
                custom_prompt = st.text_area(
                    "프롬프트를 입력하세요:",
                    placeholder="예: Extract all text from this image.",
                    value="Extract all text from this image."
                )
            
            if st.button("텍스트 추출", type="primary"):
                with st.spinner("OCR 처리 중... 이미지 크기에 따라 시간이 걸릴 수 있습니다."):
                    result = extract_text_from_image(model, tokenizer, image, prompt_type, custom_prompt)
                    
                    if result:
                        st.subheader("결과")
                        if task_option == "마크다운 변환":
                            st.markdown(result)
                        else:
                            st.text_area("추출된 텍스트", result, height=400)
                        
                        st.download_button(
                            label="결과 다운로드",
                            data=result,
                            file_name="ocr_result.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("텍스트 추출에 실패했습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()