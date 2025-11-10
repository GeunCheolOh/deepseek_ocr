import os
import glob
import re
import torch

def patch_deepseek_for_cpu():
    """
    DeepSeek-OCR 모델을 CPU에서 작동하도록 패치합니다.
    CUDA가 사용 가능한 경우 패치를 건너뜁니다.
    """
    if torch.cuda.is_available():
        print("CUDA가 감지되었습니다. CPU 패치를 건너뜁니다.")
        return True
    
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR/*/modeling_deepseekocr.py"
    )
    
    model_files = glob.glob(cache_pattern)
    
    if not model_files:
        print("모델 파일을 찾을 수 없습니다. 모델이 아직 다운로드되지 않았을 수 있습니다.")
        return False
    
    for file_path in model_files:
        backup_path = file_path + ".backup"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"백업 생성: {backup_path}")
        else:
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"백업에서 복원: {file_path}")
        
        content = re.sub(r'\.cuda\(\)', '.to(next(self.parameters()).device)', content)
        content = re.sub(r'\.to\(device\)', '.to(next(self.parameters()).device)', content)
        content = re.sub(r'\.to\(torch\.bfloat16\)', '.to(next(self.parameters()).dtype)', content)
        content = content.replace('torch.autocast("cuda"', 'torch.autocast("cpu"')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"CPU 패치 적용 완료: {file_path}")
        return True
    
    return True

if __name__ == "__main__":
    patch_deepseek_for_cpu()

