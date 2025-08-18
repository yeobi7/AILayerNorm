import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

# 모델과 프로세서 로드
model_name = "timm/swin_tiny_patch4_window7_224.ms_in1k"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
model.eval()

def get_layernorm_input_and_params(model, pixel_values):
    """
    모델의 마지막 LayerNorm 모듈에 forward hook을 등록하여
    해당 모듈에 들어가는 입력값과 gamma (weight), beta (bias) 파라미터를 추출합니다.
    """
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]

    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values=pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # shape: [768]
    beta  = final_norm.bias.detach()      # shape: [768]
    return hook_data["layernorm_input"], gamma, beta

def quantize_fp32_to_int8(x):
    """
    fp32 텐서를 symmetric quantization (zero_point=0) 방식으로 int8로 변환합니다.
    scale은 해당 텐서의 절대 최대값을 기준으로 계산합니다.
    """
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale

# 예제 이미지 로드 (경로를 실제 이미지 경로로 수정)
image = Image.open(r"C:\SYDLAB_SY\test_image_cat.jpg")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# 원래의 LayerNorm 입력값과 gamma, beta 추출
layernorm_input, gamma, beta = get_layernorm_input_and_params(model, pixel_values)
print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
print("원래 Gamma shape:", gamma.shape)                      # [768]
print("원래 Beta shape:", beta.shape)                        # [768]

# --- 평탄한 입력에서 512개 값을 선택하고, 그에 맞는 gamma, beta 추출 ---
flattened_input = layernorm_input.flatten()  # 전체 크기: 7*7*768 = 37632
total = flattened_input.numel()

# 중간 부분에서 512개 연속 값을 선택 (원하는 위치로 start_index를 조절 가능)
start_index = (total - 512) // 2
selected_input = flattened_input[start_index : start_index + 512]

# 각 선택된 값이 어느 채널에서 왔는지 계산: 
# 원래 입력의 마지막 차원(채널)이 768이므로, 인덱스 mod 768가 해당 채널 번호입니다.
indices = torch.arange(start_index, start_index + 512)
channels = indices % 768  # 각 요소의 채널 번호

# 선택된 입력에 대응하는 gamma와 beta 추출
selected_gamma = gamma[channels]
selected_beta  = beta[channels]

# 양자화 수행 (fp32 -> int8)
selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
selected_beta_int8, scale_beta   = quantize_fp32_to_int8(selected_beta)


def tensor_to_verilog_hex(tensor):
    """
    tensor (1차원, int8 값들이 들어있는) 를 Verilog 16진수 리터럴 형식의 문자열로 변환합니다.
    각 값은 2자리 16진수로 표현되며, 음수의 경우 두의 보수 표현(0~255 범위)으로 변환합니다.
    예) 3 -> "03", -35 -> (0xDD) -> "DD"
    """
    hex_strs = []
    for val in tensor:
        ival = int(val)
        if ival < 0:
            ival = ival & 0xFF
        hex_strs.append("{:02X}".format(ival))
    return "_".join(hex_strs)

# (예시) 선택된 int8 tensor들: selected_input_int8, selected_gamma_int8, selected_beta_int8
# 이미 이전 코드에서 512개씩 선택했다고 가정합니다.
i_x_hex     = tensor_to_verilog_hex(selected_input_int8)
i_gamma_hex = tensor_to_verilog_hex(selected_gamma_int8)
i_beta_hex  = tensor_to_verilog_hex(selected_beta_int8)

# 512 int8 값은 512*8 = 4096비트이므로, 4096비트 리터럴로 출력합니다.
print("i_x = 4096'h" + i_x_hex + ";")
print("i_gamma = 4096'h" + i_gamma_hex + ";")
print("i_beta = 4096'h" + i_beta_hex + ";")



print("\n선택된 LayerNorm 입력 (512 int8 값, 4096비트):\n", selected_input_int8)
print("입력 scale:", scale_input)
print("\n선택된 Gamma (512 int8 값):\n", selected_gamma_int8)
print("Gamma scale:", scale_gamma)
print("\n선택된 Beta (512 int8 값):\n", selected_beta_int8)
print("Beta scale:", scale_beta)
