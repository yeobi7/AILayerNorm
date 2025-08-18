#########################################################################################################################

######################################### AILayerNorm SW Test (512 Input-4096 bit)#######################################

#########################################################################################################################

'''
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib

# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()  # 입력값 추출
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # [channels] (예: [768])
    beta  = final_norm.bias.detach()      # [channels]
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 하드웨어/소프트웨어용 함수들 (AILayerNorm SW 코드)
# =============================================================================
def lower8(x):
    """
    주어진 실수 x를 정수로 변환한 후,
    하위 8비트를 그대로 추출하여 부호 있는 int8 값으로 해석합니다.
    (예: 190 → (190 & 0xFF)=190 → 190>=128이면 190-256 = -66)
    """
    x_int = int(round(x))
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def lower8_hw(x):
    """
    하드웨어 방식과 동일하게 정수 x의 하위 8비트를 부호 있는 int8으로 변환합니다.
    """
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s



def dynamic_compress_D(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s



def lut_std(temp_var):
    """
    LUT를 통한 stdinv 근사 (Q0.8 값):
      temp_var: 분산의 하위 16비트 값 (0~65535)
      원래 LUT 매핑 값은:
         1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 255
      여기서는 그대로 반환합니다.
    """
    if temp_var >= 32768:
        return 1
    elif temp_var >= 16384:
        return 2
    elif temp_var >= 8192:
        return 3
    elif temp_var >= 4096:
        return 4
    elif temp_var >= 2048:
        return 6
    elif temp_var >= 1024:
        return 8
    elif temp_var >= 512:
        return 11
    elif temp_var >= 256:
        return 16
    elif temp_var >= 128:
        return 23
    elif temp_var >= 64:
        return 32
    elif temp_var >= 32:
        return 45
    elif temp_var >= 16:
        return 64
    elif temp_var >= 8:
        return 90
    elif temp_var >= 4:
        return 128
    elif temp_var >= 2:
        return 181
    elif temp_var >= 1:
        return 255
    else:
        return 0

# =============================================================================
# Variant A: 분산이 음수일 경우 0으로 매핑 (기존 방식)
# =============================================================================
def stage1_detailed_variantA(X, alpha, zp, inv_n):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
    #    variance = 0
        stdinv_Q = 0
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
   # stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant B: 분산이 음수일 경우 이전 분산 값을 유지
# =============================================================================
prev_variance = None
def stage1_detailed_variantB(X, alpha, zp, inv_n):
    global prev_variance
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    if variance < 0:
        if prev_variance is not None:
            variance = prev_variance
        else:
            variance = 0
    else:
        prev_variance = variance
    temp_var = int(variance) & 0xFFFF
    stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant C: 분산이 음수일 경우 1로 매핑
# =============================================================================
def stage1_detailed_variantC(X, alpha, zp, inv_n, small_const=1):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
      #  variance = small_const
        stdinv_Q = 1
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
  #  stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant D: Dynamic Compression + RMSNorm (평균을 빼지 않고 RMS 기반 정규화)
# =============================================================================
def stage1_detailed_variantD(X, alpha, zp, inv_n, eps=1):
    """
    Variant D: RMSNorm 방식.
    - 입력 X에 대해 zero point 제거 없이 각 요소를 그대로 사용합니다.
    - RMSNorm은 평균을 빼지 않고, x의 제곱 평균의 제곱근 (RMS)을 이용합니다.
    - dynamic compression을 사용해 x의 제곱 값을 근사한 후, RMS를 계산합니다.
    """
    C = len(X)
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x)  # RMSNorm에서는 평균 제거를 하지 않습니다.
        Xc, s = dynamic_compress_D(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex2_avg = (Ex2_acc << 4) / C
    # RMS = sqrt(Ex2_avg + eps)
    rms = math.sqrt(Ex2_avg + eps)
    stdinv = 1 / rms
    mu_val = 0  # RMSNorm는 평균을 사용하지 않으므로 mu=0

    debug_info = {
        "Ex2_avg": Ex2_avg,
        "rms": rms,
        "stdinv": stdinv
    }
    # ex_unit_out, ex2_unit_out은 RMSNorm에서 사용하지 않음
    ex_unit_out = None
    ex2_unit_out = None
    return mu_val, stdinv, 0, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Stage2 및 전체 AILayerNorm 함수 (variant에 따라 호출)
# =============================================================================
def stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=True):
    """
    Stage2:
      각 채널에 대해:
         1. X_norm_shift = X[i] << alpha (zero point 제거 없음)
         2. X_diff = X_norm_shift - μ   (RMSNorm의 경우 mu=0)
         3. temp_Norm = gamma[i] * stdinv * X_diff
         4. 최종 출력: 
              - 일반적인 경우: o_Norm = lower8_hw(temp_Norm) + beta[i]
              - RMSNorm (Variant D)의 경우: o_Norm = lower8_hw(temp_Norm)  (beta 미사용)
    """
    Y = []
    for i, x in enumerate(X):
        Xi = int(x)
        X_norm_shift = Xi << alpha
        X_diff = X_norm_shift - mu
        temp_Norm = gamma[i] * stdinv * X_diff
        if add_beta:
            final = lower8_hw(temp_Norm) + beta[i]
        else:
            final = lower8_hw(temp_Norm)
        Y.append(final)
    return np.array(Y, dtype=np.int8)

def AILayerNorm(X, alpha, zp, gamma, beta, inv_n, variant="A", small_const=1):
    if variant == "A":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantA(X, alpha, zp, inv_n)
        add_beta = True
    elif variant == "B":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantB(X, alpha, zp, inv_n)
        add_beta = True
    elif variant == "C":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantC(X, alpha, zp, inv_n, small_const)
        add_beta = True
    elif variant == "D":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantD(X, alpha, zp, inv_n)
        add_beta = False  # RMSNorm: beta 미사용
    else:
        raise ValueError("알 수 없는 variant입니다. 'A', 'B', 'C', 'D' 중 하나를 선택하세요.")
    Y = stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=add_beta)
    return Y, mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info


def original_layernorm(X, gamma, beta, eps=1e-5):
    X = np.array(X, dtype=float)
    mean = np.mean(X)
    var = np.var(X, ddof=0)
    normalized = (X - mean) / math.sqrt(var + eps)
    out = normalized * gamma + beta
    quantized = np.array([lower8_hw(val) for val in out], dtype=np.int8)
    return quantized, mean, var

def display_full_input(X_full):
    width = len(X_full) * 8
    hex_str = "_".join(f"{x:02X}" for x in X_full)
    dec_str = "_".join(str(x) for x in X_full)
    print(f"Input X = {width}'h{hex_str};")
    print(f"// {width}'d{dec_str}")

def quantize_fp32_to_int8(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale

def tensor_to_verilog_hex(tensor):
    hex_strs = []
    for val in tensor:
        ival = int(val)
        if ival < 0:
            ival = ival & 0xFF
        hex_strs.append("{:02X}".format(ival))
    return "_".join(hex_strs)

# =============================================================================
# Swin 모델에서 LayerNorm 입력 및 파라미터 추출
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()
    beta  = final_norm.bias.detach()
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 메인 실행부
# =============================================================================
if __name__ == '__main__':
    # 이미지 로드 및 전처리 (timm transform 사용)
    image = Image.open(r"C:\SYDLAB_SY\test_image_road.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 원래 LayerNorm 입력 및 파라미터 추출
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("원래 Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("원래 Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화 후 중앙 512개 값 선택 (4096비트)
    flattened_input = layernorm_input.flatten()
    total = flattened_input.numel()
    start_index = (total) // 2
    #start_index = 
    selected_input = flattened_input[start_index : start_index + 512]

    # 각 선택된 값의 채널 번호 계산 (마지막 차원 크기가 768)
    indices = torch.arange(start_index, start_index + 512)
    channels = indices % gamma_tensor.numel()
    selected_gamma = gamma_tensor[channels]
    selected_beta = beta_tensor[channels]

    # fp32 -> int8 양자화
    selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
    selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
    selected_beta_int8, scale_beta = quantize_fp32_to_int8(selected_beta)

    i_x_hex     = tensor_to_verilog_hex(selected_input_int8)
    i_gamma_hex = tensor_to_verilog_hex(selected_gamma_int8)
    i_beta_hex  = tensor_to_verilog_hex(selected_beta_int8)

    print("\ni_x = 4096'h" + i_x_hex + ";")
    print("i_gamma = 4096'h" + i_gamma_hex + ";")
    print("i_beta = 4096'h" + i_beta_hex + ";")

    print("\n선택된 LayerNorm 입력 (512 int8 값, 4096비트):\n", selected_input_int8)
    print("입력 scale:", scale_input)
    print("\n선택된 Gamma (512 int8 값):\n", selected_gamma_int8)
    print("Gamma scale:", scale_gamma)
    print("\n선택된 Beta (512 int8 값):\n", selected_beta_int8)
    print("Beta scale:", scale_beta)

    # AILayerNorm 파라미터 설정 (입력이 이미 -128~127이므로 zp = 0)
    zp = 0
    alpha = 2
    inv_n = 32

    # 선택된 값들을 numpy 배열로 변환 (512개의 int8 값)
    X_full = selected_input_int8.cpu().numpy()
    gamma_full = selected_gamma_int8.cpu().numpy()
    beta_full = selected_beta_int8.cpu().numpy()

    # Variant별 결과 저장용 딕셔너리 및 MAE, Accuracy 저장 리스트
    results_A = {}
    results_B = {}
    results_C = {}
    results_D = {}  # RMSNorm variant (Variant D)
    mae_list_A = []
    mae_list_B = []
    mae_list_C = []
    mae_list_D = []
    var_list_A = []
    var_list_B = []
    var_list_C = []
    # RMSNorm variant는 별도의 분산 계산이 없으므로 var_list_D는 RMS 값로 대체할 수 있습니다.
    rms_list_D = []
    orig_results = {}
    batches = 512 // 8  # 총 64 배치

    for i in range(batches):
        X = X_full[i*8:(i+1)*8]
        gamma_batch = gamma_full[i*8:(i+1)*8]
        beta_batch = beta_full[i*8:(i+1)*8]

        # Variant A
        Y_A, mu_A, stdinv_A, Ex_acc_A, Ex2_acc_A, ex_out_A, ex2_out_A, debug_info_A = AILayerNorm(X, alpha, zp, gamma_batch, beta_batch, inv_n, variant="A")
        orig_Y, orig_mean, orig_var = original_layernorm(X, gamma_batch, beta_batch)
        mae_A = np.mean(np.abs(orig_Y - Y_A))
        accuracy_A = (1 - (mae_A / 255)) * 100
        mae_list_A.append(mae_A)
        var_list_A.append(debug_info_A["variance"])
        results_A[f"Batch {i+1}"] = {
            "Input": X,
            "Stage1_mu": mu_A,
            "Stage1_std (1/stdinv)": (1/stdinv_A if stdinv_A != 0 else 0),
            "Final Y (Variant A)": Y_A,
            "MAE": mae_A,
            "Accuracy (%)": accuracy_A,
            "Variance": debug_info_A["variance"]
        }
        orig_results[f"Batch {i+1}"] = {
            "Original LayerNorm Output": orig_Y,
            "Orig_mean": orig_mean,
            "Orig_var": orig_var
        }

        # Variant B
        Y_B, mu_B, stdinv_B, Ex_acc_B, Ex2_acc_B, ex_out_B, ex2_out_B, debug_info_B = AILayerNorm(X, alpha, zp, gamma_batch, beta_batch, inv_n, variant="B")
        mae_B = np.mean(np.abs(orig_Y - Y_B))
        accuracy_B = (1 - (mae_B / 255)) * 100
        mae_list_B.append(mae_B)
        var_list_B.append(debug_info_B["variance"])
        results_B[f"Batch {i+1}"] = {
            "Input": X,
            "Stage1_mu": mu_B,
            "Stage1_std (1/stdinv)": (1/stdinv_B if stdinv_B != 0 else 0),
            "Final Y (Variant B)": Y_B,
            "MAE": mae_B,
            "Accuracy (%)": accuracy_B,
            "Variance": debug_info_B["variance"]
        }

        # Variant C
        Y_C, mu_C, stdinv_C, Ex_acc_C, Ex2_acc_C, ex_out_C, ex2_out_C, debug_info_C = AILayerNorm(X, alpha, zp, gamma_batch, beta_batch, inv_n, variant="C", small_const=1)
        mae_C = np.mean(np.abs(orig_Y - Y_C))
        accuracy_C = (1 - (mae_C / 255)) * 100
        mae_list_C.append(mae_C)
        var_list_C.append(debug_info_C["variance"])
        results_C[f"Batch {i+1}"] = {
            "Input": X,
            "Stage1_mu": mu_C,
            "Stage1_std (1/stdinv)": (1/stdinv_C if stdinv_C != 0 else 0),
            "Final Y (Variant C)": Y_C,
            "MAE": mae_C,
            "Accuracy (%)": accuracy_C,
            "Variance": debug_info_C["variance"]
        }

        # Variant D: Dynamic Compression + RMSNorm (평균을 사용하지 않음; mu=0)
        Y_D, mu_D, stdinv_D, Ex_acc_D, Ex2_acc_D, ex_out_D, ex2_out_D, debug_info_D = AILayerNorm(X, alpha, zp, gamma_batch, beta_batch, inv_n, variant="D")
        mae_D = np.mean(np.abs(orig_Y - Y_D))
        accuracy_D = (1 - (mae_D / 255)) * 100
        mae_list_D.append(mae_D)
        # debug_info_D에는 RMSNorm 방식에서 RMS 값을 포함 (예: debug_info_D["rms"])
        rms_list_D.append(debug_info_D.get("rms", 0))
        results_D[f"Batch {i+1}"] = {
            "Input": X,
            "Stage1_mu": mu_D,
            "Stage1_std (1/stdinv)": (1/stdinv_D if stdinv_D != 0 else 0),
            "Final Y (Variant D)": Y_D,
            "MAE": mae_D,
            "Accuracy (%)": accuracy_D,
            "RMS": debug_info_D.get("rms", None)
        }

    overall_mae_A = np.mean(mae_list_A)
    overall_accuracy_A = (1 - (overall_mae_A / 255)) * 100
    overall_mae_B = np.mean(mae_list_B)
    overall_accuracy_B = (1 - (overall_mae_B / 255)) * 100
    overall_mae_C = np.mean(mae_list_C)
    overall_accuracy_C = (1 - (overall_mae_C / 255)) * 100
    overall_mae_D = np.mean(mae_list_D)
    overall_accuracy_D = (1 - (overall_mae_D / 255)) * 100

    # 한 화면에 3개 서브플롯 (분산, MAE, Accuracy) + RMS (Variant D)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 서브플롯 1: Batch별 분산/ RMS 값 비교 (Variant A, B, C는 분산, D는 RMS)
    axs[0].plot(range(1, batches+1), var_list_A, marker='o', label="Variant A (negative value→0)")
    axs[0].plot(range(1, batches+1), var_list_B, marker='x', label="Variant B (previous variance)")
    axs[0].plot(range(1, batches+1), var_list_C, marker='s', label="Variant C (negative value→1)")
   # axs[0].plot(range(1, batches+1), rms_list_D, marker='^', label="Variant D (RMSNorm)")
    axs[0].set_title("Compare Variance Values by Batch")
    axs[0].set_xlabel("Batch number")
    axs[0].set_ylabel("값")
    axs[0].legend()
    axs[0].grid(True)
    
    # 서브플롯 2: Batch별 MAE 비교
    axs[1].plot(range(1, batches+1), mae_list_A, marker='o', label="MAE Variant A")
    axs[1].plot(range(1, batches+1), mae_list_B, marker='x', label="MAE Variant B")
    axs[1].plot(range(1, batches+1), mae_list_C, marker='s', label="MAE Variant C")
    axs[1].plot(range(1, batches+1), mae_list_D, marker='^', label="MAE Variant D")
    axs[1].set_title("Comparison of MAE by Batch")
    axs[1].set_xlabel("Batch number")
    axs[1].set_ylabel("MAE")
    axs[1].legend()
    axs[1].grid(True)
    
    # 서브플롯 3: Batch별 Accuracy 비교
    axs[2].plot(range(1, batches+1), [(1 - (mae / 255)) * 100 for mae in mae_list_A], marker='o', label="Accuracy Variant A")
    axs[2].plot(range(1, batches+1), [(1 - (mae / 255)) * 100 for mae in mae_list_B], marker='x', label="Accuracy Variant B")
    axs[2].plot(range(1, batches+1), [(1 - (mae / 255)) * 100 for mae in mae_list_C], marker='s', label="Accuracy Variant C")
    axs[2].plot(range(1, batches+1), [(1 - (mae / 255)) * 100 for mae in mae_list_D], marker='^', label="Accuracy Variant D")
    axs[2].set_title("Comparison of Accuracy by Batch")
    axs[2].set_xlabel("Batch number")
    axs[2].set_ylabel("Accuracy (%)")
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

    # 전체 결과 요약 출력 (각 variant별)
    print("\n==== Summary of All Batches (Variant A) ====")
    for name in results_A.keys():
        print(f"\n{name}")
        print("Input X:", results_A[name]["Input"])
        print("Stage1: mu =", results_A[name]["Stage1_mu"], ", std =", results_A[name]["Stage1_std (1/stdinv)"])
        print("Final Y (Variant A):", results_A[name]["Final Y (Variant A)"])
        print("Original LayerNorm Output:", orig_results[name]["Original LayerNorm Output"])
        print("MAE:", results_A[name]["MAE"])
        print("Accuracy (%):", results_A[name]["Accuracy (%)"])
        print("Variance:", results_A[name]["Variance"])
    
    print("\n==== Summary of All Batches (Variant B) ====")
    for name in results_B.keys():
        print(f"\n{name}")
        print("Input X:", results_B[name]["Input"])
        print("Stage1: mu =", results_B[name]["Stage1_mu"], ", std =", results_B[name]["Stage1_std (1/stdinv)"])
        print("Final Y (Variant B):", results_B[name]["Final Y (Variant B)"])
        print("Original LayerNorm Output:", orig_results[name]["Original LayerNorm Output"])
        print("MAE:", results_B[name]["MAE"])
        print("Accuracy (%):", results_B[name]["Accuracy (%)"])
        print("Variance:", results_B[name]["Variance"])
    
    print("\n==== Summary of All Batches (Variant C) ====")
    for name in results_C.keys():
        print(f"\n{name}")
        print("Input X:", results_C[name]["Input"])
        print("Stage1: mu =", results_C[name]["Stage1_mu"], ", std =", results_C[name]["Stage1_std (1/stdinv)"])
        print("Final Y (Variant C):", results_C[name]["Final Y (Variant C)"])
        print("Original LayerNorm Output:", orig_results[name]["Original LayerNorm Output"])
        print("MAE:", results_C[name]["MAE"])
        print("Accuracy (%):", results_C[name]["Accuracy (%)"])
        print("Variance:", results_C[name]["Variance"])
    
    print("\n==== Summary of All Batches (Variant D - RMSNorm) ====")
    for name in results_D.keys():
        print(f"\n{name}")
        print("Input X:", results_D[name]["Input"])
        print("Stage1: mu =", results_D[name]["Stage1_mu"], ", std =", results_D[name]["Stage1_std (1/stdinv)"])
        print("Final Y (Variant D):", results_D[name]["Final Y (Variant D)"])
        print("Original LayerNorm Output:", orig_results[name]["Original LayerNorm Output"])
        print("MAE:", results_D[name]["MAE"])
        print("Accuracy (%):", results_D[name]["Accuracy (%)"])
        print("RMS:", results_D[name]["RMS"] if "RMS" in results_D[name] else "N/A")
    
    print("\nOverall Variant A MAE:", overall_mae_A)
    print("Overall Variant A Accuracy (%):", overall_accuracy_A)
    print("Overall Variant B MAE:", overall_mae_B)
    print("Overall Variant B Accuracy (%):", overall_accuracy_B)
    print("Overall Variant C MAE:", overall_mae_C)
    print("Overall Variant C Accuracy (%):", overall_accuracy_C)
    print("Overall Variant D MAE:", overall_mae_D)
    print("Overall Variant D Accuracy (%):", overall_accuracy_D)
'''


#########################################################################################################################

############################################ AILayerNorm SW Test (Total Input)###########################################

#########################################################################################################################
'''
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib

# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()  # 입력값 추출
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # [channels] (예: [768])
    beta  = final_norm.bias.detach()      # [channels]
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 하드웨어/소프트웨어용 함수들 (AILayerNorm SW 코드)
# =============================================================================
def lower8(x):
    """
    주어진 실수 x를 정수로 변환한 후,
    하위 8비트를 그대로 추출하여 부호 있는 int8 값으로 해석합니다.
    (예: 190 → (190 & 0xFF)=190 → 190>=128이면 190-256 = -66)
    """
    x_int = int(round(x))
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def lower8_hw(x):
    """
    하드웨어 방식과 동일하게 정수 x의 하위 8비트를 부호 있는 int8으로 변환합니다.
    """
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

def dynamic_compress_D(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

def lut_std(temp_var):
    """
    LUT를 통한 stdinv 근사 (Q0.8 값):
      temp_var: 분산의 하위 16비트 값 (0~65535)
      원래 LUT 매핑 값은:
         1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 255
      여기서는 그대로 반환합니다.
    """
    if temp_var >= 32768:
        return 1
    elif temp_var >= 16384:
        return 2
    elif temp_var >= 8192:
        return 3
    elif temp_var >= 4096:
        return 4
    elif temp_var >= 2048:
        return 6
    elif temp_var >= 1024:
        return 8
    elif temp_var >= 512:
        return 11
    elif temp_var >= 256:
        return 16
    elif temp_var >= 128:
        return 23
    elif temp_var >= 64:
        return 32
    elif temp_var >= 32:
        return 45
    elif temp_var >= 16:
        return 64
    elif temp_var >= 8:
        return 90
    elif temp_var >= 4:
        return 128
    elif temp_var >= 2:
        return 181
    elif temp_var >= 1:
        return 255
    else:
        return 0

# =============================================================================
# Variant A: 분산이 음수일 경우 0으로 매핑 (기존 방식)
# =============================================================================
def stage1_detailed_variantA(X, alpha, zp, inv_n):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
    #    variance = 0
        stdinv_Q = 0
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
   # stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant B: 분산이 음수일 경우 이전 분산 값을 유지
# =============================================================================
prev_variance = None
def stage1_detailed_variantB(X, alpha, zp, inv_n):
    global prev_variance
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    if variance < 0:
        if prev_variance is not None:
            variance = prev_variance
        else:
            variance = 0
    else:
        prev_variance = variance
    temp_var = int(variance) & 0xFFFF
    stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant C: 분산이 음수일 경우 1로 매핑
# =============================================================================
def stage1_detailed_variantC(X, alpha, zp, inv_n, small_const=1):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
      #  variance = small_const
        stdinv_Q = 1
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
  #  stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant D: Dynamic Compression + RMSNorm (RMSNorm: 평균(mu)=0, beta 미사용)
# =============================================================================
def stage1_detailed_variantD(X, alpha, zp, inv_n, eps=1):
    C = len(X)
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x)  # 평균 제거하지 않음
        Xc, s = dynamic_compress_D(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex2_avg = (Ex2_acc << 4) / C
    rms = math.sqrt(Ex2_avg + eps)
    stdinv = 1 / rms
    mu_val = 0  # RMSNorm에서는 평균 사용하지 않음

    debug_info = {
        "Ex2_avg": Ex2_avg,
        "rms": rms,
        "stdinv": stdinv
    }
    ex_unit_out = None
    ex2_unit_out = None
    return mu_val, stdinv, 0, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# RMS용 Dynamic Compression 함수 (Variant D에서 사용)
# =============================================================================
def dynamic_compress_D(x):
    """
    Dynamic Compress for RMSNorm variant:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 2
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 1
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

# =============================================================================
# Stage2 및 전체 AILayerNorm 함수 (variant에 따라 호출)
# =============================================================================
def stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=True):
    """
    Stage2:
      각 채널에 대해:
         1. X_norm_shift = X[i] << alpha (zero point 제거 없음)
         2. X_diff = X_norm_shift - μ
         3. temp_Norm = gamma[i] * stdinv * X_diff
         4. 최종 출력:
              - 일반 variant: o_Norm = lower8_hw(temp_Norm) + beta[i]
              - RMSNorm variant (D): o_Norm = lower8_hw(temp_Norm)  (beta 미사용)
    """
    Y = []
    for i, x in enumerate(X):
        Xi = int(x)
        X_norm_shift = Xi << alpha
        X_diff = X_norm_shift - mu
        temp_Norm = gamma[i] * stdinv * X_diff
        if add_beta:
            final = lower8_hw(temp_Norm) + beta[i]
        else:
            final = lower8_hw(temp_Norm)
        Y.append(final)
    return np.array(Y, dtype=np.int8)

def AILayerNorm(X, alpha, zp, gamma, beta, inv_n, variant="A", small_const=1):
    if variant == "A":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantA(X, alpha, zp, inv_n)
        add_beta = True
    elif variant == "B":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantB(X, alpha, zp, inv_n)
        add_beta = True
    elif variant == "C":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantC(X, alpha, zp, inv_n, small_const)
        add_beta = True
    elif variant == "D":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantD(X, alpha, zp, inv_n)
        add_beta = False  # RMSNorm: beta 미사용
    else:
        raise ValueError("알 수 없는 variant입니다. 'A', 'B', 'C', 'D' 중 하나를 선택하세요.")
    Y = stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=add_beta)
    return Y, mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

def original_layernorm(X, gamma, beta, eps=1e-5):
    X = np.array(X, dtype=float)
    mean = np.mean(X)
    var = np.var(X, ddof=0)
    normalized = (X - mean) / math.sqrt(var + eps)
    out = normalized * gamma + beta
    quantized = np.array([lower8_hw(val) for val in out], dtype=np.int8)
    return quantized, mean, var

def display_full_input(X_full):
    width = len(X_full) * 8
    hex_str = "_".join(f"{x:02X}" for x in X_full)
    dec_str = "_".join(str(x) for x in X_full)
    print(f"Input X = {width}'h{hex_str};")
    print(f"// {width}'d{dec_str}")

def quantize_fp32_to_int8(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale

def tensor_to_verilog_hex(tensor):
    hex_strs = []
    for val in tensor:
        ival = int(val)
        if ival < 0:
            ival = ival & 0xFF
        hex_strs.append("{:02X}".format(ival))
    return "_".join(hex_strs)

# =============================================================================
# Swin 모델에서 LayerNorm 입력 및 파라미터 추출
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()
    beta  = final_norm.bias.detach()
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 메인 실행부 (전체 입력 사용, 배치별 출력 없이 전체 overall 값만 계산)
# =============================================================================
if __name__ == '__main__':
    # 이미지 로드 및 전처리 (timm transform 사용)
    image = Image.open(r"C:\SYDLAB_SY\test_image_grand.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 원래 LayerNorm 입력 및 파라미터 추출
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("원래 Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("원래 Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화하여 전체 입력 사용 (전체 값 사용)
    flattened_input = layernorm_input.flatten()  # 예: 7*7*768 = 37632
    total = flattened_input.numel()
    selected_input = flattened_input  # 전체 입력 사용

    # 전체 입력에 대해 각 요소의 채널 번호 계산 (마지막 차원 크기 768)
    indices = torch.arange(0, total)
    channels = indices % gamma_tensor.numel()
    selected_gamma = gamma_tensor[channels]
    selected_beta = beta_tensor[channels]

    # fp32 -> int8 양자화
    selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
    selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
    selected_beta_int8, scale_beta = quantize_fp32_to_int8(selected_beta)

    # Verilog 리터럴 문자열 생성 (전체 입력에 대해)
    i_x_hex     = tensor_to_verilog_hex(selected_input_int8)
    i_gamma_hex = tensor_to_verilog_hex(selected_gamma_int8)
    i_beta_hex  = tensor_to_verilog_hex(selected_beta_int8)

    print("\ni_x = {}'h{}".format(total*8, i_x_hex) + ";")
    print("i_gamma = {}'h{}".format(total*8, i_gamma_hex) + ";")
    print("i_beta = {}'h{}".format(total*8, i_beta_hex) + ";")

    print("\n선택된 LayerNorm 입력 ({} int8 값, {} bits):\n".format(total, total*8), selected_input_int8)
    print("입력 scale:", scale_input)
    print("\n선택된 Gamma ({} int8 값):\n".format(total), selected_gamma_int8)
    print("Gamma scale:", scale_gamma)
    print("\n선택된 Beta ({} int8 값):\n".format(total), selected_beta_int8)
    print("Beta scale:", scale_beta)

    # AILayerNorm 파라미터 설정 (입력이 이미 -128~127이므로 zp = 0)
    zp = 0
    alpha = 2
    inv_n = 32

    # 전체 입력 및 파라미터를 numpy 배열로 변환
    X_full = selected_input_int8.cpu().numpy()  # shape: (total,)
    gamma_full = selected_gamma_int8.cpu().numpy()  # shape: (total,)
    beta_full = selected_beta_int8.cpu().numpy()      # shape: (total,)

    # 배치 사이즈를 8로 설정
    batches = total // 8

    # 각 variant별 전체 결과(전체 배치에 대해)를 저장할 리스트
    all_orig = []
    all_A = []
    all_B = []
    all_C = []
    all_D = []
    for i in range(batches):
        X_batch = X_full[i*8:(i+1)*8]
        gamma_batch = gamma_full[i*8:(i+1)*8]
        beta_batch = beta_full[i*8:(i+1)*8]
        # 원래 LayerNorm
        orig_Y, _, _ = original_layernorm(X_batch, gamma_batch, beta_batch)
        all_orig.append(orig_Y)
        # Variant A
        Y_A, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="A")
        all_A.append(Y_A)
        # Variant B
        Y_B, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="B")
        all_B.append(Y_B)
        # Variant C
        Y_C, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="C", small_const=1)
        all_C.append(Y_C)
        # Variant D (RMSNorm: beta 미사용, mu=0)
        Y_D, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="D")
        all_D.append(Y_D)

    # 전체 결과 배열로 결합
    all_orig = np.concatenate(all_orig)
    all_A = np.concatenate(all_A)
    all_B = np.concatenate(all_B)
    all_C = np.concatenate(all_C)
    all_D = np.concatenate(all_D)

    # 전체 MAE 및 Accuracy 계산 (각 variant)
    mae_A = np.mean(np.abs(all_orig - all_A))
    acc_A = (1 - mae_A / 255) * 100

    mae_B = np.mean(np.abs(all_orig - all_B))
    acc_B = (1 - mae_B / 255) * 100

    mae_C = np.mean(np.abs(all_orig - all_C))
    acc_C = (1 - mae_C / 255) * 100

    mae_D = np.mean(np.abs(all_orig - all_D))
    acc_D = (1 - mae_D / 255) * 100

    print("\nOverall Results:")
   # print("Original LayerNorm Output (first 20 values):", all_orig[:20])
    print("Variant A ->             MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_A, acc_A))
    print("Variant B ->             MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_B, acc_B))
    print("Variant C ->             MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_C, acc_C))
    print("AILayerNorm + RMSNorm -> MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_D, acc_D))

    # 한 화면에 1개의 서브플롯으로 전체 MAE와 Accuracy를 막대 그래프로 그리기
    variants = ['Variance = 0', 'Previous Variance', 'std = 1', 'AILayerNorm + RMSNorm']
    mae_values = [mae_A, mae_B, mae_C, mae_D]
    acc_values = [acc_A, acc_B, acc_C, acc_D]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(variants, mae_values, width=0.5, color=['blue', 'orange', 'green', 'red'])
    axs[0].set_title("Overall MAE Comparison")
    axs[0].set_ylabel("MAE")
    axs[1].bar(variants, acc_values, width=0.5, color=['blue', 'orange', 'green', 'red'])
    axs[1].set_title("Overall Accuracy Comparison")
    axs[1].set_ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.show()
'''






'''
#########################################################################################################################

################################################## AILayerNorm HW Test ##################################################

#########################################################################################################################
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib

# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()  # 입력값 추출
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # [channels] (예: [768])
    beta  = final_norm.bias.detach()      # [channels]
    return hook_data["layernorm_input"], gamma, beta


def lower8(x):
    x_int = int(round(x))
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r


def lower8_hw(x):
    """
    하드웨어 방식과 동일하게 정수 x의 하위 8비트를 부호 있는 int8으로 변환합니다.
    """
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r


def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립 (여기서는 음수면 0)
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

def lut_std(temp_var):
    """
    LUT를 통한 stdinv 근사 (Q0.8 값):
      temp_var: 분산의 하위 16비트 값 (0~65535)
      원래 LUT 매핑 값은:
         1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 255
      여기서는 그대로 반환합니다.
    """
    if temp_var >= 32768:
        return 1
    elif temp_var >= 16384:
        return 2
    elif temp_var >= 8192:
        return 3
    elif temp_var >= 4096:
        return 4
    elif temp_var >= 2048:
        return 6
    elif temp_var >= 1024:
        return 8
    elif temp_var >= 512:
        return 11
    elif temp_var >= 256:
        return 16
    elif temp_var >= 128:
        return 23
    elif temp_var >= 64:
        return 32
    elif temp_var >= 32:
        return 45
    elif temp_var >= 16:
        return 64
    elif temp_var >= 8:
        return 90
    elif temp_var >= 4:
        return 128
    elif temp_var >= 2:
        return 181
    elif temp_var >= 1:
        return 255
    else:
        return 1

def stage1_detailed(X, alpha, zp, inv_n):
    """
    Stage1:
      X: 8채널 입력 벡터 (각 요소 0~255, int)
      alpha: PTF scaling shift 값
      zp: zero point (보통 128)
      inv_n: 1/C (8-bit, Q0.8); 예: 1/8 = 0.125, Q0.8에서는 32
      
      각 채널에 대해:
         1. X_i = X[i] - zp          (부호 있는 정수)
         2. (X_c, s) = DynamicCompress(|X_i|)
         3. Xc_decomp = (X_c^2) << (4 * s)
            (미리 정의한 LUT: 0→0, 1→1, 2→4, …, 15→225)
         4. 누산:
              Ex_acc += X_i << alpha  
              Ex2_acc += Xc_decomp << (2 * alpha)
      최종적으로,
         Ex_avg = Ex_acc / 8,  
         Ex2_avg = (Ex2_acc << 4) / 8,
         μ = Ex_avg,  
         σ² = Ex2_avg - μ².
         temp_var = 하위 16비트(variance) (정수 변환 후 비트 마스킹)
         stdinv_Q = LUT(temp_var) (Q0.8 정수)
         stdinv = stdinv_Q / 256.0  (실수 값)
      
      하드웨어 시뮬레이션 결과:
         ex_unit_out = (Ex_acc * inv_n) >> 8  
         ex2_unit_out = (Ex2_acc * inv_n) >> 8
         
      디버깅 정보를 dictionary로 수집하여 반환.
    """
    C = 8  # 채널 수 고정
    Ex_acc = 0
    Ex2_acc = 0
    # 4비트 LUT for square approximation
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp  # 부호 있는 정수
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += (Xc_decomp << (2 * alpha))
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C
    
    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    # print ("Ex :", ex_unit_out)
    # print ("Ex2 :", ex2_unit_out)

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
      #  variance = small_const
        stdinv_Q = 1
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
  #  stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0



    mu_int8 = Ex_avg
    
    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    
    return mu_int8, stdinv, stdinv_Q, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

def stage2(X, alpha, mu, stdinv, gamma, beta, zp):
    Y = []
    for i, x in enumerate(X):
        Xi = int(x) - zp
        X_norm_shift = Xi << alpha
        X_diff = X_norm_shift - mu
        ############ leanable gamma, beta ############
        Y_val = gamma[i] * stdinv * X_diff + beta[i]

       ############ gamma, beta 고정 ############
        #Y_val = gamma * stdinv * X_diff + beta
        Y.append(lower8_hw(Y_val))
    return np.array(Y, dtype=np.int8)


def AILayerNorm(X, alpha, zp, gamma, beta, inv_n):
    """
    전체 AILayerNorm:
       - Stage1: μ, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info 계산 (8채널 배치)
       - Stage2: 최종 normalized 출력 Y 계산 (하위 8비트 클리핑)
    """
    mu, stdinv, stdinv_Q, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed(X, alpha, zp, inv_n)
    Y = stage2(X, alpha, mu, stdinv, gamma, beta, zp)
    return Y, mu, stdinv, stdinv_Q, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info



def display_full_input(X_full):
    """
    전체 입력 X_full (배열)을 4096비트 (512개의 8비트 값)로 표현하여 
    16진수와 10진수 형식으로 출력하는 함수.
    각 8비트 값은 '_'로 구분.
    """
    width = len(X_full)*8  # 512 * 8 = 4096비트
    hex_str = "_".join(f"{x:02X}" for x in X_full)
    dec_str = "_".join(str(x) for x in X_full)
    print(f"Input X = {width}'h{hex_str};")
    print(f"// {width}'d{dec_str}")

def quantize_fp32_to_int8(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale

def tensor_to_verilog_hex(tensor):
    hex_strs = []
    for val in tensor:
        ival = int(val)
        if ival < 0:
            ival = ival & 0xFF
        hex_strs.append("{:02X}".format(ival))
    return "_".join(hex_strs)


# =============================================================================
# 메인 실행부
# =============================================================================
if __name__ == '__main__':
    # 이미지 로드 및 전처리 (timm transform 사용)
    image = Image.open(r"C:\SYDLAB_SY\test_image_road.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 원래 LayerNorm 입력 및 파라미터 추출
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("원래 Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("원래 Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화 후 중앙 512개 값 선택 (4096비트)
    flattened_input = layernorm_input.flatten()
    total = flattened_input.numel()
    start_index = (total) // 2
    #start_index = 
    selected_input = flattened_input[start_index : start_index + 512]

    # 각 선택된 값의 채널 번호 계산 (마지막 차원 크기가 768)
    indices = torch.arange(start_index, start_index + 512)
    channels = indices % gamma_tensor.numel()
    selected_gamma = gamma_tensor[channels]
    selected_beta = beta_tensor[channels]

    # fp32 -> int8 양자화
    selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
    selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
    selected_beta_int8, scale_beta = quantize_fp32_to_int8(selected_beta)

    i_x_hex     = tensor_to_verilog_hex(selected_input_int8)
    i_gamma_hex = tensor_to_verilog_hex(selected_gamma_int8)
    i_beta_hex  = tensor_to_verilog_hex(selected_beta_int8)

    print("\ni_x = 4096'h" + i_x_hex + ";")
    print("i_gamma = 4096'h" + i_gamma_hex + ";")
    print("i_beta = 4096'h" + i_beta_hex + ";")

    print("\n선택된 LayerNorm 입력 (512 int8 값, 4096비트):\n", selected_input_int8)
    print("입력 scale:", scale_input)
    print("\n선택된 Gamma (512 int8 값):\n", selected_gamma_int8)
    print("Gamma scale:", scale_gamma)
    print("\n선택된 Beta (512 int8 값):\n", selected_beta_int8)
    print("Beta scale:", scale_beta)

    # AILayerNorm 파라미터 설정 (입력이 이미 -128~127이므로 zp = 0)
    zp = 0
    alpha = 2
    inv_n = 32
    #beta = 10
    #gamma = 100

    # 선택된 값들을 numpy 배열로 변환 (512개의 int8 값)
    X_full = selected_input_int8.cpu().numpy()
    gamma_full = selected_gamma_int8.cpu().numpy()
    beta_full = selected_beta_int8.cpu().numpy()
    
    results = {}
    batches = 512 // 8  # 64 배치
    for i in range(batches):
        X_batch = X_full[i*8:(i+1)*8]
        gamma_batch = gamma_full[i*8:(i+1)*8]
        beta_batch = beta_full[i*8:(i+1)*8]

        ############ leanable gamma, beta ############
        Y, mu, stdinv, stdinv_Q, Ex_acc, Ex2_acc, ex_out, ex2_out, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n)

        ############ gamma, beta 고정 ############
        #Y, mu, stdinv, stdinv_Q, temp_var, Ex_acc, Ex2_acc, ex_out, ex2_out, _ = AILayerNorm(X_batch, alpha, zp, gamma, beta, inv_n)
        # 표준편차 std = 1/stdinv (stdinv가 0이 아니면)
       # std = 1 / stdinv if stdinv != 0 else 0
        results[f"Batch {i+1}"] = {
            "Input": X_batch,
            "beta" : beta_batch,
            "gamma" : gamma_batch,
            "Stage1_mu": mu,
            "Stage1_stdinv_Q": stdinv_Q,
            "Final Y": Y
        }
    
    print("\n==== Summary of All Batches ====")
    for name, info in results.items():
        print(f"\n{name}")
        print("Input X:", info["Input"])
        print("beta   :", info["beta"])
        print("gamma  :", info["gamma"])
        print("Stage1: mu =", info["Stage1_mu"], "stdinv_Q =", info["Stage1_stdinv_Q"])
        print("Final normalized output Y (int8):", info["Final Y"])
'''




########################################################################################################################

########################################### AILayerNorm + RMSNorm vs. RMSNorm ##########################################

########################################################################################################################

'''
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib


# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # 예: [768]
    beta  = final_norm.bias.detach()      # 예: [768]
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 하드웨어/소프트웨어용 함수들
# =============================================================================
def lower8_hw(x):
    """정수 x의 하위 8비트를 부호 있는 int8으로 변환"""
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r


def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립 (여기서는 음수면 0)
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s



def dynamic_compress_D(x):
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    return np.clip(Xc, 0, 15), s

# =============================================================================
# Original RMSNorm (FP32 기반) – bias 없이
# =============================================================================
def original_rmsnorm(X, gamma, eps=1e-8):
    """
    FP32 RMSNorm: RMS = sqrt(mean(X^2) + eps), 출력 = gamma * (X / RMS)
    """
    X = np.array(X, dtype=float)
    rms = math.sqrt(np.mean(X**2) + eps)
    out = gamma * (X / rms)
    return out, rms

# =============================================================================
# Variant A: 분산이 음수일 경우 0으로 매핑 (기존 방식)
# =============================================================================
def stage1_detailed_variantA(X, alpha, zp, inv_n):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
    #    variance = 0
        stdinv_Q = 0
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
   # stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info



# =============================================================================
# Variant D: AILayerNorm + RMSNorm (Dynamic Compression + RMSNorm) – int8 연산
# =============================================================================
def stage1_detailed_variantD(X, alpha, zp, inv_n, eps=1):
    """
    RMSNorm Variant D (Dynamic Compression 적용):
      - X: int32 배열 (입력은 이미 -128~127 범위)
      - dynamic_compress_D()를 사용해 x의 제곱 근사 → RMS = sqrt(mean(approx(x^2)) + eps)
      - RMSNorm은 평균 제거 없이 정규화 (mu=0)
    """
    C = len(X)
    Ex2_acc = 0
    # 0~15에 대한 제곱 LUT
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x)  # 평균 제거 없음
        Xc, s = dynamic_compress_D(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        # 동적 압축 시 s에 따른 scaling 복원
        Xc_decomp = Xc_square << (4 * s)
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex2_avg = (Ex2_acc << 4) / C
    rms = math.sqrt(Ex2_avg + eps)
    stdinv = 1 / rms
    mu_val = 0  # RMSNorm은 mu=0
    debug_info = {"Ex2_avg": Ex2_avg, "rms": rms, "stdinv": stdinv}
    return mu_val, stdinv, 0, Ex2_acc, None, None, debug_info

def stage2_variantD(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=False):
    """
    RMSNorm Variant D:
      - 각 요소에 대해, x_norm = (x - zp) << alpha (mu=0)
      - 출력 = lower8_hw( gamma[i] * stdinv * x_norm )
      - beta는 사용하지 않음.
    """
    Y = []
    for i, x in enumerate(X):
        Xi = int(x) - zp
        X_norm_shift = Xi << alpha
        temp_Norm = gamma[i] * stdinv * X_norm_shift
        final = lower8_hw(temp_Norm)
        Y.append(final)
    return np.array(Y, dtype=np.int8)

def AILayerNorm_variantD(X, alpha, zp, gamma, beta, inv_n, eps=1):
    mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantD(X, alpha, zp, inv_n, eps)
    Y = stage2_variantD(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=False)
    return Y, mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# FP32 -> int8 (대칭 양자화) 함수
# =============================================================================
def quantize_fp32_to_int8_symmetric(x):
    """
    FP32 -> int8 (대칭 양자화): 
    - 최대 절대값을 기준으로 scale = max_abs / 127로 설정하고,
    - x를 scale로 나눈 후, -127 ~ 127 범위로 클램핑하여 int8로 변환.
    """
    max_abs = torch.max(torch.abs(x))
    scale = max_abs / 127 if max_abs > 0 else 1.0
    x_quant = torch.round(x / scale)
    x_quant = torch.clamp(x_quant, -127, 127)
    return x_quant.to(torch.int8), scale

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    # 이미지 로드 및 전처리
    image = Image.open(r"C:\SYDLAB_SY\test_image_road.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # LayerNorm 입력 및 파라미터 추출
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("Original LayerNorm input shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("Original Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("Original Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화: 전체 입력 사용 (전체 요소: 7*7*768 = 37632)
    flattened_input = layernorm_input.flatten()

    # --- Original RMSNorm 계산 (FP32) 및 대칭 양자화 int8 변환 ---
    total_indices = torch.arange(flattened_input.numel())
    channels = total_indices % gamma_tensor.numel()
    # 채널별 감마 확장 (FP32)
    selected_gamma_fp32 = gamma_tensor[channels].cpu().numpy().astype(np.float32)
    # beta는 RMSNorm에서 사용하지 않으므로 0으로 처리
    selected_beta_fp32 = np.zeros_like(selected_gamma_fp32)
    orig_rmsnorm_out_fp32, rms_val = original_rmsnorm(flattened_input.cpu().numpy(), selected_gamma_fp32)
    orig_rmsnorm_out_tensor = torch.tensor(orig_rmsnorm_out_fp32, dtype=torch.float32)
    orig_rmsnorm_out_int8, scale_orig = quantize_fp32_to_int8_symmetric(orig_rmsnorm_out_tensor)

    # --- AILayerNorm + RMSNorm (Variant D) 계산 (입력, 감마, 베타 모두 대칭 양자화 적용) ---
    flattened_input_tensor = torch.tensor(flattened_input.cpu().numpy(), dtype=torch.float32)
    selected_input_int8, scale_input = quantize_fp32_to_int8_symmetric(flattened_input_tensor)
    # Variant D는 int32 배열을 기대하므로 변환
    X_input = selected_input_int8.cpu().numpy().astype(np.int32)
    # 감마, 베타 대칭 양자화
    gamma_int8, scale_gamma = quantize_fp32_to_int8_symmetric(gamma_tensor.to(torch.float32))
    beta_int8, scale_beta = quantize_fp32_to_int8_symmetric(beta_tensor.to(torch.float32))
    selected_gamma_int8 = gamma_int8[channels].cpu().numpy().astype(np.int8)
    selected_beta_int8 = np.zeros_like(selected_gamma_int8)
    
    variantD_out, mu_D, stdinv_D, _, _, _, _, debug_info_D = AILayerNorm_variantD(
        selected_input_int8, alpha=2, zp=0, gamma=selected_gamma_int8, beta=selected_beta_int8, inv_n=32, eps=1
    )
    # variantD_out은 int8 결과 (전체)

    # --- Error Metrics 비교 ---
    mae = np.mean(np.abs(orig_rmsnorm_out_int8.cpu().numpy().astype(np.float32) - variantD_out.astype(np.float32)))
    mse = np.mean((orig_rmsnorm_out_int8.cpu().numpy().astype(np.float32) - variantD_out.astype(np.float32))**2)
    accuracy = (1 - (mae / 255)) * 100

    # 중간 100개 요소 추출 (전체 배열 길이의 중앙 부분)
    orig_array = orig_rmsnorm_out_int8.cpu().numpy()
    variant_array = variantD_out
    mid_start_orig = (len(orig_array) - 100) // 2
    mid_orig = orig_array[mid_start_orig: mid_start_orig + 100]
    mid_start_variant = (len(variant_array) - 100) // 2
    mid_variant = variant_array[mid_start_variant: mid_start_variant + 100]

    print("\n--- RMSNorm Comparison (Symmetric Quantization) ---")
    print("Original RMSNorm output (FP32 -> int8, symmetric) (middle 100 values):")
    print(mid_orig)
    print("\nAILayerNorm + RMSNorm (Variant D) output (int8, symmetric) (middle 100 values):")
    print(mid_variant)
    print("\nMean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Accuracy (%):", accuracy)

    # --- 그래프 비교 (일부 인덱스) ---
    plt.figure(figsize=(12, 4))
    plt.plot(orig_rmsnorm_out_int8.cpu().numpy(), label="Original RMSNorm (FP32 -> int8, symmetric)", marker='o', linestyle='--')
    plt.plot(variantD_out, label="AILayerNorm + RMSNorm (Variant D) (int8, symmetric)", marker='x', linestyle=':')
    plt.title("Comparison of RMSNorm Outputs (Symmetric Quantization)")
    plt.xlabel("Index")
    plt.ylabel("Output Value (int8)")
    plt.legend()
    plt.grid(True)
    plt.show()
    


'''
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib


# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
#model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)     # Swin-T
#model = timm.create_model('swin_small_patch4_window7_224.ms_in1k', pretrained=True)     # Swin-S
#model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=True)      # Swin-B

#model = timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=True)      # DeiT-T
#model = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True)     # DeiT-S
#model = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True)      # DeiT-B


model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()  # 입력값 추출
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # [channels] (예: [768])
    beta  = final_norm.bias.detach()      # [channels]
    return hook_data["layernorm_input"], gamma, beta



# =============================================================================
# 하드웨어/소프트웨어용 함수들
# =============================================================================
def lower8_hw(x):
    """정수 x의 하위 8비트를 부호 있는 int8으로 변환"""
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r



def quantize_fp32_to_int8(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale

def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립 (여기서는 음수면 0)
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s




def dynamic_compress_D(x):
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    # 여기서는 원래 방식과 다르게 s=1, Xc = (x+8)//16로 처리 (원래 코드와 맞추기 위해 수정)
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    return np.clip(Xc, 0, 15), s

# =============================================================================
# 16-entry LUT를 통한 stdinv 근사 함수 (우선순위 인코더 방식)
# =============================================================================

def lut_std(temp_var):
    """
    LUT를 통한 stdinv 근사 (Q0.8 값):
      temp_var: 분산의 하위 16비트 값 (0~65535)
      원래 LUT 매핑 값은:
         1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 255
      여기서는 그대로 반환합니다.
    """
    if temp_var >= 32768:
        return 1
    elif temp_var >= 16384:
        return 2
    elif temp_var >= 8192:
        return 3
    elif temp_var >= 4096:
        return 4
    elif temp_var >= 2048:
        return 6
    elif temp_var >= 1024:
        return 8
    elif temp_var >= 512:
        return 11
    elif temp_var >= 256:
        return 16
    elif temp_var >= 128:
        return 23
    elif temp_var >= 64:
        return 32
    elif temp_var >= 32:
        return 45
    elif temp_var >= 16:
        return 64
    elif temp_var >= 8:
        return 90
    elif temp_var >= 4:
        return 128
    elif temp_var >= 2:
        return 181
    elif temp_var >= 1:
        return 255
    else:
        return 1
    
   
# def lut_std(temp_var):
#     """
#     24-entry LUT for 1/sqrt(temp_var) in Q0.16 format.
#     Covers input range: 1 ~ 2^24
#     Output: integer in range 1 ~ 65535 (Q0.16 fixed-point)
#     Priority encoder 방식 (Verilog 패턴과 일치)
#     """
#     if temp_var >= 8388608:
#         return 23      # x = 8388608, ≈ 0.000345
#     elif temp_var >= 4194304:
#         return 32      # x = 4194304
#     elif temp_var >= 2097152:
#         return 45      # x = 2097152
#     elif temp_var >= 1048576:
#         return 64      # x = 1048576
#     elif temp_var >= 524288:
#         return 91      # x = 524288
#     elif temp_var >= 262144:
#         return 128     # x = 262144
#     elif temp_var >= 131072:
#         return 181     # x = 131072
#     elif temp_var >= 65536:
#         return 256     # x = 65536
#     elif temp_var >= 32768:
#         return 362     # x = 32768
#     elif temp_var >= 16384:
#         return 512     # x = 16384
#     elif temp_var >= 8192:
#         return 724     # x = 8192
#     elif temp_var >= 4096:
#         return 1024    # x = 4096
#     elif temp_var >= 2048:
#         return 1448    # x = 2048
#     elif temp_var >= 1024:
#         return 2048    # x = 1024
#     elif temp_var >= 512:
#         return 2896    # x = 512
#     elif temp_var >= 256:
#         return 4096    # x = 256
#     elif temp_var >= 128:
#         return 5793    # x = 128
#     elif temp_var >= 64:
#         return 8192    # x = 64
#     elif temp_var >= 32:
#         return 11585   # x = 32
#     elif temp_var >= 16:
#         return 16384   # x = 16
#     elif temp_var >= 8:
#         return 23170   # x = 8
#     elif temp_var >= 4:
#         return 32768   # x = 4
#     elif temp_var >= 2:
#         return 46341   # x = 2
#     else:
#         return 65535   # x = 1 (최대값, 1.0)
    
    
    
    # 24'b1xxx_xxxx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd23; // x = 8388608, 1/sqrt(x) ≈ 0.000345
    # 24'b01xx_xxxx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd32; // x = 4194304, 1/sqrt(x) ≈ 0.000488
    # 24'b001x_xxxx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd45; // x = 2097152, 1/sqrt(x) ≈ 0.000691
    # 24'b0001_xxxx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd64; // x = 1048576, 1/sqrt(x) ≈ 0.000977
    # 24'b0000_1xxx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd91; // x = 524288, 1/sqrt(x) ≈ 0.001381
    # 24'b0000_01xx_xxxx_xxxx_xxxx_xxxx: o_std = 16'd128;        // x = 262144, 1/sqrt(x) ≈ 0.001953
    # 24'b0000_001x_xxxx_xxxx_xxxx_xxxx: o_std = 16'd181;        // x = 131072, 1/sqrt(x) ≈ 0.002762
    # 24'b0000_0001_xxxx_xxxx_xxxx_xxxx: o_std = 16'd256;        // x = 65536, 1/sqrt(x) ≈ 0.003906
    # 24'b0000_0000_1xxx_xxxx_xxxx_xxxx: o_std = 16'd362;        // x = 32768, 1/sqrt(x) ≈ 0.005524
    # 24'b0000_0000_01xx_xxxx_xxxx_xxxx: o_std = 16'd512;        // x = 16384, 1/sqrt(x) ≈ 0.007812
    # 24'b0000_0000_001x_xxxx_xxxx_xxxx: o_std = 16'd724;        // x = 8192, 1/sqrt(x) ≈ 0.011049
    # 24'b0000_0000_0001_xxxx_xxxx_xxxx: o_std = 16'd1024;       // x = 4096, 1/sqrt(x) ≈ 0.015625
    # 24'b0000_0000_0000_1xxx_xxxx_xxxx: o_std = 16'd1448;       // x = 2048, 1/sqrt(x) ≈ 0.022097
    # 24'b0000_0000_0000_01xx_xxxx_xxxx: o_std = 16'd2048;       // x = 1024, 1/sqrt(x) ≈ 0.031250
    # 24'b0000_0000_0000_001x_xxxx_xxxx: o_std = 16'd2896;       // x = 512, 1/sqrt(x) ≈ 0.044194
    # 24'b0000_0000_0000_0001_xxxx_xxxx: o_std = 16'd4096;       // x = 256, 1/sqrt(x) ≈ 0.062500
    # 24'b0000_0000_0000_0000_1xxx_xxxx: o_std = 16'd5793;       // x = 128, 1/sqrt(x) ≈ 0.088388
    # 24'b0000_0000_0000_0000_01xx_xxxx: o_std = 16'd8192;       // x = 64, 1/sqrt(x) ≈ 0.125000
    # 24'b0000_0000_0000_0000_001x_xxxx: o_std = 16'd11585;      // x = 32, 1/sqrt(x) ≈ 0.176777
    # 24'b0000_0000_0000_0000_0001_xxxx: o_std = 16'd16384;      // x = 16, 1/sqrt(x) ≈ 0.250000
    # 24'b0000_0000_0000_0000_0000_1xxx: o_std = 16'd23170;      // x = 8, 1/sqrt(x) ≈ 0.353553
    # 24'b0000_0000_0000_0000_0000_01xx: o_std = 16'd32768;      // x = 4, 1/sqrt(x) ≈ 0.500000
    # 24'b0000_0000_0000_0000_0000_001x: o_std = 16'd46341;      // x = 2, 1/sqrt(x) ≈ 0.707107
    # 24'b0000_0000_0000_0000_0000_0001: o_std = 16'd65535;      // x = 1, 1/sqrt(x) ≈ 1.000000
    

# =============================================================================
# Original RMSNorm (FP32 기반) – bias 없이
# =============================================================================
def original_rmsnorm(X, gamma, eps=1e-8):
    """
    FP32 RMSNorm: RMS = sqrt(mean(X**2) + eps), 출력 = gamma * (X / RMS)
    """
    X = np.array(X, dtype=float)
    rms = math.sqrt(np.mean(X**2) + eps)
    out = gamma * (X / rms)
    return out, rms

# =============================================================================
# Variant A: 분산이 음수일 경우 0으로 매핑 (기존 방식)
# =============================================================================
def stage1_detailed_variantA(X, alpha, zp, inv_n):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
    #    variance = 0
        stdinv_Q = 0
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
   # stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

# =============================================================================
# Variant D: AILayerNorm + RMSNorm (Dynamic Compression + RMSNorm) – int8 연산
# =============================================================================
def stage1_detailed_variantD(X, alpha, zp, inv_n, eps=1):
    """
    Variant D:
      - X: int32 배열 (입력은 이미 -128~127 범위의 정수)
      - 동적 압축으로 x^2를 근사하여 Ex2_acc 누적
      - Ex2_avg = (Ex2_acc << 4) / C, C는 총 입력 개수 (24비트 수준)
      - Ex2_avg를 그대로 temp_var로 사용 (이미 24비트 값이라 가정)
      - 24-entry LUT (lut_std)를 통해 std (근사 1/sqrt(variance) 값)를 구함
      - 최종 stdinv = std >> 16 (dequantize, Q0.16 → 정수)
      - RMSNorm은 평균 제거 없이 정규화 (mu=0)
    """
    C = len(X)
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for i, x in enumerate(X):
        Xi = int(x)
        Xc, s = dynamic_compress_D(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex2_avg = Ex2_acc // C  # 24비트 근사 분산 값
    temp_var = Ex2_avg  # 여기서 temp_var는 이미 24비트 값
    std = lut_std(temp_var)  # LUT를 통한 근사값 (Q0.16)
    stdinv = std / 256      # 최종 stdinv (정수)
    mu_val = 0
    debug_info = {"Ex2_avg": Ex2_avg, "temp_var": temp_var, "std": std, "stdinv": stdinv}
    return mu_val, stdinv, 0, Ex2_avg, None, None, debug_info

# =============================================================================
# Stage2 및 전체 AILayerNorm 함수 (variant에 따라 호출)
# =============================================================================
def stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=True):
    """
    Stage2:
      각 채널에 대해:
         1. X_norm_shift = X[i] << alpha (zero point 제거 없음)
         2. X_diff = X_norm_shift - μ
         3. temp_Norm = gamma[i] * stdinv * X_diff
         4. 최종 출력:
              - 일반 variant: o_Norm = lower8_hw(temp_Norm) + beta[i]
              - RMSNorm variant (D): o_Norm = lower8_hw(temp_Norm)  (beta 미사용)
    """
    Y = []
    for i, x in enumerate(X):
        Xi = int(x)
        X_norm_shift = Xi << alpha
        X_diff = X_norm_shift - mu
        temp_Norm = gamma[i] * stdinv * X_diff
        if add_beta:
            final = lower8_hw(temp_Norm) + beta[i]
        else:
            final = lower8_hw(temp_Norm)
        Y.append(final)
    return np.array(Y, dtype=np.int8)

def AILayerNorm(X, alpha, zp, gamma, beta, inv_n, variant="A", small_const=1):
    if variant == "A":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantA(X, alpha, zp, inv_n)
        add_beta = True
    elif variant == "D":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantD(X, alpha, zp, inv_n)
        add_beta = False  # RMSNorm: beta 미사용
    else:
        raise ValueError("알 수 없는 variant입니다. 'A', 'B', 'C', 'D' 중 하나를 선택하세요.")
    Y = stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=add_beta)
    return Y, mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info

def original_layernorm(X, gamma, beta, eps=1e-5):
    X = np.array(X, dtype=float)
    mean = np.mean(X)
    var = np.var(X, ddof=0)
    normalized = (X - mean) / math.sqrt(var + eps)
    out = normalized * gamma + beta
    quantized = np.array([lower8_hw(val) for val in out], dtype=np.int8)
    return quantized, mean, var

# =============================================================================
# FP32 -> int8 (대칭 양자화) 함수, 공통 scale 사용 버전
# =============================================================================
def quantize_fp32_to_int8_symmetric_with_scale(x, scale):
    """
    FP32 -> int8 (대칭 양자화):
      - x를 주어진 scale로 나눈 후, -127 ~ 127 범위로 클램핑하여 int8로 변환
    """
    x_quant = torch.round(x / scale)
    x_quant = torch.clamp(x_quant, -127, 127)
    return x_quant.to(torch.int8)

# =============================================================================
# Main Execution
# =============================================================================
# --- Main Execution (수정 부분) ---

if __name__ == '__main__':
    # 이미지 로드 및 전처리
    image = Image.open(r"C:\SYDLAB_SY\test_image_dog.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # LayerNorm 입력 및 파라미터 추출 (FP32)
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("원래 Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("원래 Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화 후 중앙 512개 값 선택
    flattened_input = layernorm_input.flatten()  # FP32
    total = flattened_input.numel()
    start_index = total // 2
    selected_input = flattened_input[start_index: start_index + 512]  # FP32


    # 각 선택된 값의 채널 번호 계산 (마지막 차원 크기가 768)
    indices = torch.arange(start_index, start_index + 512)
    channels = indices % gamma_tensor.numel()
    selected_gamma = gamma_tensor[channels]
    selected_beta = beta_tensor[channels]

    # fp32 -> int8 양자화
    selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
    selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
    selected_beta_int8, scale_beta = quantize_fp32_to_int8(selected_beta)

    # 선택된 값들을 numpy 배열로 변환 (512개의 int8 값)
    X_full = selected_input_int8.cpu().numpy()
    gamma_full = selected_gamma_int8.cpu().numpy()
    beta_full = selected_beta_int8.cpu().numpy()

    X_full_a = selected_input
    gamma_full_a = selected_gamma.cpu().numpy()
    beta_full_a = selected_beta.cpu().numpy()

    # Variant D 결과 저장용 딕셔너리 및 관련 리스트
    results_D = {}
    results_R = {}
    

    mae_list = []
    mse_list = []
    accuracy_list = []

    alpha = 2
    zp = 0
    inv_n = 32

    batches = 512 // 8  # 총 64 배치 (512개의 값을 8개씩 배치)

    # 각 variant별 전체 결과(전체 배치에 대해)를 저장할 리스트
    all_orig = []
    all_A = []
    all_B = []
    all_D = []
    for i in range(batches):
        X_batch = X_full[i*8:(i+1)*8]
        gamma_batch = gamma_full[i*8:(i+1)*8]
        X_batch_a = X_full_a[i*8:(i+1)*8]
        gamma_batch_a = gamma_full_a[i*8:(i+1)*8]
        beta_batch = beta_full[i*8:(i+1)*8]
        # 원래 LayerNorm
        orig_Y, _, _ = original_layernorm(X_batch, gamma_batch, beta_batch)
        all_orig.append(orig_Y)
        # Variant A
        Y_A, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="A")
        all_A.append(Y_A)
        # 원래 RMSNorm
        Y_B, _ = original_rmsnorm(X_batch_a, gamma_batch_a, eps=1e-8)
        orig_rmsnorm_out_tensor = torch.tensor(Y_B, dtype=torch.float32)
        orig_rmsnorm_out_int8, scale_orig = quantize_fp32_to_int8(orig_rmsnorm_out_tensor)
        all_B.append(orig_rmsnorm_out_int8)
        # Variant D (RMSNorm: beta 미사용, mu=0)
        Y_D, _, _, _, _, _, _, _ = AILayerNorm(X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="D")
        all_D.append(Y_D)

    

    # 전체 결과 배열로 결합
    all_orig = np.concatenate(all_orig)
    all_A = np.concatenate(all_A)
    all_B = np.concatenate(all_B)
    all_D = np.concatenate(all_D)

    # 전체 MAE 및 Accuracy 계산 (각 variant)
    mae_A = np.mean(np.abs(all_orig - all_A))
    acc_A = (1 - mae_A / 255) * 100

    mae_B = np.mean(np.abs(all_orig - all_B))
    acc_B = (1 - mae_B / 255) * 100

    mae_D = np.mean(np.abs(all_orig - all_D))
    acc_D = (1 - mae_D / 255) * 100

    print("\nOverall Results:")
   # print("Original LayerNorm Output (first 20 values):", all_orig[:20])
    print("AILayerNorm           -> MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_A, acc_A))
    print("RMSNorm               -> MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_B, acc_B))
    print("AILayerNorm + RMSNorm -> MAE: {:.2f}, Accuracy: {:.2f}%".format(mae_D, acc_D))

    # 한 화면에 1개의 서브플롯으로 전체 MAE와 Accuracy를 막대 그래프로 그리기
    variants = ['AILayerNorm', 'RMSNorm', 'AILayerNorm + RMSNorm']
    mae_values = [mae_A, mae_B, mae_D]
    acc_values = [acc_A, acc_B, acc_D]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(variants, mae_values, width=0.3, color=['blue', 'orange', 'green', 'red'])
    axs[0].set_title("Overall MAE Comparison")
    axs[0].set_ylabel("MAE")
    axs[1].bar(variants, acc_values, width=0.3, color=['blue', 'orange', 'green', 'red'])
    axs[1].set_title("Overall Accuracy Comparison")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_yticks(np.arange(0, 110, 10))

    plt.tight_layout()
    plt.show()






########################################################################################################################

########################################## AILayerNorm negative variance count #########################################

########################################################################################################################

'''
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt  # pip install matplotlib

# =============================================================================
# Swin‑Tiny 모델 및 전처리 설정 (timm 사용)
# =============================================================================
#model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
model = timm.create_model('swin_base_patch4_window12_384.ms_in1k', pretrained=True)

model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# =============================================================================
# 마지막 LayerNorm 입력 및 파라미터 추출 함수
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    # 모델 내 모든 LayerNorm 모듈 중 마지막 모듈 선택
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()  # 입력값 추출
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()  # [channels] (예: [768])
    beta  = final_norm.bias.detach()      # [channels]
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 하드웨어/소프트웨어용 함수들 (AILayerNorm SW 코드)
# =============================================================================
def lower8(x):
    """
    주어진 실수 x를 정수로 변환한 후,
    하위 8비트를 그대로 추출하여 부호 있는 int8 값으로 해석합니다.
    (예: 190 → (190 & 0xFF)=190 → 190>=128이면 190-256 = -66)
    """
    x_int = int(round(x))
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def lower8_hw(x):
    """
    하드웨어 방식과 동일하게 정수 x의 하위 8비트를 부호 있는 int8으로 변환합니다.
    """
    x_int = int(x)
    r = x_int & 0xFF
    if r >= 128:
        r -= 256
    return r

def dynamic_compress(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 1
        Xc = (x + 8) // 16
    else:
        s = 0
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

def dynamic_compress_D(x):
    """
    Dynamic Compress:
      - 입력 x가 음수이면 0으로 클립
      - x의 상위 2비트가 nonzero이면 division by 16 (오른쪽 4비트 시프트 + 반올림), s = 1
      - 그렇지 않으면 division by 4 (오른쪽 2비트 시프트 + 반올림), s = 0
      - 결과는 0~15 범위의 정수.
    """
    if x < 0:
        x = 0
    top2 = (x >> 6) & 0x3
    if top2 != 0:
        s = 2
        Xc = (x + 8) // 16
    else:
        s = 1
        Xc = (x + 2) // 4
    Xc = np.clip(Xc, 0, 15)
    return Xc, s

def lut_std(temp_var):
    """
    LUT를 통한 stdinv 근사 (Q0.8 값):
      temp_var: 분산의 하위 16비트 값 (0~65535)
      원래 LUT 매핑 값은:
         1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 255
      여기서는 그대로 반환합니다.
    """
    if temp_var >= 32768:
        return 1
    elif temp_var >= 16384:
        return 2
    elif temp_var >= 8192:
        return 3
    elif temp_var >= 4096:
        return 4
    elif temp_var >= 2048:
        return 6
    elif temp_var >= 1024:
        return 8
    elif temp_var >= 512:
        return 11
    elif temp_var >= 256:
        return 16
    elif temp_var >= 128:
        return 23
    elif temp_var >= 64:
        return 32
    elif temp_var >= 32:
        return 45
    elif temp_var >= 16:
        return 64
    elif temp_var >= 8:
        return 90
    elif temp_var >= 4:
        return 128
    elif temp_var >= 2:
        return 181
    elif temp_var >= 1:
        return 255
    else:
        return 0

# =============================================================================
# Variant A: 분산이 음수일 경우 0으로 매핑 (기존 방식)
# =============================================================================
def stage1_detailed_variantA(X, alpha, zp, inv_n):
    C = len(X)
    Ex_acc = 0
    Ex2_acc = 0
    lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}
    for x in X:
        Xi = int(x) - zp
        Xc, s = dynamic_compress(abs(Xi))
        Xc_square = lut_square.get(Xc, 0)
        Xc_decomp = Xc_square << (4 * s)
        Ex_acc += Xi << alpha
        Ex2_acc += Xc_decomp << (2 * alpha)
    Ex_avg = Ex_acc / C
    Ex2_avg = (Ex2_acc << 4) / C

    ex_unit_out = (Ex_acc * inv_n) >> 8
    ex2_unit_out = (Ex2_acc * inv_n) >> 8

    variance = ex2_unit_out - (ex_unit_out ** 2)
    temp_var = int(variance) & 0xFFFF
    if variance < 0:
    #    variance = 0
        stdinv_Q = 0
    else:
        stdinv_Q = lut_std(temp_var)
   # temp_var = int(variance) & 0xFFFF
   # stdinv_Q = lut_std(temp_var)
    stdinv = stdinv_Q / 256.0

    mu_val = Ex_avg

    debug_info = {
        "Ex_avg": Ex_avg,
        "Ex2_avg": Ex2_avg,
        "variance": variance,
        "temp_var": temp_var,
        "stdinv_Q": stdinv_Q,
        "stdinv": stdinv
    }
    return mu_val, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info



# =============================================================================
# Stage2 및 전체 AILayerNorm 함수 (variant에 따라 호출)
# =============================================================================
def stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=True):
    """
    Stage2:
      각 채널에 대해:
         1. X_norm_shift = X[i] << alpha (zero point 제거 없음)
         2. X_diff = X_norm_shift - μ
         3. temp_Norm = gamma[i] * stdinv * X_diff
         4. 최종 출력:
              - 일반 variant: o_Norm = lower8_hw(temp_Norm) + beta[i]
              - RMSNorm variant (D): o_Norm = lower8_hw(temp_Norm)  (beta 미사용)
    """
    Y = []
    for i, x in enumerate(X):
        Xi = int(x)
        X_norm_shift = Xi << alpha
        X_diff = X_norm_shift - mu
        temp_Norm = gamma[i] * stdinv * X_diff
        if add_beta:
            final = lower8_hw(temp_Norm) + beta[i]
        else:
            final = lower8_hw(temp_Norm)
        Y.append(final)
    return np.array(Y, dtype=np.int8)

def AILayerNorm(X, alpha, zp, gamma, beta, inv_n, variant="A", small_const=1):
    if variant == "A":
        mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info = stage1_detailed_variantA(X, alpha, zp, inv_n)
        add_beta = True

    else:
        raise ValueError("알 수 없는 variant입니다. 'A', 'B', 'C', 'D' 중 하나를 선택하세요.")
    Y = stage2(X, alpha, mu, stdinv, gamma, beta, zp, add_beta=add_beta)
    return Y, mu, stdinv, Ex_acc, Ex2_acc, ex_unit_out, ex2_unit_out, debug_info


def display_full_input(X_full):
    width = len(X_full) * 8
    hex_str = "_".join(f"{x:02X}" for x in X_full)
    dec_str = "_".join(str(x) for x in X_full)
    print(f"Input X = {width}'h{hex_str};")
    print(f"// {width}'d{dec_str}")

def quantize_fp32_to_int8(x):
    max_val = torch.max(torch.abs(x))
    scale = max_val / 127 if max_val > 0 else 1.0
    x_int8 = torch.round(x / scale)
    x_int8 = torch.clamp(x_int8, -128, 127)
    return x_int8.to(torch.int8), scale



# =============================================================================
# Swin 모델에서 LayerNorm 입력 및 파라미터 추출
# =============================================================================
def get_layernorm_input_and_params(model, pixel_values):
    norm_layers = [module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("LayerNorm 모듈을 찾을 수 없습니다.")
    final_norm = norm_layers[-1]
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data["layernorm_input"] = input[0].detach()
    hook_handle = final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(pixel_values)
    hook_handle.remove()
    
    gamma = final_norm.weight.detach()
    beta  = final_norm.bias.detach()
    return hook_data["layernorm_input"], gamma, beta

# =============================================================================
# 메인 실행부 (전체 입력 사용, 배치별 출력 없이 전체 overall 값만 계산)
# =============================================================================
if __name__ == '__main__':
    # 이미지 로드 및 전처리 (timm transform 사용)
    image = Image.open(r"C:\SYDLAB_SY\test_image_road.jpg").convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # 원래 LayerNorm 입력 및 파라미터 추출
    layernorm_input, gamma_tensor, beta_tensor = get_layernorm_input_and_params(model, pixel_values)
    print("원래 LayerNorm 입력값 shape:", layernorm_input.shape)  # 예: [1, 7, 7, 768]
    print("원래 Gamma shape:", gamma_tensor.shape)              # 예: [768]
    print("원래 Beta shape:", beta_tensor.shape)                # 예: [768]

    # 평탄화하여 전체 입력 사용 (전체 값 사용)
    flattened_input = layernorm_input.flatten()  # 예: 7*7*768 = 37632
    total = flattened_input.numel()
    selected_input = flattened_input  # 전체 입력 사용

    # 전체 입력에 대해 각 요소의 채널 번호 계산 (마지막 차원 크기 768)
    indices = torch.arange(0, total)
    channels = indices % gamma_tensor.numel()
    selected_gamma = gamma_tensor[channels]
    selected_beta = beta_tensor[channels]

    # fp32 -> int8 양자화
    selected_input_int8, scale_input = quantize_fp32_to_int8(selected_input)
    selected_gamma_int8, scale_gamma = quantize_fp32_to_int8(selected_gamma)
    selected_beta_int8, scale_beta = quantize_fp32_to_int8(selected_beta)


    print("\n선택된 LayerNorm 입력 ({} int8 값, {} bits):\n".format(total, total*8), selected_input_int8)
    print("입력 scale:", scale_input)
    print("\n선택된 Gamma ({} int8 값):\n".format(total), selected_gamma_int8)
    print("Gamma scale:", scale_gamma)
    print("\n선택된 Beta ({} int8 값):\n".format(total), selected_beta_int8)
    print("Beta scale:", scale_beta)

    # AILayerNorm 파라미터 설정 (입력이 이미 -128~127이므로 zp = 0)
    zp = 0
    alpha = 2
    inv_n = 32

    # 전체 입력 및 파라미터를 numpy 배열로 변환
    X_full = selected_input_int8.cpu().numpy()  # shape: (total,)
    gamma_full = selected_gamma_int8.cpu().numpy()  # shape: (total,)
    beta_full = selected_beta_int8.cpu().numpy()      # shape: (total,)

    # Variant A 결과 저장용 딕셔너리
    result_A = {}
    batches = total // 8  # 총 64 배치

    for i in range(batches):
        X_batch = X_full[i*8:(i+1)*8]           # 8개 입력 (int8 numpy 배열)
        gamma_batch = gamma_full[i*8:(i+1)*8]     # 8개 gamma (int8 numpy 배열)
        beta_batch = beta_full[i*8:(i+1)*8]       # 8개 beta (int8 numpy 배열)

        # Variant A 계산 (AILayerNorm 함수 호출, variant="A")
        Y_A, mu_A, stdinv_A, Ex_acc_A, Ex2_acc_A, ex_out_A, ex2_out_A, debug_info_A = AILayerNorm(
            X_batch, alpha, zp, gamma_batch, beta_batch, inv_n, variant="A"
        )
        result_A[f"Batch {i+1}"] = {
            "Input": X_batch,
            "Stage1_mu": mu_A,
            "Stage1_std (1/stdinv)": (1/stdinv_A if stdinv_A != 0 else 0),
            "Final Y (Variant A)": Y_A,
            "Variance": debug_info_A.get("variance", None)
        }

    # 배치별 분산 값을 저장할 리스트
    variance_values = []
    negative_variance_batches = 0

    print("\n==== Summary of All Batches (Variant A) ====")
    for batch_name, info in result_A.items():
        var_val = info["Variance"]
        variance_values.append(var_val)
        if var_val is not None and var_val < 0:
            negative_variance_batches += 1
        # print(f"\n{batch_name}")
        # print("Input X:", info["Input"])
        # print("Stage1: mu =", info["Stage1_mu"], ", 1/stdinv =", info["Stage1_std (1/stdinv)"])
        # print("Final Y (Variant A):", info["Final Y (Variant A)"])
        # print("Variance:", var_val)

    total_batches = len(result_A)
    print(f"\n전체 {total_batches} 배치 중 음수 분산을 가진 배치는 {negative_variance_batches} 배치입니다.")

    # =============================================================================
    # 분산 값 막대그래프 출력 (음수인 분산은 빨간색, 양수는 파란색)
    # =============================================================================
    import matplotlib.pyplot as plt

    batch_numbers = list(range(1, total_batches + 1))
    colors = ['red' if (v is not None and v < 0) else 'blue' for v in variance_values]

    plt.figure(figsize=(10, 5))
    plt.bar(batch_numbers, variance_values, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Batch Number")
    plt.ylabel("Variance Value")
    plt.title("Variance Values per Batch (Negative in Red)")
    plt.show()
'''