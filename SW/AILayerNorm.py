import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from sqrt_hp import sqrt_rounded, sqrt_no_rounded

# — STE wrappers —
class round_ste(Function):
    @staticmethod
    def forward(ctx, x):    return torch.round(x)
    @staticmethod
    def backward(ctx, grad): return grad.clone()

class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):    return torch.floor(x)
    @staticmethod
    def backward(ctx, grad): return grad.clone()
# — QSNR 계산 함수 —
def calculate_qsnr(original_signal: torch.Tensor, quantized_signal: torch.Tensor) -> torch.Tensor:
    """
    QSNR (Quantization Signal-to-Noise Ratio)을 계산합니다.

    Args:
        original_signal (torch.Tensor): 원본 신호 텐서.
        quantized_signal (torch.Tensor): 양자화된 신호 텐서.

    Returns:
        torch.Tensor: 계산된 QSNR 값 (dB).
    """
    if original_signal.shape != quantized_signal.shape:
        raise ValueError("원본 신호와 양자화된 신호의 텐서 형태가 같아야 합니다.")

    # 원본 신호의 에너지 (제곱의 합)
    power_original = torch.sum(original_signal**2)

    # 양자화 오류 (원본 신호 - 양자화된 신호)
    quantization_error = original_signal - quantized_signal

    # 양자화 오류의 에너지 (제곱의 합)
    power_error = torch.sum(quantization_error**2)

    # 분모(power_error)가 0이 되는 것을 방지 (매우 작은 값 추가)
    # 실제 환경에서는 양자화 오류가 0인 경우는 거의 없지만,
    # 이론적으로 발생할 수 있으므로 안정성을 위해 추가합니다.
    epsilon = 1e-10
    if power_error < epsilon:
        print("경고: 양자화 오류 에너지가 매우 작습니다. QSNR이 매우 커질 수 있습니다.")
        power_error = epsilon

    # QSNR 계산 (dB 단위)
    qsnr = 10 * torch.log10(power_original / power_error)

    return qsnr

# — 기존 LUT 기반 AILayerNorm —
class AILayerNorm(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5, alpha=2):
        super().__init__()
        self.N, self.eps, self.alpha, self.affine = normalized_shape, eps, alpha, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta',  None)

        # 4-bit square LUT
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)
        # 16-entry priority-encoder LUT for 1/sqrt(var)
        lut_vals = [
            65535, 46341, 32768, 23170,
            16384, 11585,  8192,  5793,
             4096,  2896,  2048,  1448,
             1024,   724,   512,   362
        ]
        self.register_buffer('inv_sqrt_lut', torch.tensor(lut_vals, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, self.N)

        # 1) Fake‐quantize input
        scale_in = (x.abs().max() / 127.0).to(x.dtype).to(x.device)
        x_int    = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q      = x_int * scale_in                                   # de-quantize back to float


        # 2) Ex
        Ex = x_q.sum(dim=1, keepdim=True)

        # 3) Dynamic‐compress Ex2
        abs_q    = x_q.abs()
        top2     = floor_ste.apply(abs_q / 64.0)
        idx_h    = floor_ste.apply(abs_q / 16.0).clamp(0,15).long()
        idx_m    = (floor_ste.apply(abs_q / 2.0) % 16.0).clamp(0,15).long()
        idx      = torch.where(top2>=1, idx_h, idx_m)
        sq       = self.square_lut[idx]
        sq_decomp= torch.where(top2>=1, sq * 16.0, sq)
        sq_ref   = sq_decomp * (2**(2*self.alpha))
        Ex2      = sq_ref.sum(dim=1, keepdim=True)

        # 4) mean & var
        mu  = Ex / self.N
        var = Ex2 / self.N - mu*mu

        # 5) 1/sqrt(var) via LUT
        var_int = round_ste.apply(var).clamp(1, 2**16-1).long()
        msb     = floor_ste.apply(torch.log2(var_int.float())).long().clamp(0,15)
        inv_std = self.inv_sqrt_lut[msb].float() / (2**16)

        # 6) normalize + affine
        x_norm = (x_q - mu) * inv_std
        y      = x_norm * self.gamma + self.beta if self.affine else x_norm

        # 7) Fake‐quantize output
        scale_out = (y.abs().max() / 127.0).to(y.dtype).to(y.device)
        y_int     = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q       = y_int * scale_out

        return y_q


# — Improved AILayerNorm —
class ImprovedAILayerNorm(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5):
        super().__init__()
        self.N, self.eps, self.affine = normalized_shape, eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta',  None)

        # 4-bit square LUT
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, self.N)

        # 1) Fake‐quantize input
        scale_in = (x.abs().max() / 127.0).to(x.dtype).to(x.device)
        x_int    = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q      = x_int * scale_in                                   # de-quantize back to float


        # 2) Ex
        Ex = x_q.sum(dim=1, keepdim=True)


        # 3) Hybrid Squaring Ex2 (수정된 코드)
        # x_int는 float이지만 정수 값을 가짐. 연산을 위해 long 타입으로 변환
        X_abs = x_int.abs().long()

        # 상위 4비트(H)와 하위 4비트(L) 분리
        H = X_abs >> 4
        L = X_abs & 0x0F  # 0x0F는 0b1111, 하위 4비트만 남기는 마스크

        # X^2 = 256*H^2 + 32*H*L + L^2 계산
        H_sq = self.square_lut[H]
        L_sq = self.square_lut[L]
        H_x_L = H * L
        
        # 각 항을 계산하고 더해서 최종 제곱값을 구함
        x_squared_int = H_sq * 256.0 + H_x_L * 32.0 + L_sq
        
        # x_int^2에 scale_in^2을 곱해 x_q^2와 동일한 스케일로 보정
        x_squared_q = x_squared_int * (scale_in ** 2)

        # 최종 제곱값들의 합을 구함
        Ex2 = x_squared_q.sum(dim=1, keepdim=True)
        # 

        # 4) mean & var
        mu  = Ex / self.N
        var = Ex2 / self.N - mu*mu


        # 5) 1/sqrt(var) via sqrt_rounded
        # 분산을 16비트 정수로 변환합니다.
        var_int = round_ste.apply(var).clamp(1, 2**16-1).long()

        # sqrt_rounded 함수는 단일 정수 값에 대해 동작하므로,
        # 배치 내 각 원소에 대해 함수를 적용합니다.
        std_int = torch.tensor(
            [sqrt_rounded(v.item()) for v in var_int.flatten()],
            device=var.device,
            dtype=torch.float32 # 이후 나눗셈을 위해 float으로 변환
        ).view_as(var_int)

        # 0으로 나누는 것을 방지하고 역수를 계산합니다.
        inv_std = 1.0 / std_int.clamp(min=self.eps)


        # 6) normalize + affine
        x_norm = (x_q - mu) * inv_std
        y      = x_norm * self.gamma + self.beta if self.affine else x_norm

        # 7) Fake‐quantize output
        scale_out = (y.abs().max() / 127.0).to(y.dtype).to(y.device)
        y_int     = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q       = y_int * scale_out

        return y_q
    


# no rounding version
class ImprovedAILayerNorm_no_round(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5):
        super().__init__()
        self.N, self.eps, self.affine = normalized_shape, eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta',  None)

        # 4-bit square LUT
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, self.N)

        # 1) Fake‐quantize input
        scale_in = (x.abs().max() / 127.0).to(x.dtype).to(x.device)
        x_int    = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q      = x_int * scale_in                                   # de-quantize back to float


        # 2) Ex
        Ex = x_q.sum(dim=1, keepdim=True)


        # 3) Hybrid Squaring Ex2 (수정된 코드)
        # x_int는 float이지만 정수 값을 가짐. 연산을 위해 long 타입으로 변환
        X_abs = x_int.abs().long()

        # 상위 4비트(H)와 하위 4비트(L) 분리
        H = X_abs >> 4
        L = X_abs & 0x0F  # 0x0F는 0b1111, 하위 4비트만 남기는 마스크

        # X^2 = 256*H^2 + 32*H*L + L^2 계산
        H_sq = self.square_lut[H]
        L_sq = self.square_lut[L]
        H_x_L = H * L
        
        # 각 항을 계산하고 더해서 최종 제곱값을 구함
        x_squared_int = H_sq * 256.0 + H_x_L * 32.0 + L_sq
        
        # x_int^2에 scale_in^2을 곱해 x_q^2와 동일한 스케일로 보정
        x_squared_q = x_squared_int * (scale_in ** 2)

        # 최종 제곱값들의 합을 구함
        Ex2 = x_squared_q.sum(dim=1, keepdim=True)
        # 

        # 4) mean & var
        mu  = Ex / self.N
        var = Ex2 / self.N - mu*mu


        # 5) 1/sqrt(var) via sqrt_rounded
        # 분산을 16비트 정수로 변환합니다.
        var_int = round_ste.apply(var).clamp(1, 2**16-1).long()

        # sqrt_rounded 함수는 단일 정수 값에 대해 동작하므로,
        # 배치 내 각 원소에 대해 함수를 적용합니다.
        std_int = torch.tensor(
            [sqrt_no_rounded(v.item()) for v in var_int.flatten()],
            device=var.device,
            dtype=torch.float32 # 이후 나눗셈을 위해 float으로 변환
        ).view_as(var_int)

        # 0으로 나누는 것을 방지하고 역수를 계산합니다.
        inv_std = 1.0 / std_int.clamp(min=self.eps)


        # 6) normalize + affine
        x_norm = (x_q - mu) * inv_std
        y      = x_norm * self.gamma + self.beta if self.affine else x_norm

        # 7) Fake‐quantize output
        scale_out = (y.abs().max() / 127.0).to(y.dtype).to(y.device)
        y_int     = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q       = y_int * scale_out

        return y_q

# — Direct AILayerNorm (no LUT, direct sqrt/div) —

class AILayerNormDirect(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5):
        super().__init__()
        self.N, self.eps, self.affine = normalized_shape, eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta',  None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.reshape(B, -1, self.N)

        # 1) Fake‐quantize input
        scale_in = (x.abs().max() / 127.0).to(x.dtype).to(x.device)
        x_int    = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q      = x_int * scale_in  


        # 2) Ex
        Ex = x_q.sum(dim=1, keepdim=True)

        # 3) Ex2 = Σ x_q**2
        Ex2 = (x_q * x_q).sum(dim=1, keepdim=True)

        # 4) mean & var
        mu  = Ex / self.N
        var = Ex2 / self.N - mu*mu

        # 5) direct sqrt & inverse
        std    = torch.sqrt(var + self.eps)
        inv_std= 1.0 / std

        # 6) normalize + affine
        x_norm = (x_q - mu) * inv_std
        y      = x_norm * self.gamma + self.beta if self.affine else x_norm

        # 7) Fake‐quantize output
        scale_out = (y.abs().max() / 127.0).to(y.dtype).to(y.device)
        y_int     = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q       = y_int * scale_out

        return y_q

# — Test —
if __name__ == "__main__":
    torch.manual_seed(1234)
    x = torch.randn(16, 128, dtype=torch.float32, requires_grad=True) * 10
#    x = torch.randn(16, 128, dtype=torch.float32, device='cuda', requires_grad=True) * 10
    # instantiate modules
    lut_model   = AILayerNorm(128, affine=True)
    improved_model   = ImprovedAILayerNorm(128, affine=True)
    improved_model_no_round   = ImprovedAILayerNorm_no_round(128, affine=True)
    sw_model     = AILayerNormDirect(128, affine=True)

    # copy γ,β from a reference LayerNorm so both use same
    ref_ln = nn.LayerNorm(128, eps=1e-5)
    with torch.no_grad():
        lut_model.gamma.copy_(ref_ln.weight);        lut_model.beta.copy_(ref_ln.bias)
        improved_model.gamma.copy_(ref_ln.weight);   improved_model.beta.copy_(ref_ln.bias)
        improved_model_no_round.gamma.copy_(ref_ln.weight);   improved_model_no_round.beta.copy_(ref_ln.bias)
        sw_model.gamma.copy_(ref_ln.weight);         sw_model.beta.copy_(ref_ln.bias)

    # reference full-precision output
    y_fp       = ref_ln(x)
    # quantized outputs
    y_q_lut     = lut_model(x)
    y_q_improve = improved_model(x)
    y_q_improveno_round = improved_model_no_round(x)
    y_q_sw      = sw_model(x)

    # compute QSNRs
    qsnr_lut        = calculate_qsnr(y_fp, y_q_lut)         
    qsnr_improve    = calculate_qsnr(y_fp, y_q_improve)      
    qsnr_improve_no_round    = calculate_qsnr(y_fp, y_q_improveno_round)     
    qsnr_sw         = calculate_qsnr(y_fp, y_q_sw)      
    
    print(f"QSNR (SW)          : {qsnr_sw:.2f} dB")
    print(f"QSNR (Orig_HW)     : {qsnr_lut:.2f} dB")
    print(f"QSNR (Improved_HW) : {qsnr_improve:.2f} dB")
    print(f"QSNR (Improved_HW_No_Round) : {qsnr_improve_no_round:.2f} dB")


