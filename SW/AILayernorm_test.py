# FP32 기준 Layernorm 대비 INT8 기준 원본 AILayernorm과 개선된 AILayernorm과 RMSNorm의 QSNR 비교

import math
import torch
import torch.nn as nn
from torch.autograd import Function

# --- 헬퍼 함수 및 클래스 ---
sqrt_lut = [math.isqrt(i * 256 + 128) for i in range(256)]

def sqrt_rounded(d_in: int) -> int:
    if d_in == 0: return 0
    # 입력 범위를 16비트로 클램핑
    if d_in > 65535: d_in = 65535
    msb_pos = d_in.bit_length() - 1
    k = msb_pos // 2
    norm_shift = (7 - k) * 2
    d_norm = d_in << norm_shift
    lut_addr = (d_norm >> 8) & 0xFF
    sqrt_mantissa = sqrt_lut[lut_addr]
    q_floor = sqrt_mantissa >> (7 - k)
    boundary = q_floor * q_floor + q_floor
    return q_floor + 1 if d_in > boundary else q_floor

class round_ste(Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, grad): return grad.clone()

def calculate_qsnr(original_signal: torch.Tensor, quantized_signal: torch.Tensor) -> torch.Tensor:
    power_original = torch.sum(original_signal**2)
    quantization_error = original_signal - quantized_signal
    power_error = torch.sum(quantization_error**2)
    epsilon = 1e-10
    if power_error < epsilon: power_error = epsilon
    qsnr = 10 * torch.log10(power_original / power_error)
    return qsnr

# --- 모델 정의 ---

# 1. Original AILayerNorm (초기 거친 근사 버전)
class OriginalAILayerNorm(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5):
        super().__init__()
        self.N, self.eps, self.affine = normalized_shape, eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)
        # 참고: 이 LUT는 이 모델에서만 사용됨
        lut_vals = [65535, 46341, 32768, 23170, 16384, 11585, 8192, 5793, 4096, 2896, 2048, 1448, 1024, 724, 512, 362]
        self.register_buffer('inv_sqrt_lut', torch.tensor(lut_vals, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        scale_in = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_int = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q = x_int * scale_in
        Ex = x_q.sum(dim=1, keepdim=True)
        # Dynamic-compress (스케일 보정 없음)
        abs_q = x_q.abs()
        top2 = torch.floor(abs_q / 64.0)
        idx_h = torch.floor(abs_q / 16.0).clamp(0, 15).long()
        idx_m = (torch.floor(abs_q / 2.0) % 16.0).clamp(0, 15).long()
        idx = torch.where(top2 >= 1, idx_h, idx_m)
        sq = self.square_lut[idx]
        sq_decomp = torch.where(top2 >= 1, sq * 256.0, sq * 16.0)
        Ex2 = sq_decomp.sum(dim=1, keepdim=True) # 스케일이 맞지 않는 Ex2
        mu = Ex / self.N
        var = Ex2 / self.N - mu*mu # 잘못된 분산 계산
        var_int = round_ste.apply(var).clamp(1, 2**16 - 1).long()
        msb = torch.floor(torch.log2(var_int.float() + 1e-8)).long().clamp(0, 15)
        inv_std = self.inv_sqrt_lut[msb].float() / (2**16)
        x_norm = (x_q - mu) * inv_std
        y = x_norm * self.gamma + self.beta
        scale_out = (y.abs().max() / 127.0).clamp(min=1e-8)
        y_int = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q = y_int * scale_out
        return y_q

# 2. Improved AILayerNorm (개선 버전)
class ImprovedAILayerNorm(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5):
        super().__init__()
        self.N, self.eps, self.affine = normalized_shape, eps, affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        scale_in = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_int = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q = x_int * scale_in
        Ex = x_q.sum(dim=1, keepdim=True)
        # Hybrid Squaring (스케일 보정 포함)
        X_abs = x_int.abs().long()
        H = X_abs >> 4
        L = X_abs & 0x0F
        H_sq = self.square_lut[H]
        L_sq = self.square_lut[L]
        H_x_L = H * L
        x_squared_int = H_sq * 256.0 + H_x_L * 32.0 + L_sq
        x_squared_q = x_squared_int * (scale_in ** 2)
        Ex2 = x_squared_q.sum(dim=1, keepdim=True)
        mu = Ex / self.N
        var = (Ex2 / self.N - mu*mu).clamp(min=0)
        var_int = round_ste.apply(var).clamp(1, 2**16-1).long()
        std_int = torch.tensor([sqrt_rounded(v.item()) for v in var_int.flatten()], device=var.device, dtype=torch.float32).view_as(var_int)
        inv_std = 1.0 / std_int.clamp(min=self.eps)
        x_norm = (x_q - mu) * inv_std
        y = x_norm * self.gamma + self.beta
        scale_out = (y.abs().max() / 127.0).clamp(min=1e-8)
        y_int = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q = y_int * scale_out
        return y_q

# 3. Approximated RMSNorm (ARMSNorm)
class ARMSNorm(nn.Module):
    def __init__(self, normalized_shape, affine=True, eps=1e-5, bias=False):
        super().__init__()
        self.d, self.eps, self.affine, self.use_bias = normalized_shape, eps, affine, bias
        if affine: self.weight = nn.Parameter(torch.ones(normalized_shape))
        else: self.register_parameter('weight', None)
        if self.use_bias: self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else: self.register_parameter('bias', None)
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_in = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_int = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q = x_int * scale_in
        X_abs = x_int.abs().long()
        H = X_abs >> 4
        L = X_abs & 0x0F
        H_sq = self.square_lut[H]
        L_sq = self.square_lut[L]
        H_x_L = H * L
        x_squared_int = H_sq * 256.0 + H_x_L * 32.0 + L_sq
        x_squared_q = x_squared_int * (scale_in ** 2)
        Ex2 = x_squared_q.sum(dim=-1, keepdim=True)
        mean_sq = (Ex2 / self.d).clamp(min=0)
        var_int = round_ste.apply(mean_sq).clamp(1, 2**16-1).long()
        std_int = torch.tensor([sqrt_rounded(v.item()) for v in var_int.flatten()], device=mean_sq.device, dtype=torch.float32).view_as(var_int)
        inv_std = 1.0 / std_int.clamp(min=self.eps)
        x_normed = x_q * inv_std
        y = self.weight * x_normed
        if self.use_bias: y = y + self.bias
        scale_out = (y.abs().max() / 127.0).clamp(min=1e-8)
        y_int = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q = y_int * scale_out
        return y_q

# --- 비교 테스트 ---
if __name__ == "__main__":
    D_MODEL = 128
    BATCH_SIZE = 16
    torch.manual_seed(1234)

    # 테스트 입력
    x = torch.randn(BATCH_SIZE, D_MODEL, dtype=torch.float32) * 5

    # --- 기준 모델 (정답) ---
    ref_ln = nn.LayerNorm(D_MODEL)

    # --- 비교할 근사 모델들 ---
    orig_ai_ln = OriginalAILayerNorm(D_MODEL)
    impr_ai_ln = ImprovedAILayerNorm(D_MODEL)
    approx_rms_ln = ARMSNorm(D_MODEL)

    # 공정한 비교를 위해 기준 모델의 gamma, beta 값을 모두 복사
    with torch.no_grad():
        orig_ai_ln.gamma.copy_(ref_ln.weight)
        orig_ai_ln.beta.copy_(ref_ln.bias)
        impr_ai_ln.gamma.copy_(ref_ln.weight)
        impr_ai_ln.beta.copy_(ref_ln.bias)
        approx_rms_ln.weight.copy_(ref_ln.weight) # ARMSNorm은 beta(bias)가 없으므로 weight만 복사

    # --- 각 모델의 출력 계산 ---
    y_ref = ref_ln(x)
    y_orig_ai = orig_ai_ln(x)
    y_impr_ai = impr_ai_ln(x)
    y_approx_rms = approx_rms_ln(x)

    # --- QSNR 계산 ---
    qsnr_orig_ai = calculate_qsnr(y_ref, y_orig_ai)
    qsnr_impr_ai = calculate_qsnr(y_ref, y_impr_ai)
    qsnr_approx_rms = calculate_qsnr(y_ref, y_approx_rms)

    print("="*20, " LayerNorm 대체 방식 QSNR 비교 ", "="*20)
    print(f"(기준: FP32 nn.LayerNorm)")
    print("-" * 65)
    print(f"1. Original AILayerNorm: {qsnr_orig_ai:.2f} dB  (초기 거친 근사 버전)")
    print(f"2. Improved AILayerNorm: {qsnr_impr_ai:.2f} dB  (개선된 LayerNorm 근사)")
    print(f"3. ARMSNorm            : {qsnr_approx_rms:.2f} dB  (RMSNorm 근사)")
    print("-" * 65)