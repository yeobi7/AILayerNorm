import math
import torch
import torch.nn as nn
from torch.autograd import Function

# --- 헬퍼 함수 및 클래스 (이전과 동일) ---
sqrt_lut = [math.isqrt(i * 256 + 128) for i in range(256)]
def sqrt_rounded(d_in: int) -> int:
    if d_in == 0: return 0
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

    
    # --- INT8 정밀도 근사 모델 ---
class Int8LayerNorm(nn.Module):
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

# --- 비교 테스트 ---
if __name__ == "__main__":
    D_MODEL = 128
    BATCH_SIZE = 16
    torch.manual_seed(1234)

    # 테스트 입력
    x = torch.randn(BATCH_SIZE, D_MODEL, dtype=torch.float32) * 5

    # 1. 기준 모델 (nn.LayerNorm, full-precision)
    ref_ln = nn.LayerNorm(D_MODEL)

    # 2. INT8 근사 모델 (저희 방식 적용)
    approx_ln = Int8LayerNorm(D_MODEL)

    # 공정한 비교를 위해 두 모델이 동일한 gamma, beta 값을 갖도록 설정
    with torch.no_grad():
        approx_ln.gamma.copy_(ref_ln.weight)
        approx_ln.beta.copy_(ref_ln.bias)

    # 각 모델의 출력 계산
    y_ref = ref_ln(x)
    y_approx = approx_ln(x)

    # 두 출력 간의 QSNR 계산
    qsnr = calculate_qsnr(y_ref, y_approx)

    print("--- nn.LayerNorm vs. Int8LayerNorm QSNR 비교 ---")
    print(f"원본 (nn.LayerNorm) 대비 근사 모델의 QSNR: {qsnr:.2f} dB")