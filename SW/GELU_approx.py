import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import matplotlib.pyplot as plt


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


# — STE wrappers —
class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class round_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# — Single-Module 구현 —
class QuantApproxGELU(nn.Module):
    def __init__(self):
        super().__init__()
        # 근사 GELU 파라미터
        self.a, self.b, self.delta1 = -0.2888, -1.769, 1.0
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # — 1) Fake-quantize INPUT (linear scale) —
        scale_in = (x.abs().max() / 127.0).to(x.dtype).to(x.device)
        x_int    = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q      = x_int * scale_in

        # — 2) Approximate GELU on quantized input —
        z      = x_q * self.inv_sqrt2
        x_clip = torch.clamp(torch.abs(z), max=-self.b)
        l_erf  = torch.sign(z) * self.delta1 * (self.a * (x_clip + self.b)**2 + 1)
        y      = x_q * 0.5 * (1 + l_erf)

        # — 3) Fake-quantize OUTPUT (linear scale) —
        scale_out = (y.abs().max() / 127.0).to(y.dtype).to(y.device)
        y_int     = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q       = y_int * scale_out

        return y_q

# — 검증 (CPU sweep + QSNR + backprop 확인) —
if __name__ == "__main__":
    torch.manual_seed(1234)
    x = torch.randn(1, 128, 512, 512, dtype=torch.float32, requires_grad=True) * 10
#    x = torch.randn(1, 128, 512, 512, dtype=torch.float32, device='cuda', requires_grad=True) * 10
    act  = QuantApproxGELU()
    y_q  = act(x)                     # forward(x) → x_int * scale
    y_fp = F.gelu(x)                  # reference
    
    
    # QSNR 계산
    qsnr = calculate_qsnr(y_fp, y_q)
    print(f"QSNR: {qsnr:.2f} dB")


#     # radient 흐름 확인 (STE)
#     (y_q.sum()).backward()
#     print("x.grad shape:", x.grad.shape)
#     print("x.grad sample:", x.grad.view(-1)[:5])

