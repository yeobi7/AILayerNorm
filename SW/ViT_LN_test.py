import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader
from datasets import load_dataset # torchvision 대신 datasets 라이브러리 사용
from transformers import ViTFeatureExtractor, ViTForImageClassification
from tqdm import tqdm

# --- 헬퍼 함수 및 사용자 정의 클래스 (이전 코드와 동일) ---
# sqrt_rounded, round_ste, ImprovedAILayerNorm, ARMSNorm 클래스 정의...
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
class ImprovedAILayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta  = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        sq = torch.arange(16, dtype=torch.float32)
        self.register_buffer('square_lut', sq * sq)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[-1]
        scale_in = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_int = round_ste.apply(x / scale_in).clamp(-127.0, 127.0)
        x_q = x_int * scale_in
        Ex = x_q.sum(dim=-1, keepdim=True)
        X_abs = x_int.abs().long()
        H = X_abs >> 4
        L = X_abs & 0x0F
        H_sq = self.square_lut[H]
        L_sq = self.square_lut[L]
        H_x_L = H * L
        x_squared_int = H_sq * 256.0 + H_x_L * 32.0 + L_sq
        x_squared_q = x_squared_int * (scale_in ** 2)
        Ex2 = x_squared_q.sum(dim=-1, keepdim=True)
        mu = Ex / N
        var = (Ex2 / N - mu*mu).clamp(min=0)
        var_int = round_ste.apply(var).clamp(1, 2**16-1).long()
        std_int = torch.tensor([sqrt_rounded(v.item()) for v in var_int.flatten()], device=var.device, dtype=torch.float32).view_as(var_int)
        inv_std = 1.0 / std_int.clamp(min=self.eps)
        x_norm = (x - mu) * inv_std
        y = x_norm * self.gamma + self.beta
        scale_out = (y.abs().max() / 127.0).clamp(min=1e-8)
        y_int = round_ste.apply(y / scale_out).clamp(-127.0, 127.0)
        y_q = y_int * scale_out
        return y_q
class ARMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True, bias=False):
        super().__init__()
        self.d = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
        self.eps, self.affine, self.use_bias = eps, affine, bias
        if affine: self.weight = nn.Parameter(torch.ones(self.d))
        else: self.register_parameter('weight', None)
        if self.use_bias: self.bias = nn.Parameter(torch.zeros(self.d))
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

# --- 모델 수정 및 평가를 위한 유틸리티 함수 ---
def replace_layernorm(module, new_layer_class):
    """모듈 내의 모든 nn.LayerNorm을 새로운 클래스로 재귀적으로 교체 (수정됨)"""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            init_args = {
                'normalized_shape': child.normalized_shape,
                'eps': child.eps
            }
            # 클래스별로 다른 init 인자 이름을 처리
            if new_layer_class == ARMSNorm:
                init_args['affine'] = child.elementwise_affine
            else:
                init_args['elementwise_affine'] = child.elementwise_affine
            
            new_module = new_layer_class(**init_args).to(child.weight.device)
            
            if child.elementwise_affine:
                if hasattr(new_module, 'gamma'):
                    new_module.gamma.data.copy_(child.weight.data)
                    new_module.beta.data.copy_(child.bias.data)
                elif hasattr(new_module, 'weight'):
                    new_module.weight.data.copy_(child.weight.data)
            setattr(module, name, new_module)
        else:
            replace_layernorm(child, new_layer_class)

def evaluate(model, dataloader, device):
    """모델의 정확도 평가"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- 메인 실행 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/vit-base-patch16-224"
    
    # 1. 데이터셋 준비 (ImageNet-1k 검증 셋의 일부)
    print("--- Loading ImageNet-1k validation split (subset) ---")
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    # 전체를 다운받지 않고 스트리밍 모드로 1000개만 사용
    val_dataset = load_dataset("imagenet-1k", split='validation', streaming=True).take(1000)

    # .map() 메서드를 사용하여 스트리밍 데이터에 직접 변환 함수를 적용
    def transform(batch):
        # Feature extractor를 각 이미지에 적용
        inputs = feature_extractor([img.convert("RGB") for img in batch['image']], return_tensors='pt')
        # label도 함께 반환
        inputs['label'] = batch['label']
        return inputs
    
    # map을 사용해 변환된 데이터셋을 새로 생성
    transformed_dataset = val_dataset.map(transform, batched=True)
    transformed_dataset = transformed_dataset.remove_columns(['image'])

    # DataLoader는 변환이 완료된 데이터셋을 사용
    dataloader = DataLoader(transformed_dataset, batch_size=32)

    print("--- Dataset ready ---\n")

    # ImageNet은 1000개의 클래스를 가짐
    NUM_LABELS = 1000

    # 2. 기준 모델(Original ViT) 정확도 측정
    print("--- 1. Evaluating Original ViT with nn.LayerNorm ---")
    original_model = ViTForImageClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    original_model.to(device)
    original_accuracy = evaluate(original_model, dataloader, device)
    print(f"\nOriginal ViT Accuracy: {original_accuracy:.2f}%\n")

    # 3. ImprovedAILayerNorm 적용 모델 정확도 측정
    print("--- 2. Evaluating ViT with ImprovedAILayerNorm ---")
    ai_layernorm_model = ViTForImageClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    replace_layernorm(ai_layernorm_model, ImprovedAILayerNorm)
    ai_layernorm_model.to(device)
    ai_layernorm_accuracy = evaluate(ai_layernorm_model, dataloader, device)
    print(f"\nViT + ImprovedAILayerNorm Accuracy: {ai_layernorm_accuracy:.2f}%\n")
    
    # 4. ARMSNorm 적용 모델 정확도 측정
    print("--- 3. Evaluating ViT with ARMSNorm ---")
    armsnorm_model = ViTForImageClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    replace_layernorm(armsnorm_model, ARMSNorm)
    armsnorm_model.to(device)
    armsnorm_accuracy = evaluate(armsnorm_model, dataloader, device)
    print(f"\nViT + ARMSNorm Accuracy: {armsnorm_accuracy:.2f}%\n")

    # --- 최종 결과 요약 ---
    print("="*20, " 최종 정확도 비교 (ImageNet-1k val subset) ", "="*20)
    print(f"Original ViT (nn.LayerNorm)  : {original_accuracy:.2f}%")
    print(f"ViT + ImprovedAILayerNorm    : {ai_layernorm_accuracy:.2f}% (Accuracy Drop: {original_accuracy - ai_layernorm_accuracy:.2f}%)")
    print(f"ViT + ARMSNorm               : {armsnorm_accuracy:.2f}% (Accuracy Drop: {original_accuracy - armsnorm_accuracy:.2f}%)")
    print("="*65)