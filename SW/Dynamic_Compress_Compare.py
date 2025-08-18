import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# =============================================================================
# Baseline: 단순히 x^2 (절대값 제곱) 계산
# =============================================================================
def baseline_square(x):
    return int(abs(x)) ** 2

# =============================================================================
# LUT 사전 (0~15에 대해)
# =============================================================================
lut_square = {0:0, 1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64, 9:81, 10:100, 11:121, 12:144, 13:169, 14:196, 15:225}

# =============================================================================
# Method 1: dynamic_compress() 기반 제곱 근사
# =============================================================================
def dynamic_compress(x):
    x = int(round(x))
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

def approx_square_method1(x):
    Xc, s = dynamic_compress(abs(x))
    return lut_square[Xc] << (4 * s + 3)   # s=0이면 <<0, s=1이면 <<4


# =============================================================================
# Method 2: dynamic_compress_D() 기반 제곱 근사
# =============================================================================
def dynamic_compress_D(x):
    x = int(round(x))
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

def approx_square_method2(x):
    Xc, s = dynamic_compress_D(abs(x))
    return lut_square[Xc] << (4 * s)   # s=1이면 <<4, s=2이면 <<8

# =============================================================================
# 테스트를 위한 입력 생성: 0부터 127까지 FP32
# =============================================================================
selected_input = torch.arange(0, 128, dtype=torch.float32)  # shape: (128,)

# (여기서는 gamma, beta는 사용하지 않으므로 dummy로 1과 0으로 설정)
selected_gamma = torch.ones_like(selected_input)
selected_beta  = torch.zeros_like(selected_input)

# 전체 입력을 numpy 배열로 변환 (FP32)
X_full_a = selected_input.cpu().numpy()  # FP32 값, 0~127

# 배치 사이즈 8
batches = X_full_a.shape[0] // 8  # 128 // 8 = 16 배치

results_comparison = {}  # 배치별 결과 저장

for i in range(batches):
    # 배치별 FP32 입력 (8개)
    X_batch = X_full_a[i*8:(i+1)*8]
    baseline_vals = []
    method1_vals = []
    method2_vals = []
    for x in X_batch:
        bsq = baseline_square(x)
        m1 = approx_square_method1(x)
        m2 = approx_square_method2(x)
        baseline_vals.append(bsq)
        method1_vals.append(m1)
        method2_vals.append(m2)
    baseline_vals = np.array(baseline_vals, dtype=np.float32)
    method1_vals = np.array(method1_vals, dtype=np.float32)
    method2_vals = np.array(method2_vals, dtype=np.float32)
    
    mae_m1 = np.mean(np.abs(baseline_vals - method1_vals))
    mse_m1 = np.mean((baseline_vals - method1_vals)**2)
    mae_m2 = np.mean(np.abs(baseline_vals - method2_vals))
    mse_m2 = np.mean((baseline_vals - method2_vals)**2)
    
    # Accuracy: 기준을 배치 내 최대 baseline 값으로 계산 (RMSE 기준을 사용하려면 sqrt(mse) 사용 가능)
    max_baseline = baseline_vals.max() if baseline_vals.max() != 0 else 1
    # 여기서는 MAE를 기준으로 Accuracy 계산: (1 - MAE / max_baseline)*100
    acc_m1 = (1 - (mae_m1 / max_baseline)) * 100
    acc_m2 = (1 - (mae_m2 / max_baseline)) * 100
    
    results_comparison[f"Batch {i+1}"] = {
        "Input FP32": X_batch,
        "Baseline (x^2)": baseline_vals,
        "Method1 Approx": method1_vals,
        "Method2 Approx": method2_vals,
        "MAE Method1": mae_m1,
        "MSE Method1": mse_m1,
        "Accuracy Method1 (%)": acc_m1,
        "MAE Method2": mae_m2,
        "MSE Method2": mse_m2,
        "Accuracy Method2 (%)": acc_m2
    }

# 배치별 결과 출력
for batch_name, info in results_comparison.items():
    print(f"\n{batch_name}")
    print("Input X (FP32):", info["Input FP32"])
    print("Baseline x^2:", info["Baseline (x^2)"])
    print("Method1 Approximation:", info["Method1 Approx"])
    print("Method2 Approximation:", info["Method2 Approx"])
    print("MAE Method1:", info["MAE Method1"])
    print("MSE Method1:", info["MSE Method1"])
    print("Accuracy Method1 (%):", info["Accuracy Method1 (%)"])
    print("MAE Method2:", info["MAE Method2"])
    print("MSE Method2:", info["MSE Method2"])
    print("Accuracy Method2 (%):", info["Accuracy Method2 (%)"])

# 전체 배치 평균 계산
all_mae_m1 = np.mean([info["MAE Method1"] for info in results_comparison.values()])
all_mse_m1 = np.mean([info["MSE Method1"] for info in results_comparison.values()])
all_acc_m1 = np.mean([info["Accuracy Method1 (%)"] for info in results_comparison.values()])

all_mae_m2 = np.mean([info["MAE Method2"] for info in results_comparison.values()])
all_mse_m2 = np.mean([info["MSE Method2"] for info in results_comparison.values()])
all_acc_m2 = np.mean([info["Accuracy Method2 (%)"] for info in results_comparison.values()])

print("\n==== Overall Comparison Metrics ====")
print(f"Method1 -> MAE: {all_mae_m1:.2f}, MSE: {all_mse_m1:.2f}, Accuracy: {all_acc_m1:.2f}%")
print(f"Method2 -> MAE: {all_mae_m2:.2f}, MSE: {all_mse_m2:.2f}, Accuracy: {all_acc_m2:.2f}%")


# 전체 입력: 0부터 127까지
x_vals = np.arange(0, 128, dtype=np.float32)

# Baseline: 정확한 제곱 값
baseline_vals = np.array([baseline_square(x) for x in x_vals], dtype=np.float32)

# Method 1: dynamic_compress() 기반 근사
method1_vals = np.array([approx_square_method1(x) for x in x_vals], dtype=np.float32)

# Method 2: dynamic_compress_D() 기반 근사
method2_vals = np.array([approx_square_method2(x) for x in x_vals], dtype=np.float32)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, baseline_vals, 'k-', linewidth=2, label='Baseline (x^2)')
plt.plot(x_vals, method1_vals, 'b--', linewidth=2, label='AILayerNorm Method')
plt.plot(x_vals, method2_vals, 'r-.', linewidth=2, label='AIRMSNorm Method')
plt.xlabel("Input x ")
plt.ylabel("Squared Value")
#plt.title("Comparison of Baseline vs. Approximations")
plt.legend()
plt.grid(True)
plt.show()
