import math

# LUT 생성 코드는 이전과 동일
sqrt_lut = [math.isqrt(i * 256 + 128) for i in range(256)]

def sqrt_rounded(d_in: int) -> int:
    """
    정규화+LUT 방식으로 제곱근을 계산하고,
    가장 가까운 정수로 반올림하는 함수
    """
    # 0에 대한 예외 처리
    if d_in == 0:
        return 0
    
    # 1. 이전과 동일한 방식으로 내림(floor) 값 계산
    msb_pos = d_in.bit_length() - 1
    k = msb_pos // 2
    norm_shift = (7 - k) * 2
    d_norm = d_in << norm_shift
    lut_addr = (d_norm >> 8) & 0xFF
    sqrt_mantissa = sqrt_lut[lut_addr]
    q_floor = sqrt_mantissa >> (7 - k)
    
    # 2. 반올림 경계값과 비교하여 최종 결과 결정
    # 경계값 = Q^2 + Q
    boundary = q_floor * q_floor + q_floor
    
    if d_in > boundary:
        return q_floor + 1  # 올림
    else:
        return q_floor      # 버림


def sqrt_no_rounded(d_in: int) -> int:
    """
    정규화+LUT 방식으로 제곱근을 계산하고,
    가장 가까운 정수로 반올림하는 함수
    """
    # 0에 대한 예외 처리
    if d_in == 0:
        return 0
    
    # 1. 이전과 동일한 방식으로 내림(floor) 값 계산
    msb_pos = d_in.bit_length() - 1
    k = msb_pos // 2
    norm_shift = (7 - k) * 2
    d_norm = d_in << norm_shift
    lut_addr = (d_norm >> 8) & 0xFF
    sqrt_mantissa = sqrt_lut[lut_addr]
    q_floor = sqrt_mantissa >> (7 - k)
    
    return q_floor
    

# --- 테스트 ---
if __name__ == "__main__":

    test_values = [8, 15, 101, 115] 
    
    print("--- 반올림 제곱근 알고리즘 테스트 ---")
    for val in test_values:
        custom_sqrt = sqrt_rounded(val)
        actual_rounded_sqrt = int(val**0.5 + 0.5)
        
        status = "✅ OK" if custom_sqrt == actual_rounded_sqrt else "❌ Fail"
        
        print(f"Input: {val:>3} | Algo Result: {custom_sqrt:>2} | Actual rounded: {actual_rounded_sqrt:>2} | Status: {status}")