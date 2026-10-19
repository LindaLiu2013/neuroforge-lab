# 真实LLM感知校准报告

**模型**: qwen3.8-27b
**时间**: 2026-10-18T15:43:20.321348
**测试规模**: 5个场景 × 3次重复

## 核心发现

1. **检出率**: 52.0%（模拟器假设~42%）
   → 与模拟器假设接近

2. **误报率**: 27.1%（模拟器假设~8%）
   → 高于模拟器假设

3. **慢变量检出率**: 60.0%（模拟器假设~30%）
   → 需要调整模拟器参数

4. **平均置信度**: 81.1%
   → 存在过度自信

5. **严重度偏差**: +0.018
   → 严重度判断较准确

6. **JSON解析失败**: 0次

## 对模拟器的校准建议

- 调整慢变量检出率：当前模拟器假设30%，实测60%
- 调高误报率：当前模拟器假设8%，实测27%
- 确认置信度膨胀系数：真实LLM确实表现出过度自信

## 逐场景详情

### S1_salience_mix: Mixed urgency: 1 dramatic + 2 slow drift issues
验证假设: H1+H2
  第1次: 检出6/4, TP=3, FP=3, 漏检=1
    equipment_fault: 真实严重度=0.85, 报告=0.85, 置信度=0.95
    cert_missing: 真实严重度=0.60, 报告=0.90, 置信度=0.95
    sample_missing: 真实严重度=0.50, 报告=0.85, 置信度=0.95
  第2次: 检出6/4, TP=4, FP=2, 漏检=0
    equipment_fault: 真实严重度=0.85, 报告=0.85, 置信度=0.95
    cert_missing: 真实严重度=0.60, 报告=0.90, 置信度=0.98
    fifo_violation: 真实严重度=0.45, 报告=0.60, 置信度=0.65
    sample_missing: 真实严重度=0.50, 报告=0.85, 置信度=0.98
  第3次: 检出5/4, TP=3, FP=2, 漏检=1
    equipment_fault: 真实严重度=0.85, 报告=0.85, 置信度=0.95
    cert_missing: 真实严重度=0.60, 报告=0.90, 置信度=0.95
    sample_missing: 真实严重度=0.50, 报告=0.80, 置信度=0.95

### S2_all_clear: No real issues — test false positive rate
验证假设: H3
  第1次: 检出0/0, TP=0, FP=0, 漏检=0
  第2次: 检出0/0, TP=0, FP=0, 漏检=0
  第3次: 检出0/0, TP=0, FP=0, 漏检=0

### S3_slow_drift_only: Only slow-drift issues, no dramatic events
验证假设: H2
  第1次: 检出2/5, TP=2, FP=0, 漏检=3
    record_gap: 真实严重度=0.30, 报告=0.45, 置信度=0.85
    fifo_violation: 真实严重度=0.35, 报告=0.40, 置信度=0.50
  第2次: 检出3/5, TP=3, FP=0, 漏检=2
    record_gap: 真实严重度=0.30, 报告=0.55, 置信度=0.90
    fifo_violation: 真实严重度=0.35, 报告=0.50, 置信度=0.60
    fatigue_trend: 真实严重度=0.55, 报告=0.40, 置信度=0.60
  第3次: 检出4/5, TP=4, FP=0, 漏检=1
    record_gap: 真实严重度=0.30, 报告=0.55, 置信度=0.90
    fifo_violation: 真实严重度=0.35, 报告=0.45, 置信度=0.60
    fatigue_trend: 真实严重度=0.55, 报告=0.35, 置信度=0.60
    compliance_erosion: 真实严重度=0.50, 报告=0.40, 置信度=0.50

### S4_contradictory_signals: Conflicting data — test narrative consistency
验证假设: H6
  第1次: 检出4/3, TP=2, FP=2, 漏检=1
    temp_violation: 真实严重度=0.70, 报告=0.65, 置信度=0.85
    cross_contamination: 真实严重度=0.50, 报告=0.50, 置信度=0.80
  第2次: 检出4/3, TP=2, FP=2, 漏检=1
    temp_violation: 真实严重度=0.70, 报告=0.70, 置信度=0.90
    cross_contamination: 真实严重度=0.50, 报告=0.50, 置信度=0.85
  第3次: 检出3/3, TP=2, FP=1, 漏检=1
    temp_violation: 真实严重度=0.70, 报告=0.60, 置信度=0.85
    cross_contamination: 真实严重度=0.50, 报告=0.50, 置信度=0.70

### S5_severity_calibration: Issues at different severity levels — test distortion
验证假设: H5
  第1次: 检出2/2, TP=1, FP=1, 漏检=0
    equipment_fault: 真实严重度=0.90, 报告=0.20, 置信度=0.90
  第2次: 检出2/2, TP=1, FP=1, 漏检=0
    equipment_fault: 真实严重度=0.90, 报告=0.30, 置信度=0.85
  第3次: 检出2/2, TP=1, FP=1, 漏检=0
    equipment_fault: 真实严重度=0.90, 报告=0.30, 置信度=0.70
