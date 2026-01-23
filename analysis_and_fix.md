# Phân Tích Loss Calculation & Background Penalty

## 1. Vấn Đề Hiện Tại

### 1.1 Cách YOLO Tính Loss Hiện Tại

Trong `ultralytics/utils/loss.py`, class `v8DetectionLoss` tính loss như sau:

```python
# Line 425 - Classification Loss
loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
```

**Vấn đề:**
- `target_scores`: Tensor shape `(B, num_anchors, num_classes)` với giá trị từ TAL (Task-Aligned Learning)
- Khi **không có object** (background), `target_scores = 0` cho tất cả classes
- Model dự đoán `pred_scores` (logits) → sau sigmoid → xác suất
- **BCE Loss** chỉ phạt khi model dự đoán **SAI**, nhưng:
  - Nếu model dự đoán `background = high` (tất cả classes = low) → **KHÔNG BỊ PHẠT NẶNG**
  - Nếu model dự đoán `class_i = high` nhưng GT là `background` → Bị phạt **NHƯNG** không đủ mạnh khi có nhiều classes

### 1.2 Tại Sao Cần Background Penalty?

Trong **Semi-Supervised Learning** với **Pseudo-Labels**:

1. **Teacher model** có thể tạo pseudo-labels **SAI** (ví dụ: một vùng background nhưng teacher dự đoán là class)
2. **Student model** học từ cả labeled data (chính xác) + pseudo-labeled data (có noise)
3. Nếu pseudo-labels có nhiều **False Positives** (background nhưng được label là object):
   - Student sẽ học **overconfident** trên background regions
   - Model sẽ bị **bias** về phía dự đoán objects thay vì background

**Giải pháp:** Thêm **Background Penalty** để:
- Phạt nặng hơn khi model dự đoán **bất kỳ class nào** trên vùng background
- Khuyến khích model **conservative hơn** với pseudo-labels
- Giảm False Positives trong quá trình semi-supervised training

---

## 2. Mathematical Formulation

### 2.1 BCE Loss Hiện Tại

Cho anchor point $i$ với prediction $\hat{y}_i \in \mathbb{R}^C$ (C classes) và target $y_i \in [0,1]^C$:

$$
\mathcal{L}_{\text{BCE}}^i = -\frac{1}{C} \sum_{c=1}^C \left[ y_i^c \log(\sigma(\hat{y}_i^c)) + (1-y_i^c) \log(1-\sigma(\hat{y}_i^c)) \right]
$$

Với $\sigma(x) = \frac{1}{1+e^{-x}}$ là sigmoid function.

**Vấn đề:** Khi $y_i = \mathbf{0}$ (background), loss chỉ phạt từng class riêng lẻ:

$$
\mathcal{L}_{\text{BCE}}^i = -\frac{1}{C} \sum_{c=1}^C \log(1-\sigma(\hat{y}_i^c))
$$

Nếu model dự đoán **tất cả** classes với xác suất thấp (ví dụ: $\sigma(\hat{y}_i^c) = 0.1 \ \forall c$), loss = nhỏ.

### 2.2 Background Penalty (Đề Xuất)

Thêm penalty dựa trên **maximum class score** cho background anchors:

$$
\mathcal{L}_{\text{bg-penalty}}^i = \lambda_{\text{bg}} \cdot \max_{c=1}^C \sigma(\hat{y}_i^c)
$$

**Intuition:**
- Phạt dựa trên **class có score cao nhất**
- Nếu model "tự tin" dự đoán **bất kỳ class nào** trên background → bị phạt nặng
- $\lambda_{\text{bg}}$: trọng số penalty (hyperparameter)

### 2.3 Total Classification Loss (Cải Tiến)

$$
\mathcal{L}_{\text{cls}} = \frac{1}{N_{\text{pos}} + N_{\text{bg}}} \left[ \sum_{i \in \text{pos}} \mathcal{L}_{\text{BCE}}^i + \sum_{i \in \text{bg}} \left( \mathcal{L}_{\text{BCE}}^i + \mathcal{L}_{\text{bg-penalty}}^i \right) \right]
$$

Với:
- $N_{\text{pos}}$: số positive anchors (có object)
- $N_{\text{bg}}$: số background anchors (không object)

---

## 3. Implementation Strategy

### 3.1 Modify `v8DetectionLoss`

Tạo custom loss class kế thừa từ `v8DetectionLoss`:

```python
class v8DetectionLossWithBgPenalty(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None, lambda_bg=1.0):
        super().__init__(model, tal_topk, tal_topk2)
        self.lambda_bg = lambda_bg  # Background penalty weight
        
    def compute_cls_loss_with_bg_penalty(
        self, 
        pred_scores,      # (B, num_anchors, num_classes)
        target_scores,    # (B, num_anchors, num_classes)
        target_scores_sum # scalar
    ):
        """
        Compute classification loss with background penalty.
        
        Args:
            pred_scores: Predicted class scores (logits), shape (B, H*W, C)
            target_scores: Target score from TAL, shape (B, H*W, C)
            target_scores_sum: Sum of target_scores for normalization
            
        Returns:
            loss_cls: Classification loss with background penalty
        """
        # Standard BCE loss
        bce_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype))  # (B, H*W, C)
        
        # Identify background anchors: sum(target_scores) == 0 for all classes
        is_background = target_scores.sum(dim=-1) == 0  # (B, H*W)
        
        # Background penalty: penalize max class score on background
        if is_background.any():
            # Sigmoid of predictions
            pred_probs = pred_scores.sigmoid()  # (B, H*W, C)
            
            # Max class probability for each anchor
            max_class_prob, _ = pred_probs.max(dim=-1)  # (B, H*W)
            
            # Apply penalty only on background anchors
            bg_penalty = (max_class_prob * is_background).sum()
            
            # Weighted by lambda_bg
            bg_penalty = self.lambda_bg * bg_penalty / max(is_background.sum(), 1)
        else:
            bg_penalty = torch.tensor(0.0, device=pred_scores.device)
        
        # Total classification loss
        total_cls_loss = bce_loss.sum() / target_scores_sum + bg_penalty
        
        return total_cls_loss, bg_penalty
```

### 3.2 Modify `get_assigned_targets_and_loss`

Override method để sử dụng custom cls loss:

```python
def get_assigned_targets_and_loss(self, preds, batch):
    """Override to use custom classification loss with background penalty."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    pred_distri, pred_scores = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
    )
    # ... (same as original until cls loss computation)
    
    # Instead of:
    # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
    
    # Use custom loss:
    loss[1], bg_penalty = self.compute_cls_loss_with_bg_penalty(
        pred_scores, target_scores, target_scores_sum
    )
    
    # ... (rest remains same)
    return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()
```

### 3.3 Integration với Semi-Trainer

Trong `semi_trainer.py`, chỉnh loss initialization:

```python
# Trong _setup_train() hoặc __init__()
from path.to.custom_loss import v8DetectionLossWithBgPenalty

# Thay thế default loss
model.criterion = v8DetectionLossWithBgPenalty(
    model, 
    tal_topk=10,
    lambda_bg=self.semi_cfg.get('lambda_bg', 1.0)  # From config
)
```

---

## 4. Hyperparameter Tuning

### 4.1 $\lambda_{\text{bg}}$ Values

| $\lambda_{\text{bg}}$ | Behavior                                  | Use Case                     |
|-----------------------|-------------------------------------------|------------------------------|
| 0.0                   | No penalty (standard BCE)                 | Pure supervised learning     |
| 0.5 - 1.0             | **Moderate penalty** (recommended start)  | Semi-supervised with filters |
| 1.0 - 2.0             | **Strong penalty**                        | Noisy pseudo-labels          |
| > 2.0                 | Very aggressive (risk missing objects)    | Very high FPR in pseudo-labels|

### 4.2 Adaptive $\lambda_{\text{bg}}$

```python
def _get_lambda_bg(self, epoch: int) -> float:
    """Adaptive background penalty weight."""
    if epoch < self.burn_in_epochs:
        return 0.0  # No penalty during burn-in
    
    # Increase penalty gradually during semi-supervised phase
    effective_epoch = epoch - self.burn_in_epochs
    max_lambda = self.semi_cfg.get('lambda_bg_max', 1.5)
    
    if effective_epoch < 5:  # Warmup 5 epochs
        return max_lambda * (effective_epoch / 5)
    
    return max_lambda
```

---

## 5. Experimental Validation

### 5.1 Metrics to Monitor

1. **Precision/Recall trên Validation Set:**
   - Precision ↑ → ít False Positives hơn
   - Recall không giảm đáng kể → không miss objects

2. **Pseudo-Label Statistics:**
   - Số lượng pseudo-labels sau filter (không giảm quá nhiều)
   - Confidence distribution (không bị suppress quá mức)

3. **Loss Components:**
   ```python
   LOGGER.info(f'Epoch {epoch}: BCE={bce_loss:.4f}, BG_Penalty={bg_penalty:.4f}')
   ```

### 5.2 Ablation Study

| Experiment | $\lambda_{\text{bg}}$ | Filters      | Expected Outcome              |
|------------|-----------------------|--------------|-------------------------------|
| Baseline   | 0.0                   | DSAT + DFL   | Current performance           |
| Exp-1      | 0.5                   | DSAT + DFL   | ↑ Precision, ~ Recall         |
| Exp-2      | 1.0                   | DSAT + DFL   | ↑↑ Precision, ↓ Recall (nhẹ)  |
| Exp-3      | 1.5                   | DSAT + DFL   | ↑↑↑ Precision, ↓↓ Recall      |

---

## 6. Alternative Approaches

### 6.1 Focal Loss for Background

Thay vì max penalty, dùng **Focal Loss** trên background:

$$
\mathcal{L}_{\text{focal-bg}}^i = -\sum_{c=1}^C (1 - \sigma(\hat{y}_i^c))^\gamma \log(1 - \sigma(\hat{y}_i^c))
$$

**Advantage:** Tự động focus vào "hard negatives" (background được model dự đoán cao).

### 6.2 Objectness Branch

Thêm một **objectness score** riêng (như YOLOv3):

$$
\mathcal{L}_{\text{obj}} = \text{BCE}(\text{objectness\_pred}, \text{is\_object\_target})
$$

Objectness = 0 cho background, 1 cho có object.

---

## 7. Numerical Stability Considerations

### 7.1 Gradient Clipping

Khi $\lambda_{\text{bg}}$ lớn, gradients có thể explode:

```python
# In training loop
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
```

### 7.2 Sigmoid Saturation

Với $\hat{y}_i^c$ rất lớn hoặc nhỏ, $\sigma(\hat{y}_i^c) \approx 1$ hoặc $0$ → gradients vanish.

**Solution:** Sử dụng `torch.clamp` trên logits:

```python
pred_scores_clamped = pred_scores.clamp(min=-10, max=10)
```

---

## 8. Recommended Configuration

```yaml
# trong config/semi_config.yaml
semi:
  burn_in: 5
  lambda_unsup: 1.0
  lambda_bg: 1.0          # Background penalty weight
  lambda_bg_warmup: 5      # Warmup epochs for bg penalty
  lambda_bg_schedule: 'linear'  # 'linear' or 'step'
  
  filters:
    - name: dsat
      params:
        num_classes: 7
        init_threshold: 0.5
    - name: dfl_entropy
      params:
        threshold: 0.5
```

---

## 9. References

1. **Focal Loss for Dense Object Detection** (Lin et al., 2017)
   - https://arxiv.org/abs/1708.02002
   
2. **Unbiased Teacher for Semi-Supervised Object Detection** (Liu et al., 2021)
   - Discusses class imbalance in pseudo-labels
   - https://arxiv.org/abs/2102.09480

3. **Task-Aligned One-stage Object Detection** (Feng et al., 2022)
   - YOLOv8's TAL assignment
   - https://arxiv.org/abs/2108.07755

---

## 10. Complexity Analysis

### Time Complexity

- **Original BCE:** $O(B \times N \times C)$ với $B$ = batch, $N$ = anchors, $C$ = classes
- **With BG Penalty:** $O(B \times N \times C) + O(B \times N)$ (max operation)
- **Overhead:** Negligible (~5% increase)

### Space Complexity

- Additional tensors: `is_background`, `max_class_prob` → $O(B \times N)$
- **Total:** Same asymptotic complexity

---

## 11. Testing Protocol

```python
def test_bg_penalty_loss():
    # Setup
    B, N, C = 2, 100, 7
    pred_scores = torch.randn(B, N, C)
    
    # Case 1: All background (target_scores = 0)
    target_scores_bg = torch.zeros(B, N, C)
    loss_bg, penalty = compute_cls_loss_with_bg_penalty(pred_scores, target_scores_bg, 1.0)
    
    assert penalty > 0, "Background penalty should be positive for all-bg batch"
    
    # Case 2: All foreground (target_scores > 0)
    target_scores_fg = torch.rand(B, N, C)
    loss_fg, penalty = compute_cls_loss_with_bg_penalty(pred_scores, target_scores_fg, target_scores_fg.sum())
    
    assert penalty == 0 or penalty.item() < 0.01, "Background penalty should be near-zero for all-fg batch"
    
    print("✓ All tests passed!")
```
