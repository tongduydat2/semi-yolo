# Công Thức Toán Học - Loss Function Semi-Supervised YOLO

## Notation (Ký Hiệu)

### Batch và Data
- $B$: Batch size
- $N$: Số lượng anchor points (tổng qua tất cả scale levels)
- $C$: Số lượng classes
- $H_l, W_l$: Height và width của feature map ở level $l$
- $S = \{8, 16, 32\}$: Tập stride values (3 detection heads)

### Predictions và Targets
- $\hat{\mathbf{y}}_{ij} \in \mathbb{R}^C$: Predicted class logits cho anchor $j$ trong image $i$
- $\hat{\mathbf{b}}_{ij} \in \mathbb{R}^4$: Predicted bounding box (xyxy format)
- $\hat{\mathbf{d}}_{ij} \in \mathbb{R}^{64}$: Predicted distribution (DFL, reg_max=16)
- $\mathbf{y}_{ij} \in [0,1]^C$: Target alignment scores từ TAL
- $\mathbf{b}_{ij}^{gt} \in \mathbb{R}^4$: Ground truth bounding box

### Functions
- $\sigma(x) = \frac{1}{1+e^{-x}}$: Sigmoid function
- $\text{IoU}(\hat{\mathbf{b}}, \mathbf{b}^{gt})$: Intersection over Union
- $\text{CIoU}(\hat{\mathbf{b}}, \mathbf{b}^{gt})$: Complete IoU (bao gồm aspect ratio)

---

## 1. YOLO Base Detection Loss

### 1.1 Total Loss

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \cdot \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \cdot \mathcal{L}_{\text{dfl}}
$$

Với hyperparameters mặc định:
- $\lambda_{\text{box}} = 7.5$
- $\lambda_{\text{cls}} = 0.5$
- $\lambda_{\text{dfl}} = 1.5$

---

### 1.2 Classification Loss (BCE)

**Binary Cross-Entropy Loss với TAL (Task-Aligned Learning):**

$$
\mathcal{L}_{\text{cls}} = \frac{1}{\sum_{i,j} \mathbf{y}_{ij}} \sum_{i=1}^B \sum_{j=1}^N \sum_{c=1}^C \mathcal{L}_{\text{BCE}}(\hat{y}_{ij}^c, y_{ij}^c)
$$

Trong đó:

$$
\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -\left[ y \log \sigma(\hat{y}) + (1-y) \log(1-\sigma(\hat{y})) \right]
$$

**Giải thích:**
- $\hat{y}_{ij}^c$: Logit (chưa qua sigmoid) cho class $c$ tại anchor $(i,j)$
- $y_{ij}^c$: Target alignment score từ TAL (range [0,1])
- Normalization: Chia cho tổng $\sum_{i,j} \mathbf{y}_{ij}$ (tổng alignment scores)

**Chi tiết TAL Assignment:**

Target scores $\mathbf{y}_{ij}$ được tính từ:

$$
y_{ij}^c = \begin{cases}
t_{ij}^\alpha \cdot \text{IoU}(\hat{\mathbf{b}}_{ij}, \mathbf{b}_{ij}^{gt})^\beta & \text{if anchor } j \text{ assigned to object} \\
0 & \text{if background}
\end{cases}
$$

Với:
- $t_{ij} = \sigma(\hat{y}_{ij}^{c^*})$: Predicted probability cho ground truth class $c^*$
- $\alpha = 0.5, \beta = 6.0$: TAL hyperparameters
- Assignment dựa trên top-k highest alignment scores

---

### 1.3 Bounding Box Loss (CIoU + DFL)

$$
\mathcal{L}_{\text{box}} = \frac{1}{\sum_{i,j} \mathbf{y}_{ij}} \sum_{(i,j) \in \mathcal{F}} w_{ij} \cdot (1 - \text{CIoU}(\hat{\mathbf{b}}_{ij}, \mathbf{b}_{ij}^{gt}))
$$

Trong đó:
- $\mathcal{F} = \{(i,j) : \text{foreground mask} = 1\}$: Tập foreground anchors
- $w_{ij} = \sum_{c=1}^C y_{ij}^c$: Weight từ alignment scores

**Complete IoU:**

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2(\mathbf{c}, \mathbf{c}^{gt})}{d^2} - \alpha v
$$

Với:
- $\rho(\mathbf{c}, \mathbf{c}^{gt})$: Euclidean distance giữa box centers
- $d$: Diagonal length của smallest enclosing box
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$: Aspect ratio consistency
- $\alpha = \frac{v}{(1-\text{IoU}) + v}$: Trade-off parameter

---

### 1.4 Distribution Focal Loss (DFL)

$$
\mathcal{L}_{\text{dfl}} = \frac{1}{\sum_{i,j} \mathbf{y}_{ij}} \sum_{(i,j) \in \mathcal{F}} w_{ij} \cdot \text{DFL}(\hat{\mathbf{d}}_{ij}, \mathbf{t}_{ij})
$$

**DFL chi tiết:**

Cho mỗi edge (left, top, right, bottom), target $t \in [0, 15]$ (reg_max=16):

$$
\text{DFL}(\hat{\mathbf{d}}, t) = -\left[ (t_r - t) \log p_{t_l} + (t - t_l) \log p_{t_r} \right]
$$

Với:
- $t_l = \lfloor t \rfloor$: Left integer
- $t_r = t_l + 1$: Right integer  
- $p_k = \frac{e^{\hat{d}_k}}{\sum_{m=0}^{15} e^{\hat{d}_m}}$: Softmax probability

**Intuition:** Linear interpolation giữa 2 integer bins gần nhất.

---

## 2. BACKGROUND PENALTY (Custom Addition)

### 2.1 Modified Classification Loss

$$
\mathcal{L}_{\text{cls}}^{\text{new}} = \mathcal{L}_{\text{cls}}^{\text{BCE}} + \mathcal{L}_{\text{bg-penalty}}
$$

### 2.2 Background Penalty Term

**Định nghĩa background anchors:**

$$
\mathcal{B} = \left\{ (i,j) : \sum_{c=1}^C y_{ij}^c < \epsilon \right\}
$$

Với $\epsilon = 10^{-6}$ (numerical threshold).

**Simple Max Penalty (default):**

$$
\mathcal{L}_{\text{bg-penalty}} = \frac{\lambda_{\text{bg}}}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \max_{c=1}^C \sigma(\hat{y}_{ij}^c)
$$

**Focal-style Penalty (optional, use_focal_bg=true):**

$$
\mathcal{L}_{\text{bg-penalty}}^{\text{focal}} = \frac{\lambda_{\text{bg}}}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \sum_{c=1}^C \left[ -(1-p_{ij}^c)^\gamma \log(1-p_{ij}^c) \right]
$$

Với:
- $p_{ij}^c = \sigma(\hat{y}_{ij}^c)$: Predicted probability
- $\gamma = 2.0$: Focal parameter
- $\lambda_{\text{bg}}$: Background penalty weight (hyperparameter)

---

### 2.3 Gradient Flow của Background Penalty

**Đạo hàm theo logits $\hat{y}_{ij}^c$ cho anchor $(i,j) \in \mathcal{B}$:**

Với **Simple Max Penalty**, giả sử class $c^* = \arg\max_c \sigma(\hat{y}_{ij}^c)$:

$$
\frac{\partial \mathcal{L}_{\text{bg-penalty}}}{\partial \hat{y}_{ij}^{c^*}} = \frac{\lambda_{\text{bg}}}{|\mathcal{B}|} \cdot \sigma(\hat{y}_{ij}^{c^*}) \cdot (1 - \sigma(\hat{y}_{ij}^{c^*}))
$$

$$
\frac{\partial \mathcal{L}_{\text{bg-penalty}}}{\partial \hat{y}_{ij}^c} = 0 \quad \forall c \neq c^*
$$

**Intuition:** 
- Chỉ class có confidence cao nhất bị phạt
- Gradient giảm khi $\sigma(\hat{y}) \to 0$ hoặc $\sigma(\hat{y}) \to 1$ (sigmoid derivative)
- Mạnh nhất khi $\sigma(\hat{y}) \approx 0.5$

---

## 3. SEMI-SUPERVISED LOSS

### 3.1 Total Training Loss

$$
\mathcal{L}_{\text{semi}} = \mathcal{L}_{\text{sup}} + \lambda_u(t) \cdot \mathcal{L}_{\text{unsup}}
$$

Với:
- $\mathcal{L}_{\text{sup}}$: Supervised loss trên labeled data
- $\mathcal{L}_{\text{unsup}}$: Unsupervised loss trên pseudo-labeled data
- $\lambda_u(t)$: Time-varying unsupervised weight

---

### 3.2 Supervised Loss

$$
\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{total}}(\mathbf{X}_L, \mathbf{Y}_L)
$$

Với:
- $\mathbf{X}_L$: Labeled images (strong augmentation)
- $\mathbf{Y}_L$: Ground truth labels

**Components:**

$$
\mathcal{L}_{\text{sup}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}}^{\text{sup}} + \lambda_{\text{cls}} (\mathcal{L}_{\text{cls}}^{\text{sup}} + \mathcal{L}_{\text{bg}}^{\text{sup}}) + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}^{\text{sup}}
$$

---

### 3.3 Unsupervised Loss (Pseudo-Label)

**Step 1: Teacher generates pseudo-labels**

$$
\tilde{\mathbf{Y}}_U = \text{Teacher-EMA}(\mathbf{X}_U^{\text{weak}})
$$

Với:
- $\mathbf{X}_U^{\text{weak}}$: Unlabeled images với weak augmentation
- Teacher model: Exponential Moving Average của student

**Step 2: Filter pseudo-labels**

$$
\tilde{\mathbf{Y}}_U^{\text{filtered}} = \text{FilterChain}(\tilde{\mathbf{Y}}_U)
$$

FilterChain bao gồm:
1. **DSAT Filter**: Uncertainty-based filtering
2. **DFL Entropy Filter**: Distribution entropy check  
3. **TAL Alignment Filter**: Alignment score threshold

**Step 3: Student loss on filtered pseudo-labels**

$$
\mathcal{L}_{\text{unsup}} = \mathcal{L}_{\text{total}}(\mathbf{X}_U^{\text{strong}}, \tilde{\mathbf{Y}}_U^{\text{filtered}})
$$

Với $\mathbf{X}_U^{\text{strong}}$: Strong augmentation của unlabeled images.

---

### 3.4 Unsupervised Weight Schedule

$$
\lambda_u(t) = \begin{cases}
0 & t < t_{\text{burn-in}} \\
\lambda_u^{\max} \cdot \min\left(1, \frac{t - t_{\text{burn-in}}}{t_{\text{warmup}}}\right) & t_{\text{burn-in}} \leq t < t_{\text{burn-in}} + t_{\text{warmup}} \\
\lambda_u^{\max} & t \geq t_{\text{burn-in}} + t_{\text{warmup}}
\end{cases}
$$

Với:
- $t$: Current epoch
- $t_{\text{burn-in}} = 5$: Burn-in epochs (supervised only)
- $t_{\text{warmup}} = 5$: Warmup epochs  
- $\lambda_u^{\max} = 1.0$: Maximum unsupervised weight

---

### 3.5 Background Penalty Schedule

Tương tự, $\lambda_{\text{bg}}$ cũng có schedule:

$$
\lambda_{\text{bg}}(t) = \begin{cases}
0 & t < t_{\text{burn-in}} \\
\lambda_{\text{bg}}^{\max} \cdot \frac{t - t_{\text{burn-in}}}{t_{\text{bg-warmup}}} & t_{\text{burn-in}} \leq t < t_{\text{burn-in}} + t_{\text{bg-warmup}} \\
\lambda_{\text{bg}}^{\max} & t \geq t_{\text{burn-in}} + t_{\text{bg-warmup}}
\end{cases}
$$

Với $t_{\text{bg-warmup}} = 5$, $\lambda_{\text{bg}}^{\max} = 1.0$.

---

## 4. EMA Teacher Update

$$
\theta_{\text{teacher}}^{(t+1)} = \alpha \cdot \theta_{\text{teacher}}^{(t)} + (1-\alpha) \cdot \theta_{\text{student}}^{(t)}
$$

Với:
- $\alpha = 0.999$: EMA decay rate
- $\theta$: Model parameters

**Momentum form:**

$$
\theta_{\text{teacher}}^{(t+1)} = \theta_{\text{teacher}}^{(t)} + (1-\alpha)(\theta_{\text{student}}^{(t)} - \theta_{\text{teacher}}^{(t)})
$$

---

## 5. COMPLETE FORWARD PASS ANALYSIS

### 5.1 Single Training Iteration

**Input:**
- Batch labeled: $\{\mathbf{X}_L, \mathbf{Y}_L\}$, size $B_L$
- Batch unlabeled weak: $\mathbf{X}_U^{\text{weak}}$, size $B_U$
- Batch unlabeled strong: $\mathbf{X}_U^{\text{strong}}$, size $B_U$

**Forward Pass:**

1. **Student trên labeled data:**
   $$
   \hat{\mathbf{Y}}_L = \text{Student}(\mathbf{X}_L)
   $$
   
2. **Teacher trên unlabeled weak:**
   $$
   \tilde{\mathbf{Y}}_U = \text{Teacher}(\mathbf{X}_U^{\text{weak}})
   $$
   
3. **Filter pseudo-labels:**
   $$
   \tilde{\mathbf{Y}}_U^{\text{filtered}} = \text{FilterChain}(\tilde{\mathbf{Y}}_U)
   $$
   
4. **Student trên unlabeled strong:**
   $$
   \hat{\mathbf{Y}}_U = \text{Student}(\mathbf{X}_U^{\text{strong}})
   $$

**Loss Computation:**

$$
\begin{align}
\mathcal{L}_{\text{sup}} &= \mathcal{L}_{\text{total}}(\hat{\mathbf{Y}}_L, \mathbf{Y}_L) \\
\mathcal{L}_{\text{unsup}} &= \mathcal{L}_{\text{total}}(\hat{\mathbf{Y}}_U, \tilde{\mathbf{Y}}_U^{\text{filtered}}) \\
\mathcal{L} &= \mathcal{L}_{\text{sup}} + \lambda_u(t) \cdot \mathcal{L}_{\text{unsup}}
\end{align}
$$

**Breakdown:**

$$
\begin{align}
\mathcal{L}_{\text{sup}} &= \lambda_{\text{box}} \mathcal{L}_{\text{box}}^{\text{sup}} + \lambda_{\text{cls}} (\mathcal{L}_{\text{BCE}}^{\text{sup}} + \lambda_{\text{bg}}(t) \mathcal{L}_{\text{bg}}^{\text{sup}}) + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}^{\text{sup}} \\
\mathcal{L}_{\text{unsup}} &= \lambda_{\text{box}} \mathcal{L}_{\text{box}}^{\text{unsup}} + \lambda_{\text{cls}} (\mathcal{L}_{\text{BCE}}^{\text{unsup}} + \lambda_{\text{bg}}(t) \mathcal{L}_{\text{bg}}^{\text{unsup}}) + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}^{\text{unsup}}
\end{align}
$$

---

## 6. NUMERICAL STABILITY CONSIDERATIONS

### 6.1 BCE Loss Stability

$$
\log \sigma(\hat{y}) = \log\left(\frac{1}{1+e^{-\hat{y}}}\right) = -\log(1+e^{-\hat{y}})
$$

**Numerically stable implementation:**

$$
\log \sigma(\hat{y}) = \begin{cases}
-\log(1+e^{-\hat{y}}) & \text{if } \hat{y} \geq 0 \\
\hat{y} - \log(1+e^{\hat{y}}) & \text{if } \hat{y} < 0
\end{cases}
$$

Tránh overflow khi $\hat{y}$ rất lớn hoặc rất nhỏ.

### 6.2 Background Penalty Clamping

$$
p_{ij}^c = \sigma(\hat{y}_{ij}^c) \in [\epsilon, 1-\epsilon]
$$

Với $\epsilon = 10^{-7}$ để tránh $\log(0)$.

---

## 7. GRADIENT MAGNITUDE ANALYSIS

### 7.1 Classification Loss Gradient

Cho anchor $(i,j)$ với target $y_{ij}^c$:

$$
\frac{\partial \mathcal{L}_{\text{cls}}}{\partial \hat{y}_{ij}^c} = \frac{1}{\sum y_{ij}} \cdot (\sigma(\hat{y}_{ij}^c) - y_{ij}^c)
$$

**Magnitude:** $|\nabla| \in [0, \frac{1}{\sum y_{ij}}]$

### 7.2 Background Penalty Gradient

$$
\frac{\partial \mathcal{L}_{\text{bg}}}{\partial \hat{y}_{ij}^{c^*}} = \frac{\lambda_{\text{bg}}}{|\mathcal{B}|} \cdot p_{ij}^{c^*}(1-p_{ij}^{c^*})
$$

**Maximum magnitude:** $\frac{\lambda_{\text{bg}}}{4|\mathcal{B}|}$ khi $p = 0.5$

### 7.3 Gradient Balance

**Tỉ lệ gradient contributions:**

$$
\frac{|\nabla \mathcal{L}_{\text{bg}}|}{|\nabla \mathcal{L}_{\text{BCE}}|} \approx \frac{\lambda_{\text{bg}} \sum y_{ij}}{4|\mathcal{B}|}
$$

**Khi $\lambda_{\text{bg}} = 1.0$:**
- Nếu $\sum y_{ij} \approx 100$, $|\mathcal{B}| \approx 6000$
- Tỉ lệ $\approx 0.004$ → Background penalty là perturbation nhỏ

---

## 8. EXPECTED LOSS BEHAVIOR

### 8.1 Burn-in Phase ($t < 5$)

$$
\mathcal{L} = \mathcal{L}_{\text{sup}}^{\text{BCE+box+dfl}}
$$

- Không có unsupervised loss
- Không có background penalty
- Training thuần supervised

### 8.2 Warmup Phase ($5 \leq t < 10$)

$$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_u(t) \mathcal{L}_{\text{unsup}}
$$

Với $\lambda_u(t), \lambda_{\text{bg}}(t)$ tăng tuyến tính.

**Expected trends:**
- $\mathcal{L}_{\text{sup}}$: Giảm (model improving)
- $\mathcal{L}_{\text{unsup}}$: Có thể tăng ban đầu (pseudo-labels noisy)
- $\mathcal{L}_{\text{bg}}$: Giảm dần (model learns to be conservative)

### 8.3 Full Semi-SSL Phase ($t \geq 10$)

$$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \mathcal{L}_{\text{unsup}}
$$

Với cả $\lambda_u = 1.0$ và $\lambda_{\text{bg}} = 1.0$.

---

## 9. LOSS COMPONENTS ORDER OF MAGNITUDE

Giá trị điển hình ở epoch 10 (sau warmup):

| Component | Typical Value | Range |
|-----------|---------------|-------|
| $\mathcal{L}_{\text{box}}$ | 0.5 - 1.0 | [0.3, 2.0] |
| $\mathcal{L}_{\text{cls}}^{\text{BCE}}$ | 0.2 - 0.5 | [0.1, 1.0] |
| $\mathcal{L}_{\text{bg}}$ | 0.05 - 0.15 | [0.01, 0.3] |
| $\mathcal{L}_{\text{dfl}}$ | 0.3 - 0.6 | [0.2, 1.0] |
| **Total** | **1.0 - 2.5** | **[0.6, 4.0]** |

**After weighting:**

$$
\mathcal{L}_{\text{weighted}} = 7.5 \times 0.5 + 0.5 \times 0.4 + 1.5 \times 0.4 \approx 4.5
$$

---

## 10. KEY INSIGHTS

### 10.1 Tại Sao Background Penalty Hiệu Quả?

1. **Explicit regularization** trên background regions
2. **Asymmetric penalty:** Chỉ phạt predictions, không phạt targets
3. **Adaptive:** Penalty mạnh nhất khi model confident nhầm

### 10.2 Vai Trò của TAL

TAL alignment scores $y_{ij}^c$ làm **soft targets** thay vì hard 0/1:
- Giảm gradient noise
- Better credit assignment
- Smooth optimization landscape

### 10.3 Interplay giữa Components

$$
\mathcal{L}_{\text{cls}} \uparrow \implies \text{Model predictions less confident}
$$
$$
\mathcal{L}_{\text{bg}} \uparrow \implies \text{Background predictions too high}
$$
$$
\mathcal{L}_{\text{box}} \uparrow \implies \text{Localization error}
$$

Cân bằng 3 components này quyết định quality của pseudo-labels.

---

## 11. ABLATION ANALYSIS

### Loại bỏ Background Penalty ($\lambda_{\text{bg}} = 0$)

$$
\mathcal{L}_{\text{cls}} = \frac{1}{\sum y_{ij}} \sum_{i,j,c} \mathcal{L}_{\text{BCE}}(\hat{y}_{ij}^c, y_{ij}^c)
$$

**Vấn đề:**
- Background anchors với $y_{ij} = 0$ chỉ phạt riêng lẻ từng class
- Model có thể dự đoán nhiều classes cùng confidence trung bình → Tổng confidence cao
- False positives tăng trong pseudo-labels

### Tăng Background Penalty ($\lambda_{\text{bg}} = 2.0$)

**Trade-off:**
- ✅ Precision tăng mạnh (ít FP)
- ❌ Recall giảm (model quá conservative)
- ❌ Pseudo-labels ít hơn → Underutilize unlabeled data

---

## Summary Table

| Loss Term | Formula | Purpose | Typical Value |
|-----------|---------|---------|---------------|
| Box Loss | $\frac{1}{\Sigma y} \sum w(1-\text{CIoU})$ | Localization | 0.5-1.0 |
| BCE Loss | $\frac{1}{\Sigma y} \sum \text{BCE}(\hat{y}, y)$ | Classification | 0.2-0.5 |
| BG Penalty | $\frac{\lambda_{bg}}{\|\mathcal{B}\|} \sum \max_c \sigma(\hat{y}_c)$ | FP reduction | 0.05-0.15 |
| DFL Loss | $\frac{1}{\Sigma y} \sum w \cdot \text{DFL}$ | Box refinement | 0.3-0.6 |

---

**Tổng kết:** Loss function hiện tại kết hợp:
1. Standard YOLO detection loss (box + cls + dfl)
2. Semi-supervised paradigm (sup + unsup)
3. Background penalty (custom addition)

Tất cả components được cân bằng qua hyperparameters và scheduling.
