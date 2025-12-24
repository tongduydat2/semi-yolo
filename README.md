# Semi-Supervised Object Detection (SSOD) Framework

Hệ thống huấn luyện Semi-Supervised Object Detection cho chuyển đổi miền Gray → IronRed với YOLOv11.

## Cấu trúc thư mục

```
semi_training/
├── config/
│   └── ssod_config.yaml      # Cấu hình hyperparameters
├── data/
│   ├── spectrum_analysis.py  # Phân tích phổ màu IronRed
│   ├── colorization.py       # Pipeline "Nhuộm màu" (LUT)
│   └── augmentation.py       # Weak/Strong Augmentation
├── models/
│   ├── ema.py                # EMA updater cho Teacher
│   └── teacher_student.py    # Teacher-Student Framework
├── training/
│   ├── pseudo_labeler.py     # Sinh nhãn giả + Adaptive Threshold
│   ├── losses.py             # SSOD Loss functions
│   └── ssod_trainer.py       # Main training loop
├── utils/
│   ├── visualization.py      # Visualize kết quả
│   └── metrics.py            # Đánh giá mAP, precision, recall
├── train_ssod.py             # Entry point
└── README.md
```

## Cài đặt

```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
```

## Chuẩn bị dữ liệu

### 1. Cấu trúc thư mục dữ liệu (YOLO format)

```
datasets/
├── labeled/           # Ảnh Gray có nhãn
│   ├── images/
│   └── labels/
├── unlabeled/         # Ảnh IronRed thật (không nhãn)
│   └── images/
├── fake_ironred/      # Sẽ được tạo bởi colorization
│   ├── images/
│   └── labels/
└── val/               # Tập validation
    ├── images/
    └── labels/
```

### 2. Chuyển đổi Gray → IronRed

```bash
cd semi_training

# Phân tích phổ màu từ ảnh IronRed thật (tuỳ chọn)
python train_ssod.py analyze --input ../datasets/unlabeled/images -o spectrum.png

# Chuyển đổi ảnh Gray sang Fake IronRed
python train_ssod.py colorize \
    --input ../datasets/labeled/images \
    --output ../datasets/fake_ironred \
    --labels ../datasets/labeled/labels
```

## Huấn luyện

### Cấu hình

Chỉnh sửa `config/ssod_config.yaml`:

```yaml
model:
  base_model: "yolo11n.pt"  # hoặc yolo11s.pt, yolo11m.pt, ...
  num_classes: 1

training:
  max_epochs: 100
  burn_in_epochs: 15        # Số epoch chỉ train supervised
  batch_size: 16

ssod:
  confidence_threshold: 0.75  # Tau
  unsupervised_weight: 2.0    # Lambda
  ema_rate: 0.999             # Alpha
```

### Chạy training

```bash
python train_ssod.py train --config config/ssod_config.yaml
```

### Resume training

```bash
python train_ssod.py train --config config/ssod_config.yaml --resume runs/ssod/checkpoint.pt
```

## Đánh giá

```bash
python train_ssod.py evaluate --model runs/ssod/student_final.pt --data path/to/test.yaml
```

## Thuật toán SSOD

```
ALGORITHM: SSOD_Training_Loop

FOR Epoch in 1 to Max_Epochs:
    
    // Giai đoạn 1: Burn-in (Lambda = 0)
    IF Epoch < Burn_In_Epochs THEN
        Current_Lambda = 0
    ELSE
        Current_Lambda = Lambda
    END IF

    // 1. Teacher sinh nhãn giả (không gradient)
    Pseudo_Labels = Teacher.Predict(Unlabeled_Images)
    Filter by Confidence > Tau

    // 2. Student học trên labeled data
    Loss_Sup = YOLO_Loss(Student.Predict(Labeled), Ground_Truth)

    // 3. Student học trên pseudo-labels
    Loss_Unsup = YOLO_Loss(Student.Predict(Unlabeled_Strong_Aug), Pseudo_Labels)

    // 4. Cập nhật Student
    Total_Loss = Loss_Sup + Lambda * Loss_Unsup
    Backpropagation(Total_Loss)

    // 5. Cập nhật Teacher (EMA)
    W_Teacher = Alpha * W_Teacher + (1 - Alpha) * W_Student

END FOR
```

## Tiêu chuẩn đánh giá

| Giai đoạn | Tiêu chuẩn |
|-----------|------------|
| Colorization | Histogram similarity > 80% |
| Burn-in | Box loss giảm nhanh |
| SSOD | mAP tăng 3-5% so với supervised-only |

## API Usage

```python
from semi_training import SSODTrainer, IronRedColorizer

# Colorize images
colorizer = IronRedColorizer()
colorizer.colorize_directory("input", "output", copy_labels=True)

# Train SSOD
trainer = SSODTrainer("config/ssod_config.yaml")
trainer.train()
```
# semi-yolo
# semi-yolo
