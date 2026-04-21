"""
Mô đun đánh giá chỉ số hiệu năng phân loại (Metrics evaluation) dành riêng cho hệ thống PhaBERT-CNN.

Các thông số tĩnh phân loại nhị phân chuẩn (Standard Binary Classification Metrics):
- Độ nhạy (Sensitivity - Sn) = TP / (TP + FN)  -- Thước đo biểu thị sức mạnh sàng lọc chủng thực khuẩn thể độc lực
- Độ đặc hiệu (Specificity - Sp) = TN / (TN + FP)  -- Thước đo biểu thị mức độ nhận diện chính xác chủng ôn hoà
- Độ chính xác toàn cục (Accuracy - Acc) = (TP + TN) / (TP + FN + TN + FP)

Lược đồ quy ước nhãn (Label Convention):
- Nhóm dương (Positive Class - Y = 1) = Khảo phân nhóm chủng Độc lực (Virulent pha)
- Nhóm âm  (Negative Class - Y = 0) = Khảo phân nhóm chủng Ôn hoà (Temperate pha)
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(
    y_true,
    y_pred,
    y_score: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Quy trình tính toán đồng bộ độ nhạy (sensitivity), độ đặc hiệu (specificity), độ chính xác (accuracy) 
    và các tùy chọn diện tích dưới đường cong nhận tín hiệu hoạt động bộ giải đoán (AUC-ROC).

    Tham số đầu vào (Args):
        y_true:  Danh sách lưu vết nhãn thực tiễn chuẩn (Ground truth): (0 = ôn hoà, 1 = độc lực)
        y_pred:  Danh sách biến nhãn ước lượng thông qua suy luận mô hình
        y_score: Giá trị ngưỡng xác suất định biên cho lớp 1 (chủng độc lực). Khi tham số được cấp quyền, thông số AUC-ROC
                 sẽ tự động thiết lập và hội hợp vào kết quả.

    Kết xuất đầu ra (Returns):
        Cấu trúc bộ từ điển dữ liệu (Dictionary) lưu trữ thông tin sensitivity, specificity, accuracy (Đơn vị cấu trúc %)
        và tùy biến auc (tỉ lệ thang 0-100), bao gồm tổng kết đếm các điểm đánh giá hệ ma trận nhầm lẫn (TP/TN/FP/FN).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    result = {
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'accuracy':    accuracy    * 100,
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }

    if y_score is not None:
        try:
            result['auc'] = roc_auc_score(y_true, y_score) * 100
        except ValueError:
            result['auc'] = float('nan')

    return result


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Hỗ trợ in ấn kết quả định dạng ma trận chỉ số đánh giá (metrics format printing)."""
    print(f"{prefix}Sensitivity (sn): {metrics['sensitivity']:.2f}%")
    print(f"{prefix}Specificity (sp): {metrics['specificity']:.2f}%")
    print(f"{prefix}Accuracy   (acc): {metrics['accuracy']:.2f}%")
    if 'auc' in metrics:
        print(f"{prefix}AUC-ROC:          {metrics['auc']:.2f}%")


def aggregate_fold_metrics(fold_metrics: list) -> Dict[str, float]:
    """
    Kích hoạt chức năng gộp kết quả chuỗi k nếp gấp (k-fold aggregation) cung cấp ước lượng đặc điểm xu hướng với Mean (trung bình) ± Std (độ lệch chuẩn).

    Tham biến (Args):
        fold_metrics: Dữ liệu phân bố tập (Dictionary list) xuất phát từ tham số qua từng vòng lặp fold.

    Dữ liệu trả về (Returns):
        Từ điển liên kết thông số trung bình (`{metric}_mean`) và kích thước dao động (`{metric}_std`) thuộc các hạng mục tương ứng được tích hợp.
    """
    keys = ['sensitivity', 'specificity', 'accuracy']
    if any('auc' in m for m in fold_metrics):
        keys.append('auc')

    result = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
        result[f'{key}_mean'] = float(np.mean(values)) if values else float('nan')
        result[f'{key}_std']  = float(np.std(values))  if values else float('nan')
    return result
