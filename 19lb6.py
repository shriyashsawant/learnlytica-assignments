"""Day 19 / Session 3 - Lab Assignment 2: Scoring a Detector - Precision/Recall/F1 at an IoU Threshold."""


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union == 0:
        return 0.0
    return float(inter / union)


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Greedy matching: highest-scoring prediction first, each ground-truth box can be matched only once."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    used = set()
    tp = 0
    fp = 0
    for i in order:
        best_iou = 0.0
        best_gt = None
        for j in range(len(gt_boxes)):
            if j in used:
                continue
            value = iou(pred_boxes[i], gt_boxes[j])
            if value > best_iou:
                best_iou = value
                best_gt = j
        if best_gt is not None and best_iou >= iou_thresh:
            tp += 1
            used.add(best_gt)
        else:
            fp += 1
    fn = len(gt_boxes) - len(used)               # ground truth nobody matched
    return tp, fp, fn


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    gt_boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]
    pred_boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [100, 100, 110, 110]]
    pred_scores = [0.90, 0.85, 0.80, 0.70]
    tp, fp, fn = match_predictions(pred_boxes, pred_scores, gt_boxes, 0.5)
    print(tp, fp, fn)                                          # 2 2 1
    p, r = precision_recall(tp, fp, fn)
    print(round(p, 3), round(r, 3))                            # 0.5 0.667
    print(round(f1_score(p, r), 3))                            # 0.571
    print(precision_recall(0, 0, 5))                           # (0.0, 0.0)
    print(match_predictions(gt_boxes, [0.9, 0.8, 0.7], gt_boxes, 0.5))             # (3, 0, 0)
    near = [[0, 0, 10, 8]]                                     # IoU 0.8 with the box [0,0,10,10]? 80/100
    print(match_predictions([[0, 0, 10, 6]], [0.9], [[0, 0, 10, 10]], 0.5), match_predictions([[0, 0, 10, 6]], [0.9], [[0, 0, 10, 10]], 0.7))  # (1,0,0) (0,1,1)
    print(match_predictions([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [[0, 0, 10, 10]], 0.5))             # (1, 1, 0)
