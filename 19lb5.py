"""Day 19 / Session 3 - Lab Assignment 1: IoU, Confidence Filtering & Non-Max Suppression."""


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return float(inter / union)


def filter_by_confidence(boxes, scores, thresh):
    return [i for i in range(len(scores)) if scores[i] >= thresh]    # original order


def nms(boxes, scores, iou_thresh):
    # visit candidates from the highest score down (sorted is stable, so ties keep input order)
    remaining = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    while remaining:
        best = remaining.pop(0)                  # highest-scoring remaining box
        keep.append(best)
        remaining = [i for i in remaining if iou(boxes[best], boxes[i]) <= iou_thresh]   # drop heavy overlaps
    return keep


if __name__ == "__main__":
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30], [21, 21, 30, 30]]
    scores = [0.90, 0.80, 0.70, 0.60]
    print(round(iou([0, 0, 10, 10], [0, 0, 10, 10]), 4))        # 1.0
    print(round(iou([0, 0, 10, 10], [20, 20, 30, 30]), 4))      # 0.0
    print(round(iou([0, 0, 10, 10], [5, 5, 15, 15]), 4))        # 0.1429
    print(filter_by_confidence(boxes, scores, 0.5))             # [0, 1, 2, 3]
    print(nms(boxes, scores, 0.5))                              # [0, 2]
    print(nms(boxes, scores, 0.95))                             # [0, 1, 2, 3]
    print(round(iou([0, 0, 10, 10], [4, 4, 6, 6]), 4), iou([0, 0, 10, 10], [4, 4, 6, 6]) == iou([4, 4, 6, 6], [0, 0, 10, 10]))  # 0.04 True
    print(nms([[0, 0, 10, 10], [1, 1, 10, 10]], [0.3, 0.9], 0.5))   # [1] (max score wins, not first-in-array)
