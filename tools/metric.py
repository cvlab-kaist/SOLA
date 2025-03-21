
def recall_per_track(
    gt_anno_ids,
    preds,
    labels,
    corresponding_gt_anno_ids,
):
    '''
    args:
        gt_anno_ids: list of gt_anno_ids
        preds: tensor of shape (num_preds,)
        labels: tensor of shape (num_preds,)
        corresponding_gt_anno_ids: list of gt_anno_ids corresponding to each pred (len = num_preds)
    return:
        recall_per_track: list of recall for each track in gt_anno_ids
    '''
    recall_per_track = []
    for gt_anno_id in gt_anno_ids:
        tp, fn = 0, 0
        for pred, label, corresponding_gt_anno_id in zip(preds, labels, corresponding_gt_anno_ids):
            if corresponding_gt_anno_id == gt_anno_id and label == 1:
                if pred > 0:
                    tp += 1
                else:
                    fn += 1
            else:
                continue
        if tp + fn == 0:
            continue
        recall = tp / (tp + fn)
        recall_per_track.append(recall)
    return recall_per_track

def recall_per_exp(
    gt_anno_ids,
    preds,
    labels,
    corresponding_gt_anno_ids,
):
    '''
    args:
        gt_anno_ids: list of gt_anno_ids
        preds: tensor of shape (num_preds,)
        labels: tensor of shape (num_preds,)
        corresponding_gt_anno_ids: list of gt_anno_ids corresponding to each pred (len = num_preds)
    return:
        metric_per_exp: list of metric for each track in gt_anno_ids
    '''
    n_total = len(gt_anno_ids)
    n_detected = 0
    for gt_anno_id in gt_anno_ids:
        for pred, label, corresponding_gt_anno_id in zip(preds, labels, corresponding_gt_anno_ids):
            if corresponding_gt_anno_id == gt_anno_id and label == 1 and pred > 0:
                n_detected += 1
                break
            else:
                continue
    recall = n_detected / n_total
    return recall