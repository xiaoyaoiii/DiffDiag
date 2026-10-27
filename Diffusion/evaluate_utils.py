import math


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for k in topN:
        sum_precision = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_mrr = 0.0
        effective_user = 0

        for gt, pred in zip(GroundTruth, predictedIndices):
            if len(gt) == 0:
                continue
            effective_user += 1
            gt_set = set(gt)
            user_hit = 0
            dcg = 0.0
            idcg = 0.0
            user_mrr = 0.0
            mrr_flag = True
            idcg_count = len(gt)

            for rank in range(min(k, len(pred))):
                if pred[rank] in gt_set:
                    user_hit += 1
                    dcg += 1.0 / math.log2(rank + 2)
                    if mrr_flag:
                        user_mrr = 1.0 / (rank + 1.0)
                        mrr_flag = False
                if idcg_count > 0:
                    idcg += 1.0 / math.log2(rank + 2)
                    idcg_count -= 1

            ndcg = dcg / idcg if idcg > 0 else 0.0
            sum_precision += user_hit / k
            sum_recall += user_hit / len(gt)
            sum_ndcg += ndcg
            sum_mrr += user_mrr

        denom = max(effective_user, 1)
        precision.append(round(sum_precision / denom, 4))
        recall.append(round(sum_recall / denom, 4))
        NDCG.append(round(sum_ndcg / denom, 4))
        MRR.append(round(sum_mrr / denom, 4))

    return precision, recall, NDCG, MRR


def print_results(loss, valid_result, test_result):
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))
