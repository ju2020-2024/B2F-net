import torch
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, roc_curve
import numpy as np

def test(test_loader, model, gt,args):
    model.eval()
    pred = torch.zeros(0).to(args.device)
    with torch.no_grad():
        for inputs in test_loader:

            inputs = inputs.to(args.device)

            out = model(inputs,None) # 只要片段分数
            s1, s2, s3 = out["frame_scores"]
            segment_scores = (s1 + s2 + s3) / 3
            segment_scores = torch.sigmoid(segment_scores)
            segment_scores = torch.mean(segment_scores,0)
            pred = torch.cat((pred, segment_scores))

        pred_np = list(pred.cpu().detach().numpy())
        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred_np, 16))
        pr_auc = auc(recall, precision)
        fpr, tpr, threshold = roc_curve(list(gt), np.repeat(pred_np, 16))
        rec_auc = auc(fpr, tpr)



        return rec_auc, pr_auc


