# import torch
# import pickle
#
#
# def save_res(img, gt, prob, filenames, metric_meter):
#     _, pred_label = torch.max(prob, dim=1)
#     img_ = img.cpu().numpy()
#     gt_ = gt.cpu().numpy()
#     prob_ = prob.cpu().numpy()
#     for idx, file in enumerate(filenames):
#         metric_meter.reset()
#         rgb = img_[idx, :]
#         mask_gt = gt_[idx, :]
#         pred_map = prob_[idx, :]
#         pred_label = pred_label[idx, :]
#         metric_meter.add(pred_label, mask_gt)
#         miou, ious = metric_meter.miou()
#
