import os

from utils.args import parse


if __name__ == "__main__":
    cfg = parse("test")
    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    import torch

    from dataset.ev_uav import EvUAV
    from model.PACT import PACT
    from utils.eval import evalute

    net = PACT(cfg).eval().cuda()
    net.load_state_dict(torch.load(cfg.checkpoint_path))
    print(f"Loaded checkpoint: {cfg.checkpoint_path}")

    dataset = EvUAV(cfg, mode=cfg.split)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate,
    )
    evaluator = evalute(cfg, roc=cfg.roc)
    with torch.no_grad():
        for sample, events in enumerate(loader):
            preds, _, _, _ = net(events["voxel_ev"])
            p2v_map = events["p2v_map"].long().cuda()
            preds = preds[p2v_map].squeeze().cpu()
            labels = events["seg_label_ev"].float().cpu()
            evaluator.matches[str(sample)] = {"seg_pred": preds, "seg_gt": labels}
            if cfg.roc:
                locs = events["locs"].float()
                evaluator.roc_update(
                    locs[:, 3], preds, events["idx_label"], labels, locs,
                    thresh=cfg.prediction_thresh,
                )

    iou = evaluator.evaluate_semantic_segmantation_miou(cfg.prediction_thresh)
    acc = evaluator.evaluate_semantic_segmantation_accuracy(cfg.prediction_thresh)
    result = f"iou:{iou},seg_acc:{acc}"
    if cfg.roc:
        pd, fa = evaluator.cal_roc()
        result += f",pd:{pd},fa:{fa}"
    print(result)
