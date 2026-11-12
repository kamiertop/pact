# -*- coding: utf-8 -*-
import os
import random
import sys
from pathlib import Path
from time import perf_counter

from utils.args import parse


def setup(seed: int):
    import numpy as np
    import torch

    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)


if __name__ == '__main__':
    # --------- reproducibility ----------
    cfg = parse("train")
    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    import torch
    import torch.optim as optim

    from dataset.ev_uav import EvUAV
    from model.loss_utils import build_p2v_map_for_level, vel_loss_v1_only
    from model.PACT import PACT
    from utils.eval import evalute
    from utils.r1_loss import R1Loss
    from utils.run_manager import RunManager
    from utils.stcloss import STCLoss

    seed = cfg.seed
    setup(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- model ----------
    net = PACT(cfg).train().to(device)

    # --------- data ----------
    dataset = EvUAV(cfg, mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=train_sampler)

    val_dataset = EvUAV(cfg, mode='val')
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=val_dataset.custom_collate,
        num_workers=0,
        pin_memory=True
    )

    # --------- losses ----------
    stc_criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).to(device)
    r1_criterion = R1Loss(cfg) if any((cfg.risk_weight, cfg.iou_weight, cfg.hardneg_weight)) else None

    # --------- optim ----------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    best_iou = float("-inf")
    best_iou_epoch = -1

    # --------- evaluator ----------
    evaluter = evalute(cfg, roc=False)

    run = RunManager(cfg.run_root, cfg.run_name, cfg)
    print(f"Run directory: {run.run_dir}", flush=True)

    for epoch in range(cfg.epochs):
        epoch_started = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        loss_total = 0.0
        net.train()
        for batch_idx, batch in enumerate(train_loader):
            # ---- inputs ----
            x = batch['voxel_ev']
            # spconv 的 SparseConvTensor 支持 .to
            try:
                x = x.to(device)
            except AttributeError:
                pass

            # ---- forward ----
            ev_bxyt = torch.as_tensor(batch['locs'], device=device, dtype=torch.long)  # [Ne,4]
            seg_label_ev = torch.as_tensor(batch['seg_label_ev'], device=device, dtype=torch.long)  # [Ne]
            p2v_map_ev = batch['p2v_map'].to(device).long()  # [Ne] 事件→预测体素

            output = net(x, return_logits=r1_criterion is not None)
            preds, voxel, v1, idx_v1 = output[:4]
            # 事件坐标与标签（dataset给出的原始分辨率体素坐标）
            assert set(ev_bxyt[:, 0].unique().tolist()) == set(idx_v1[:, 0].unique().tolist())
            p2v_v1 = build_p2v_map_for_level(ev_bxyt, idx_v1.long(), stride=(1, 1, 1))


            # 主任务：STC 权重交叉熵 —— 这里仍然用**张量** p2v_map_ev
            loss_task = stc_criterion(voxel, p2v_map_ev, preds, seg_label_ev.float())

            # v1 速度监督（事件级 → v1 体素）
            loss_v1 = vel_loss_v1_only(
                v1=v1,
                idx_v1=idx_v1,
                batch={'ev_bxyt': ev_bxyt, 'seg_label': seg_label_ev},
                p2v_map_v1=p2v_v1,
                huber_delta=0.5,
                return_logs=False,
            )

            loss = loss_task + 0.3 * loss_v1
            r1_metrics = {}
            if r1_criterion is not None:
                instance_ids = None
                if cfg.risk_weight:
                    instance_ids = torch.as_tensor(batch["idx_label"], device=device)
                loss_r1, components = r1_criterion(
                    output[4][p2v_map_ev], seg_label_ev, ev_bxyt[:, 0], instance_ids,
                )
                loss = loss + loss_r1
                r1_metrics = {f"loss_{name}": value.detach().item()
                              for name, value in components.items()}
                r1_metrics["loss_r1"] = loss_r1.detach().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach().item()
            loss_total += batch_loss
            run.log_metric(epoch, batch_idx, loss=batch_loss, lr=lr,
                           loss_task=loss_task.detach().item(), loss_v1=loss_v1.detach().item(),
                           **r1_metrics)

            # 释放显存碎片
            del output, preds, voxel

        mean_loss = loss_total / len(train_loader)
        if mean_loss < best_loss:
            best_loss = mean_loss
            run.save_checkpoint(net, f"best_loss_seed{seed}.pt")
        scheduler.step()

        # ---------------- 验证（保持你原有逻辑） ----------------
        net.eval()
        try:
            evaluter.reset()
        except Exception:
            evaluter.matches = {}

        with torch.no_grad():
            for sample_idx, batch in enumerate(val_loader):
                x_val = batch['voxel_ev']
                label_val = batch['seg_label_ev'].float().to(device)
                p2v_map_val = batch['p2v_map'].long().to(device)  # 验证集仍是事件->体素**张量**

                try:
                    x_val = x_val.to(device)
                except AttributeError:
                    pass

                preds_val, voxel_val, v1, idx_v1 = net(x_val)

                seg_pred = preds_val[p2v_map_val].squeeze().detach().cpu()
                seg_gt   = label_val.cpu()

                evaluter.matches[str(sample_idx)] = {}
                evaluter.matches[str(sample_idx)]['seg_pred'] = seg_pred
                evaluter.matches[str(sample_idx)]['seg_gt']   = seg_gt

                del preds_val, voxel_val, x_val
                torch.cuda.empty_cache()

        iou = float(evaluter.evaluate_semantic_segmantation_miou(cfg.prediction_thresh))
        if iou > best_iou:
            best_iou = iou
            best_iou_epoch = epoch
            run.save_checkpoint(net, f"best_iou_seed{seed}.pt")
        patience = cfg.early_stopping_patience
        early_stopped = (
            patience > 0 and best_iou_epoch >= 0
            and epoch - best_iou_epoch >= patience and epoch + 1 < cfg.epochs
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = perf_counter() - epoch_started
        run.log_metric(epoch, loss=mean_loss, lr=lr, val_mIoU=iou,
                       best_val_mIoU=best_iou, epoch_seconds=epoch_seconds,
                       early_stopped=early_stopped)
        stop_message = f" | early stop ({patience} epochs without improvement)" if early_stopped else ""
        print(f"Epoch {epoch + 1}/{cfg.epochs} | loss={mean_loss:.6f} | "
              f"lr={lr:.3g} | val IoU={iou:.4f} | best IoU={best_iou:.4f} | "
              f"time={epoch_seconds:.2f}s{stop_message}", flush=True)
        if early_stopped:
            break

    if cfg.test_after_train:
        run_dir = run.run_dir.resolve()
        print(f"Testing the best validation checkpoint in {run_dir}", flush=True)
        command = [
            sys.executable, "-u", str(Path(__file__).resolve().with_name("test.py")),
            "--split", "test", "--run_dir", str(run_dir), "--batch_size", "1",
            "--output_json", str(run_dir / "test_metrics.json"),
        ]
        # Replace the training process so testing starts with a fresh CUDA context.
        os.execv(sys.executable, command)
