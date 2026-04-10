
# ══════════════════════════════════════════════════════════════════════════════
#  Integration Hooks: Co-occurrence GCN → YOLOv11m-Seg (Ultralytics)
#  Paste this into your existing PathoLink training/inference script.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
from pathlib import Path

# ── 1. Load pre-trained co-occurrence modules ──────────────────────────────────
A_norm        = np.load("outputs/matrices/A_norm.npy")
Z_embeddings  = np.load("outputs/gcn/class_embeddings.npy")
# gcn, dyn_adj, X_init are defined in co_occurrence_guided_patholink.py

# ── 2. Wrap YOLO prediction call ──────────────────────────────────────────────
from ultralytics import YOLO

def patholink_predict(model_path, image_path, save_dir="runs/patholink"):
    """
    Run PathoLink inference with co-occurrence guidance.
    Replaces the vanilla model.predict() call.
    """
    model  = YOLO(model_path)
    # Standard YOLO inference
    result = model.predict(
        source     = image_path,
        conf       = 0.10,           # low threshold — GCN will re-score
        iou        = 0.45,
        save       = False,
        verbose    = False,
    )[0]

    # Convert to our dict format
    yolo_dets = []
    if result.boxes is not None:
        for box in result.boxes:
            yolo_dets.append({
                "class_id": int(box.cls.item()),
                "conf"    : float(box.conf.item()),
                "bbox"    : box.xywhn.squeeze().tolist(),
            })

    # Co-occurrence guided re-scoring
    rescored, cooc_mat = cooccurrence_guided_rescore(
        yolo_dets, gcn, X_init, dyn_adj
    )

    # Filter by threshold
    final = [d for d in rescored if d["conf"] >= CFG["conf_threshold"]]

    # Generate heatmap report
    img_id = Path(image_path).stem
    generate_cooccurrence_heatmap(cooc_mat, rescored, img_id)

    return final, cooc_mat

# ── 3. Training integration (custom callback) ─────────────────────────────────
#
#  In your YOLOv11m-Seg training script, add this callback AFTER each epoch:
#
#  from ultralytics import YOLO
#  model = YOLO("yolo11m-seg.yaml")
#
#  def on_train_epoch_end(trainer):
#      """Re-mine co-occurrence matrix and update GCN every N epochs."""
#      if trainer.epoch % 5 == 0 and trainer.epoch > 0:
#          # Re-mine from current predictions on val set
#          val_preds = trainer.validator.pred_results   # list of {class_id, conf}
#          new_count, new_cond, new_freq, new_pairs = mine_cooccurrence(
#              val_preds  # pass current val predictions as pseudo-labels
#          )
#          new_A, _, _ = build_adjacency_matrix(new_cond)
#          gcn.layer1.A = new_A
#          gcn.layer2.A = new_A
#          # One epoch of GCN fine-tuning
#          simulate_gcn_training(gcn, data["val"], n_epochs=5)
#          print(f"  Co-occurrence GCN updated at epoch {trainer.epoch}")
#
#  model.add_callback("on_train_epoch_end", on_train_epoch_end)
#  model.train(data="data.yaml", epochs=100, imgsz=640, ...)

# ── 4. Custom loss term (add to YOLOv11m-Seg training) ────────────────────────
#
#  To propagate co-occurrence guidance into the YOLO backbone itself,
#  add a co-occurrence consistency loss term:
#
#  def cooccurrence_consistency_loss(cls_logits, gt_multilabel):
#      """
#      cls_logits   : (B, N) — raw class logits from YOLO head
#      gt_multilabel: (B, N) — multi-hot ground truth
#      Returns scalar L_cooc to add to the standard YOLO loss.
#      """
#      import torch, torch.nn.functional as F
#      B, N = cls_logits.shape
#      A_t = torch.tensor(A_norm, dtype=torch.float32).to(cls_logits.device)
#      # GCN smoothing: encouraged prediction should be A-consistent
#      probs    = torch.sigmoid(cls_logits)           # (B, N)
#      A_probs  = probs @ A_t                         # (B, N)  neighbour-aggregated
#      # Penalise inconsistency between prediction and co-occurrence expectation
#      L_cooc   = F.mse_loss(A_probs, gt_multilabel.float() @ A_t)
#      return L_cooc
#
#  # In the training loop:
#  # total_loss = yolo_loss + lambda_cooc * cooccurrence_consistency_loss(logits, gt)
#  # Recommended lambda_cooc = 0.05 - 0.10
