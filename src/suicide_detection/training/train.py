import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from suicide_detection.data_processing.load import load_dataset_secure
from suicide_detection.data_processing.splitting import temporal_split
from suicide_detection.evaluation.metrics import compute_metrics, fairness_metrics
from suicide_detection.evaluation.plots import save_confusion, save_curves
from suicide_detection.models.bert_model import TextDataset, build_model_and_tokenizer
from suicide_detection.models.bilstm_attention import BiLSTMAttention, BiLSTMAttentionConfig
from suicide_detection.models.svm_baseline import SVMBaseline
from suicide_detection.utils.logging import get_logger
from suicide_detection.utils.seed import set_global_seed

# -------------------------
# Shared helpers
# -------------------------

def _load_configs(default_path: Optional[str], override_path: Optional[str]) -> Dict[str, Any]:
    cfg_default: Dict[str, Any] = {}
    cfg_model: Dict[str, Any] = {}
    if default_path and Path(default_path).exists():
        try:
            cfg_default = yaml.safe_load(Path(default_path).read_text()) or {}
        except Exception:
            cfg_default = {}
    if override_path and Path(override_path).exists():
        try:
            cfg_model = yaml.safe_load(Path(override_path).read_text()) or {}
        except Exception:
            cfg_model = {}
    return {**cfg_default, **cfg_model}


def _setup_mlflow(logger, cfg_file: Path = Path("configs/mlflow.yaml")) -> bool:
    ml_enabled = False
    if cfg_file.exists():
        try:
            cfg = yaml.safe_load(cfg_file.read_text())
            ml_enabled = bool(cfg.get("enabled", False))
            if ml_enabled:
                tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(cfg.get("experiment_name", "suicide_detection_research"))
        except Exception as e:
            logger.warning(f"Failed to read MLflow config: {e}")
    return ml_enabled


# BiLSTM helpers factored out to reduce run_bilstm complexity

def bilstm_build_vocab(texts, min_freq: int = 2, max_size: int = 50000):
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(str(t).split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def bilstm_encode(text, vocab, max_len: int):
    return [vocab.get(tok, 1) for tok in str(text).split()][:max_len]


def bilstm_make_dataset(X, y, vocab, max_len: int):
    enc = [torch.tensor(bilstm_encode(t, vocab, max_len), dtype=torch.long) for t in X]
    attn = [torch.ones_like(e, dtype=torch.bool) for e in enc]
    labels = [torch.tensor(int(lbl), dtype=torch.long) for lbl in y]
    return list(zip(enc, attn, labels))


def bilstm_collate(batch, max_len: int):
    from torch.nn.utils.rnn import pad_sequence

    ids, attn, labels = zip(*batch)
    ids_pad = pad_sequence(ids, batch_first=True, padding_value=0)
    attn_pad = pad_sequence(attn, batch_first=True, padding_value=0)
    labels_t = torch.stack(labels)
    return ids_pad[:, :max_len], attn_pad[:, :max_len], labels_t


def bilstm_evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ids, attn, labels in loader:
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            logits = model(ids, attn)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            ys.append(labels.cpu())
            ps.append(prob.cpu())
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(ps).numpy()
    return compute_metrics(y_true, y_prob)


def bilstm_collect(model, loader, device):
    ys, ps = [], []
    with torch.no_grad():
        for ids, attn, labels in loader:
            ids, attn = ids.to(device), attn.to(device)
            outputs = model(ids, attn)
            prob = torch.softmax(outputs, dim=-1)[:, 1]
            ys.append(labels)
            ps.append(prob.cpu())
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(ps).numpy()
    return y_true, y_prob


def bilstm_train(model, train_loader, val_loader, device, logger, criterion, optimizer, output_dir: Path, num_epochs: int, early_patience: int):
    best_f1, patience, max_patience = 0.0, 0, early_patience
    for epoch in range(1, num_epochs + 1):
        model.train()
        for ids, attn, labels in train_loader:
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids, attn)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_res = bilstm_evaluate(model, val_loader, device)
        logger.info(f"BiLSTM epoch {epoch} val F1={val_res.f1:.4f}")
        if val_res.f1 > best_f1 + 1e-4:
            best_f1 = val_res.f1
            patience = 0
            out_model = output_dir / "bilstm.pt"
            out_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_model)
        else:
            patience += 1
            if patience > max_patience:
                break
    return model


def bilstm_save_attention_visualizations(model, val_loader, device, output_dir: Path, logger):
    try:
        import matplotlib.pyplot as plt

        vis_dir = output_dir / "attention"
        vis_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        model.eval()
        with torch.no_grad():
            for ids, attn, _labels in val_loader:
                ids, attn = ids.to(device), attn.to(device)
                _ = model(ids, attn)
                attn_w = getattr(model, "last_attn", None)
                if attn_w is None:
                    break
                attn_w = attn_w.cpu().numpy()
                for i in range(min(attn_w.shape[0], 5 - count)):
                    plt.figure(figsize=(8, 2))
                    plt.imshow(attn_w[i][None, : ids.shape[1]], aspect="auto", cmap="viridis")
                    plt.colorbar()
                    plt.yticks([])
                    plt.xlabel("Token index")
                    plt.title("BiLSTM Attention Weights (sample {})".format(i))
                    plt.savefig(vis_dir / f"val_attn_{count+i}.png", dpi=150, bbox_inches="tight")
                    plt.close()
                count += min(attn_w.shape[0], 5 - count)
                if count >= 5:
                    break
    except Exception as e:
        logger.warning(f"Failed to save attention visualizations: {e}")


# BERT helpers factored out to reduce run_bert complexity

def bert_hf_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    res = compute_metrics(labels, probs)
    return {
        "accuracy": res.accuracy,
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "roc_auc": res.roc_auc if res.roc_auc is not None else float("nan"),
        "pr_auc": res.pr_auc if res.pr_auc is not None else float("nan"),
    }


def bert_save_outputs(trainer, val_ds, y_val, y_test, y_prob_test, X_test, output_dir: Path):
    try:
        y_val_prob_local = trainer.predict(val_ds).predictions
        y_val_prob_local = torch.softmax(torch.tensor(y_val_prob_local), dim=-1)[:, 1].numpy()
        np.save(output_dir / "bert_val_probs.npy", np.array(y_val_prob_local))
        np.save(output_dir / "bert_val_y.npy", np.array(y_val))
        np.save(output_dir / "bert_test_probs.npy", np.array(y_prob_test))
        np.save(output_dir / "bert_test_y.npy", np.array(y_test))
        # Standardized outputs for error analysis
        np.save(output_dir / "test_probabilities.npy", np.array(y_prob_test))
        np.save(output_dir / "test_labels.npy", np.array(y_test))
        np.save(output_dir / "test_predictions.npy", (np.array(y_prob_test) >= 0.5).astype(int))
        try:
            test_texts = list(map(str, X_test))
            (output_dir / "test_texts.json").write_text(
                json.dumps(test_texts, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass
    except Exception:
        pass


def bert_save_plots_and_fairness(trainer, val_ds, y_val, y_test, y_prob_test, output_dir: Path, groups: Optional[dict]):
    y_val_prob_local = trainer.predict(val_ds).predictions
    y_val_prob_local = torch.softmax(torch.tensor(y_val_prob_local), dim=-1)[:, 1].numpy()
    y_val_true = np.array(y_val)
    save_curves(y_val_true, y_val_prob_local, output_dir, "bert_val")
    save_confusion(y_val_true, (y_val_prob_local >= 0.5).astype(int), output_dir, "bert_val")
    save_curves(np.array(y_test), y_prob_test, output_dir, "bert_test")
    save_confusion(np.array(y_test), (y_prob_test >= 0.5).astype(int), output_dir, "bert_test")

    # Fairness outputs if group labels provided
    if groups:
        if groups.get("val") is not None:
            fm = fairness_metrics(np.array(y_val), np.array(y_val_prob_local), np.array(groups["val"]))
            with open(output_dir / "bert_val_fairness.json", "w") as f:
                json.dump(fm, f, indent=2)
        if groups.get("test") is not None:
            fm = fairness_metrics(np.array(y_test), np.array(y_prob_test), np.array(groups["test"]))
            with open(output_dir / "bert_test_fairness.json", "w") as f:
                json.dump(fm, f, indent=2)


def _svm_probability(clf, X):
    """Return probability-like scores for SVM pipelines, robust to missing predict_proba."""
    try:
        pp = clf.predict_proba(X)
        if pp.shape[1] == 2:
            return pp[:, 1]
        # Single probability column; assume class mapping not available
        return pp.ravel()
    except Exception:
        # Fallback to decision_function and min-max scale
        if hasattr(clf, "decision_function"):
            import numpy as _np

            dfc = clf.decision_function(X)
            dfc = (dfc - dfc.min()) / (dfc.max() - dfc.min() + 1e-8)
            return dfc
        # ultimate fallback: zeros
        import numpy as _np

        return _np.zeros(len(X))


def _evaluate_and_save(
    best,
    X,
    y,
    output_dir: Path,
    split_name: str,
    logger,
    groups: Optional[dict] = None,
):
    """Evaluate a trained pipeline, save metrics/probs/labels/plots/fairness/standard outputs."""
    import numpy as _np

    y_prob = _svm_probability(best, X)
    y_pred = (y_prob >= 0.5).astype(int)

    res = compute_metrics(y, y_prob)
    logger.info(f"SVM {split_name} metrics: {res}")

    out_metrics = output_dir / f"svm_{split_name}_metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(res.__dict__, f, indent=2)

    # Save probabilities and labels for downstream analyses / ensembles
    try:
        _np.save(output_dir / f"svm_{split_name}_probs.npy", _np.array(y_prob))
        _np.save(output_dir / f"svm_{split_name}_y.npy", _np.array(y))
    except Exception:
        pass

    # Standardized outputs for error analysis (for test split)
    try:
        if split_name == "test":
            _np.save(output_dir / "test_probabilities.npy", _np.array(y_prob))
            _np.save(output_dir / "test_labels.npy", _np.array(y))
            _np.save(output_dir / "test_predictions.npy", _np.array(y_pred))
    except Exception:
        pass

    # Fairness metrics if group labels provided
    if groups and split_name in groups and groups[split_name] is not None:
        gvec = groups[split_name]
        fm = fairness_metrics(_np.array(y), _np.array(y_prob), _np.array(gvec))
        with open(output_dir / f"svm_{split_name}_fairness.json", "w") as f:
            json.dump(fm, f, indent=2)

    # Plots
    save_curves(y, y_prob, output_dir, f"svm_{split_name}")
    save_confusion(y, y_pred, output_dir, f"svm_{split_name}")


def _run_svm_cv(model, X_train, y_train, n_splits: int, logger, output_dir: Path):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_cv = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]
        clf = model.build()
        clf.fit(X_tr, y_tr)
        y_prob = _svm_probability(clf, X_va)
        res = compute_metrics(y_va, y_prob)
        metrics_cv.append(res.__dict__)
        logger.info(f"Fold {fold}: F1={res.f1:.4f} AUC={res.roc_auc}")
    # Save CV summary
    out = output_dir / "svm_cv_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"folds": metrics_cv}, f, indent=2)


def _save_linear_svm_feature_importance(best, X_train, y_train, output_dir: Path, logger):
    """Train a linear SVM on vectorized features to export indicative n-grams."""
    try:
        import numpy as _np
        from sklearn.svm import LinearSVC

        features = best.named_steps["features"]
        X_train_vec = features.transform(X_train)
        lin = LinearSVC(class_weight="balanced")
        lin.fit(X_train_vec, y_train)

        # Map feature weights to n-grams
        if hasattr(features, "get_feature_names_out"):
            feat_names = features.get_feature_names_out()
        else:
            feat_names = _np.array([f"f_{i}" for i in range(X_train_vec.shape[1])])
        coefs = lin.coef_.ravel()
        order_pos = _np.argsort(-coefs)[:50]
        order_neg = _np.argsort(coefs)[:50]
        rows = [(feat_names[i], float(coefs[i]), "positive") for i in order_pos]
        rows += [(feat_names[i], float(coefs[i]), "negative") for i in order_neg]
        out_fp = output_dir / "svm_feature_importance.csv"
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(out_fp, "w") as f:
            f.write("feature,weight,polarity\n")
            for name, w, pol in rows:
                f.write(f"{name},{w},{pol}\n")
        logger.info(f"Saved feature importance to {out_fp}")
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")


def device_auto(prefer: Optional[str] = None) -> torch.device:
    """Select best available device with optional preference.

    Preference order by default on Apple Silicon should be MPS -> CUDA -> CPU.
    If prefer is provided ("mps"|"cuda"|"cpu"), try that first if available.
    """
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    # Try explicit preference first
    if prefer == "mps" and has_mps:
        return torch.device("mps")
    if prefer == "cuda" and has_cuda:
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")

    # Default heuristics: prefer MPS on Apple Silicon, otherwise CUDA
    if has_mps:
        return torch.device("mps")
    if has_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    temporal: bool = False,
    timestamp_col: str = "",
):
    if temporal and timestamp_col and timestamp_col in df.columns:
        train_df, val_df, test_df = temporal_split(
            df, timestamp_col, test_size=test_size, val_size=val_size
        )
        return (
            (train_df["text"].values, train_df["label"].values),
            (
                val_df["text"].values,
                val_df["label"].values,
            ),
            (test_df["text"].values, test_df["label"].values),
        )
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].values,
    )
    val_ratio_of_trainval = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_svm(
    train,
    val,
    test,
    output_dir: Path,
    logger,
    groups: Optional[dict] = None,
    use_cv: bool = False,
    n_splits: int = 5,
):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    # Allow CI/smoke runs to disable grid search via env var
    grid = os.getenv("DISABLE_GRID", "0") != "1"
    model = SVMBaseline(grid_search=grid)

    if use_cv:
        logger.info(f"Running Stratified {n_splits}-fold CV for SVM baseline...")
        _run_svm_cv(model, X_train, y_train, n_splits, logger, output_dir)
        # After CV, fit on full train set

    if not grid:
        logger.info("Fitting SVM without grid search (smoke mode)...")
        if len(np.unique(y_train)) < 2:
            logger.warning("Training data has a single class; using DummyClassifier for smoke run.")
            from sklearn.dummy import DummyClassifier
            from sklearn.pipeline import Pipeline as SKPipeline

            features = SVMBaseline().build().named_steps["features"]
            dummy = DummyClassifier(strategy="prior")
            best = SKPipeline([("features", features), ("clf", dummy)])
            best.fit(X_train, y_train)
        else:
            best = model.build()
            best.fit(X_train, y_train)
    else:
        logger.info("Fitting SVM with grid search...")
        gs = model.tune(X_train, y_train)
        best = gs.best_estimator_
        logger.info(f"Best params: {gs.best_params_}")

    # Post-hoc interpretability: train linear SVM on fixed TF-IDF features to extract feature importances
    _save_linear_svm_feature_importance(best, X_train, y_train, output_dir, logger)

    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        _evaluate_and_save(best, X, y, output_dir, split_name, logger, groups)


def run_bilstm(
    train,
    val,
    test,
    output_dir: Path,
    logger,
    max_len: int = 256,
    groups: Optional[dict] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    prefer_device: Optional[str] = None,
):
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader

    device = device_auto(prefer_device)
    logger.info(f"Using device for BiLSTM: {device}")

    # Simple whitespace tokenizer + vocabulary
    def build_vocab(texts, min_freq: int = 2, max_size: int = 50000):
        from collections import Counter

        counter = Counter()
        for t in texts:
            counter.update(str(t).split())
        vocab = {"<pad>": 0, "<unk>": 1}
        for tok, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(vocab) >= max_size:
                break
            vocab[tok] = len(vocab)
        return vocab

    def encode(text, vocab):
        return [vocab.get(tok, 1) for tok in str(text).split()][:max_len]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test
    vocab = build_vocab(X_train)

    def make_dataset(X, y):
        enc = [torch.tensor(encode(t, vocab), dtype=torch.long) for t in X]
        attn = [torch.ones_like(e, dtype=torch.bool) for e in enc]
        labels = [torch.tensor(int(lbl), dtype=torch.long) for lbl in y]
        return list(zip(enc, attn, labels))

    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    def collate(batch):
        ids, attn, labels = zip(*batch)
        ids_pad = pad_sequence(ids, batch_first=True, padding_value=0)
        attn_pad = pad_sequence(attn, batch_first=True, padding_value=0)
        labels_t = torch.stack(labels)
        return ids_pad[:, :max_len], attn_pad[:, :max_len], labels_t

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)

    # Configure model/training from config overrides
    hcfg = BiLSTMAttentionConfig(vocab_size=max(vocab.values()) + 1)
    if cfg_overrides and "bilstm" in cfg_overrides:
        b = cfg_overrides["bilstm"]
        for k in ["embedding_dim", "hidden_dim", "num_layers", "bidirectional", "dropout"]:
            if k in b:
                setattr(hcfg, k, b[k])
    model = BiLSTMAttention(hcfg).to(device)

    lr = 2e-4
    wd = 1e-2
    bs = 64
    num_epochs = 15
    early_patience = 3
    use_focal = False
    class_weights = None
    if cfg_overrides and "bilstm" in cfg_overrides:
        b = cfg_overrides["bilstm"]
        lr = float(b.get("learning_rate", lr))
        wd = float(b.get("weight_decay", wd))
        bs = int(b.get("batch_size", bs))
        num_epochs = int(b.get("num_epochs", num_epochs))
        early_patience = int(b.get("early_stopping_patience", early_patience))
        use_focal = bool(b.get("focal_loss", False))
        if b.get("class_weights", False):
            # compute simple inverse frequency weights only if both classes present
            counts = np.bincount(y_train)
            if counts.size == 2 or (counts > 0).sum() == 2:
                weights = counts.sum() / (2 * counts + 1e-8)
                class_weights = torch.tensor(weights, dtype=torch.float32)
            else:
                class_weights = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if use_focal:
        from suicide_detection.utils.losses import FocalLoss

        alpha = None
        if class_weights is not None:
            alpha = class_weights.to(device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

    # rebuild loaders with batch size from config
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate)

    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for ids, attn, labels in loader:
                ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
                logits = model(ids, attn)
                prob = torch.softmax(logits, dim=-1)[:, 1]
                ys.append(labels.cpu())
                ps.append(prob.cpu())
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        return compute_metrics(y_true, y_prob)

    # Helper to collect probs and labels (used by fairness and plots)
    def collect(loader):
        ys, ps = [], []
        with torch.no_grad():
            for ids, attn, labels in loader:
                ids, attn = ids.to(device), attn.to(device)
                outputs = model(ids, attn)
                prob = torch.softmax(outputs, dim=-1)[:, 1]
                ys.append(labels)
                ps.append(prob.cpu())
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        return y_true, y_prob

    def _train_bilstm(model):
        best_f1, patience, max_patience = 0.0, 0, early_patience
        for epoch in range(1, num_epochs + 1):
            model.train()
            for ids, attn, labels in train_loader:
                ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(ids, attn)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            val_res = evaluate(val_loader)
            logger.info(f"BiLSTM epoch {epoch} val F1={val_res.f1:.4f}")
            if val_res.f1 > best_f1 + 1e-4:
                best_f1 = val_res.f1
                patience = 0
                out_model = output_dir / "bilstm.pt"
                out_model.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_model)
            else:
                patience += 1
                if patience > max_patience:
                    break
        return model

    model = _train_bilstm(model)

    # Final eval
    res_val = evaluate(val_loader)
    res_test = evaluate(test_loader)

    # Save probabilities and labels for downstream analyses / ensembles
    try:
        yv, pv = collect(val_loader)
        yt, pt = collect(test_loader)
        np.save(output_dir / "bilstm_val_probs.npy", np.array(pv))
        np.save(output_dir / "bilstm_val_y.npy", np.array(yv))
        np.save(output_dir / "bilstm_test_probs.npy", np.array(pt))
        np.save(output_dir / "bilstm_test_y.npy", np.array(yt))
        # Standardized outputs for error analysis
        np.save(output_dir / "test_probabilities.npy", np.array(pt))
        np.save(output_dir / "test_labels.npy", np.array(yt))
        np.save(output_dir / "test_predictions.npy", (np.array(pt) >= 0.5).astype(int))
        try:
            # Save test texts
            test_texts = list(map(str, X_test))
            (output_dir / "test_texts.json").write_text(
                json.dumps(test_texts, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass
    except Exception:
        pass


    # Optional fairness outputs using provided groups
    if groups:
        if groups.get("val") is not None:
            yv, pv = collect(val_loader)
            fm = fairness_metrics(np.array(yv), np.array(pv), np.array(groups["val"]))
            with open(output_dir / "bilstm_val_fairness.json", "w") as f:
                json.dump(fm, f, indent=2)
        if groups.get("test") is not None:
            yt, pt = collect(test_loader)
            fm = fairness_metrics(np.array(yt), np.array(pt), np.array(groups["test"]))
            with open(output_dir / "bilstm_test_fairness.json", "w") as f:
                json.dump(fm, f, indent=2)

    # Save plots
    # For plots we need probabilities and predictions from loaders again
    yv, pv = collect(val_loader)
    yt, pt = collect(test_loader)
    save_curves(yv, pv, output_dir, "bilstm_val")
    save_confusion(yv, (pv >= 0.5).astype(int), output_dir, "bilstm_val")
    save_curves(yt, pt, output_dir, "bilstm_test")
    save_confusion(yt, (pt >= 0.5).astype(int), output_dir, "bilstm_test")

    # Attention heatmap visualization for first few validation samples
    def _save_bilstm_attention_visualizations():
        try:
            import matplotlib.pyplot as plt
            vis_dir = output_dir / "attention"
            vis_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            model.eval()
            with torch.no_grad():
                for ids, attn, _labels in val_loader:
                    ids, attn = ids.to(device), attn.to(device)
                    _ = model(ids, attn)
                    attn_w = getattr(model, "last_attn", None)
                    if attn_w is None:
                        break
                    attn_w = attn_w.cpu().numpy()
                    for i in range(min(attn_w.shape[0], 5 - count)):
                        plt.figure(figsize=(8, 2))
                        plt.imshow(attn_w[i][None, : ids.shape[1]], aspect="auto", cmap="viridis")
                        plt.colorbar()
                        plt.yticks([])
                        plt.xlabel("Token index")
                        plt.title("BiLSTM Attention Weights (sample {})".format(i))
                        plt.savefig(vis_dir / f"val_attn_{count+i}.png", dpi=150, bbox_inches="tight")
                        plt.close()
                    count += min(attn_w.shape[0], 5 - count)
                    if count >= 5:
                        break
        except Exception as e:
            logger.warning(f"Failed to save attention visualizations: {e}")

    _save_bilstm_attention_visualizations()

    for split, res in [("val", res_val), ("test", res_test)]:
        with open(output_dir / f"bilstm_{split}_metrics.json", "w") as f:
            json.dump(res.__dict__, f, indent=2)


def run_bert(
    train,
    val,
    test,
    output_dir: Path,
    logger,
    model_name: str = "mental/mental-bert-base-uncased",
    max_len: int = 256,
    groups: dict | None = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    prefer_device: Optional[str] = None,
):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    device = device_auto(prefer_device)
    logger.info(f"Using device for BERT: {device}")
    model, tokenizer = build_model_and_tokenizer(
        type("Cfg", (), {"model_name": model_name, "num_labels": 2})
    )

    train_ds = TextDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_ds = TextDataset(X_val, y_val, tokenizer, max_len=max_len)
    test_ds = TextDataset(X_test, y_test, tokenizer, max_len=max_len)

    # Defaults
    lr = 2e-5
    weight_decay = 0.01
    batch_size = 16
    num_epochs = 4
    warmup_ratio = 0.1
    grad_accum = 1
    fp16 = False
    bf16 = False
    early_patience = 2
    use_focal = False
    class_weights = None
    if cfg_overrides and "bert" in cfg_overrides:
        b = cfg_overrides["bert"]
        lr = float(b.get("learning_rate", lr))
        weight_decay = float(b.get("weight_decay", weight_decay))
        batch_size = int(b.get("batch_size", batch_size))
        num_epochs = int(b.get("num_epochs", num_epochs))
        warmup_ratio = float(b.get("warmup_ratio", warmup_ratio))
        grad_accum = int(b.get("gradient_accumulation_steps", grad_accum))
        fp16 = bool(b.get("fp16", fp16))
        bf16 = bool(b.get("bf16", bf16))
        early_patience = int(b.get("early_stopping_patience", early_patience))
        use_focal = bool(b.get("focal_loss", False))
        if b.get("class_weights", False):
            counts = np.bincount(y_train)
            if counts.size == 2 or (counts > 0).sum() == 2:
                weights = counts.sum() / (2 * counts + 1e-8)
                class_weights = torch.tensor(weights, dtype=torch.float32)
            else:
                class_weights = None
    # Mixed-precision selection based on device
    if str(device) == "cuda":
        fp16 = True if not fp16 and not bf16 else fp16
        bf16 = False if fp16 else bf16
    elif str(device) == "mps":
        # Keep full precision by default on MPS for stability
        fp16 = False
        bf16 = False

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=grad_accum,
        logging_steps=50,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
    )

    def hf_metrics(eval_pred):

        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        res = compute_metrics(labels, probs)
        return {
            "accuracy": res.accuracy,
            "precision": res.precision,
            "recall": res.recall,
            "f1": res.f1,
            "roc_auc": res.roc_auc if res.roc_auc is not None else float("nan"),
            "pr_auc": res.pr_auc if res.pr_auc is not None else float("nan"),
        }

    # Optional focal loss: fall back to standard CrossEntropy if Trainer doesn't accept custom loss
    if use_focal:
        from suicide_detection.utils.losses import FocalLoss

        alpha = class_weights.to(device) if class_weights is not None else None
        focal = FocalLoss(gamma=2.0, alpha=alpha)

        # We'll wrap the model's forward via a simple subclass if needed
        class ModelWithFocal(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, **inputs):
                labels = inputs.pop("labels", None)
                out = self.base(**inputs)
                if labels is not None:
                    loss = focal(out.logits, labels)
                    out.loss = loss
                return out

        model = ModelWithFocal(model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_patience)],
    )

    trainer.train()
    metrics_val = trainer.evaluate(eval_dataset=val_ds)
    preds = trainer.predict(test_ds)
    y_prob = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    res_test = compute_metrics(np.array(y_test), y_prob)

    def _save_bert_outputs():
        # Save probabilities and labels for downstream analyses / ensembles
        try:
            y_val_prob_local = trainer.predict(val_ds).predictions
            y_val_prob_local = torch.softmax(torch.tensor(y_val_prob_local), dim=-1)[:, 1].numpy()
            np.save(output_dir / "bert_val_probs.npy", np.array(y_val_prob_local))
            np.save(output_dir / "bert_val_y.npy", np.array(y_val))
            np.save(output_dir / "bert_test_probs.npy", np.array(y_prob))
            np.save(output_dir / "bert_test_y.npy", np.array(y_test))
            # Standardized outputs for error analysis
            np.save(output_dir / "test_probabilities.npy", np.array(y_prob))
            np.save(output_dir / "test_labels.npy", np.array(y_test))
            np.save(output_dir / "test_predictions.npy", (np.array(y_prob) >= 0.5).astype(int))
            try:
                test_texts = list(map(str, X_test))
                (output_dir / "test_texts.json").write_text(
                    json.dumps(test_texts, ensure_ascii=False, indent=2)
                )
            except Exception:
                pass
        except Exception:
            pass

    _save_bert_outputs()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "bert_val_metrics.json", "w") as f:
        json.dump(metrics_val, f, indent=2)
    with open(output_dir / "bert_test_metrics.json", "w") as f:
        json.dump(res_test.__dict__, f, indent=2)

    # Plots for BERT
    def _save_bert_plots_and_fairness():
        y_val_prob_local = trainer.predict(val_ds).predictions
        y_val_prob_local = torch.softmax(torch.tensor(y_val_prob_local), dim=-1)[:, 1].numpy()
        y_val_true = np.array(y_val)
        save_curves(y_val_true, y_val_prob_local, output_dir, "bert_val")
        save_confusion(y_val_true, (y_val_prob_local >= 0.5).astype(int), output_dir, "bert_val")
        save_curves(np.array(y_test), y_prob, output_dir, "bert_test")
        save_confusion(np.array(y_test), (y_prob >= 0.5).astype(int), output_dir, "bert_test")

        # Fairness outputs if group labels provided
        if groups:
            if groups.get("val") is not None:
                fm = fairness_metrics(np.array(y_val), np.array(y_val_prob_local), np.array(groups["val"]))
                with open(output_dir / "bert_val_fairness.json", "w") as f:
                    json.dump(fm, f, indent=2)
            if groups.get("test") is not None:
                fm = fairness_metrics(np.array(y_test), np.array(y_prob), np.array(groups["test"]))
                with open(output_dir / "bert_test_fairness.json", "w") as f:
                    json.dump(fm, f, indent=2)

    _save_bert_plots_and_fairness()


def _load_data(args, logger):
    """Load and prepare train/val/test splits and optional group vectors."""
    df = None
    train = val = test = None
    group_vectors = {"train": None, "val": None, "test": None}

    if args.dataset:
        split_dir = Path(f"data/{args.dataset}/splits")
        train_df = pd.read_csv(split_dir / "train.csv")
        val_df = pd.read_csv(split_dir / "val.csv")
        test_df = pd.read_csv(split_dir / "test.csv")
        train = (train_df["text"].values, train_df["label"].values)
        val = (val_df["text"].values, val_df["label"].values)
        test = (test_df["text"].values, test_df["label"].values)
    elif args.train_path and args.val_path and args.test_path:
        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.val_path)
        test_df = pd.read_csv(args.test_path)
        train = (train_df["text"].values, train_df["label"].values)
        val = (val_df["text"].values, val_df["label"].values)
        test = (test_df["text"].values, test_df["label"].values)
    else:
        data_path = Path(args.data_path)
        df = load_dataset_secure(data_path)
        if df["label"].dtype != int and df["label"].dtype != bool:
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"].astype(str))
        train, val, test = prepare_splits(
            df,
            temporal=args.temporal_split,
            timestamp_col=args.timestamp_col,
        )
        if args.group_col and args.group_col in df.columns:
            group_vectors = {"train": None, "val": None, "test": None}
    return train, val, test, group_vectors


def _run_selected_model(args, train, val, test, output_dir: Path, logger, cfg_all, group_vectors, use_cv, n_splits, ml_enabled):
    """Dispatch to model-specific training and handle optional MLflow logging."""
    def _log_metrics(split_prefix: str, metrics_path: Path):
        if ml_enabled and metrics_path.exists():
            import json as _json
            m = _json.loads(metrics_path.read_text())
            for k, v in m.items() if isinstance(m, dict) else []:
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{split_prefix}_{k}", float(v))

    if args.model == "svm":
        run_svm(train, val, test, output_dir, logger, groups=group_vectors, use_cv=use_cv, n_splits=n_splits)
        _log_metrics("val", output_dir / "svm_val_metrics.json")
        _log_metrics("test", output_dir / "svm_test_metrics.json")
    elif args.model == "bilstm":
        run_bilstm(train, val, test, output_dir, logger, groups=group_vectors, cfg_overrides=cfg_all, prefer_device=args.prefer_device)
        _log_metrics("val", output_dir / "bilstm_val_metrics.json")
        _log_metrics("test", output_dir / "bilstm_test_metrics.json")
    elif args.model == "bert":
        run_bert(train, val, test, output_dir, logger, model_name=args.bert_model_name, groups=group_vectors, cfg_overrides=cfg_all, prefer_device=args.prefer_device)
        _log_metrics("val", output_dir / "bert_val_metrics.json")
        _log_metrics("test", output_dir / "bert_test_metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["svm", "bilstm", "bert"], required=True)
    ap.add_argument("--data_path", default="data/raw/dataset.csv")
    ap.add_argument("--train_path", default=None)
    ap.add_argument("--val_path", default=None)
    ap.add_argument("--test_path", default=None)
    ap.add_argument("--output_dir", default="results/model_outputs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--default_config", default=None)
    ap.add_argument("--temporal_split", action="store_true")
    ap.add_argument("--timestamp_col", default="")
    ap.add_argument(
        "--group_col", default=None, help="Optional demographic group column for fairness analysis"
    )
    ap.add_argument(
        "--dataset",
        choices=["kaggle", "mendeley"],
        default=None,
        help="If set, will use data/<dataset>/splits train/val/test CSVs",
    )
    ap.add_argument(
        "--bert_model_name",
        default="bert-base-uncased",
        help="HuggingFace model name for BERT training",
    )
    ap.add_argument(
        "--prefer_device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Prefer this device if available (default: MPS->CUDA->CPU)",
    )
    args = ap.parse_args()

    # Load configs if provided
    cfg_default: Dict[str, Any] = {}
    cfg_model: Dict[str, Any] = {}
    if args.default_config and Path(args.default_config).exists():
        try:
            cfg_default = yaml.safe_load(Path(args.default_config).read_text()) or {}
        except Exception:
            cfg_default = {}
    if args.config and Path(args.config).exists():
        try:
            cfg_model = yaml.safe_load(Path(args.config).read_text()) or {}
        except Exception:
            cfg_model = {}
    cfg_all = {**cfg_default, **cfg_model}

    # Configure logger (optionally to file)
    log_file = None
    if cfg_all.get("logging", {}).get("log_to_file", False):
        log_file = Path(cfg_all.get("logging", {}).get("file_path", "results/project.log"))
    logger = get_logger("train", log_file=log_file)

    # Set seed
    set_global_seed(int(cfg_all.get("seed", 42)))

    # Optional MLflow setup
    mlflow_cfg_path = Path("configs/mlflow.yaml")
    ml_enabled = False
    if mlflow_cfg_path.exists():
        try:
            cfg = yaml.safe_load(mlflow_cfg_path.read_text())
            ml_enabled = cfg.get("enabled", False)
            if ml_enabled:
                tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(cfg.get("experiment_name", "suicide_detection_research"))
        except Exception as e:
            logger.warning(f"Failed to read MLflow config: {e}")

    # Prepare data
    train, val, test, group_vectors = _load_data(args, logger)

    output_dir = Path(args.output_dir)

    # Evaluation config
    eval_cfg = cfg_all.get("evaluation", {})
    use_cv = bool(eval_cfg.get("use_stratified_kfold", False))
    n_splits = int(eval_cfg.get("n_splits", 5))

    # Train and optionally log with MLflow
    if ml_enabled:
        with mlflow.start_run(run_name=args.model):
            mlflow.log_params({"model": args.model})
            if args.temporal_split:
                mlflow.log_param("temporal_split", True)
                mlflow.log_param("timestamp_col", args.timestamp_col)
            _run_selected_model(
                args,
                train,
                val,
                test,
                output_dir,
                logger,
                cfg_all,
                group_vectors,
                use_cv,
                n_splits,
                ml_enabled=True,
            )
    else:
        _run_selected_model(
            args,
            train,
            val,
            test,
            output_dir,
            logger,
            cfg_all,
            group_vectors,
            use_cv,
            n_splits,
            ml_enabled=False,
        )


if __name__ == "__main__":
    main()
