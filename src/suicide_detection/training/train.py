from pathlib import Path
import argparse
import json
from typing import Dict, Any
import yaml
import mlflow

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from suicide_detection.utils.seed import set_global_seed
from suicide_detection.utils.logging import get_logger
from suicide_detection.data_processing.load import load_dataset_secure
from suicide_detection.data_processing.splitting import temporal_split
from suicide_detection.models.svm_baseline import SVMBaseline
from suicide_detection.models.bilstm_attention import BiLSTMAttention, BiLSTMAttentionConfig
from suicide_detection.models.bert_model import build_model_and_tokenizer, TextDataset
from suicide_detection.evaluation.metrics import compute_metrics
from suicide_detection.evaluation.plots import save_curves, save_confusion


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
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
        train_df, val_df, test_df = temporal_split(df, timestamp_col, test_size=test_size, val_size=val_size)
        return (train_df["text"].values, train_df["label"].values), (
            val_df["text"].values,
            val_df["label"].values,
        ), (test_df["text"].values, test_df["label"].values)
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


def run_svm(train, val, test, output_dir: Path, logger):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    # SMOTE on training set only
    sm = SMOTE(random_state=42)
    Xy_train = pd.DataFrame({"text": X_train, "label": y_train})
    # For SMOTE, we need vectorized features first; workaround: use TF-IDF then SMOTE then SVM? For simplicity,
    # we will rely on class_weight in SVC if imbalance is severe. Alternatively, implement pipeline-level Imbalance.

    model = SVMBaseline(grid_search=True)
    logger.info("Fitting SVM with grid search...")
    gs = model.tune(X_train, y_train)
    best = gs.best_estimator_
    logger.info(f"Best params: {gs.best_params_}")

    # Post-hoc interpretability: train linear SVM on fixed TF-IDF features to extract feature importances
    try:
        from sklearn.svm import LinearSVC
        import numpy as np
        features = best.named_steps["features"]
        X_train_vec = features.transform(X_train)
        lin = LinearSVC(class_weight="balanced")
        lin.fit(X_train_vec, y_train)
        # Map feature weights to n-grams
        feat_names = []
        # Attempt to get names from FeatureUnion
        if hasattr(features, "get_feature_names_out"):
            feat_names = features.get_feature_names_out()
        else:
            feat_names = np.array([f"f_{i}" for i in range(X_train_vec.shape[1])])
        coefs = lin.coef_.ravel()
        order_pos = np.argsort(-coefs)[:50]
        order_neg = np.argsort(coefs)[:50]
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

    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob = best.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        res = compute_metrics(y, y_prob)
        logger.info(f"SVM {split_name} metrics: {res}")
        out = output_dir / f"svm_{split_name}_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res.__dict__, f, indent=2)
        save_curves(y, y_prob, output_dir, f"svm_{split_name}")
        save_confusion(y, y_pred, output_dir, f"svm_{split_name}")


def run_bilstm(train, val, test, output_dir: Path, logger, max_len: int = 256):
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    device = device_auto()

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

    cfg = BiLSTMAttentionConfig(vocab_size=max(vocab.values()) + 1)
    model = BiLSTMAttention(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

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

    best_f1, patience, max_patience = 0.0, 0, 3
    for epoch in range(1, 16):
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

    # Final eval
    res_val = evaluate(val_loader)
    res_test = evaluate(test_loader)
    # Save plots
    # For plots we need probabilities and predictions from loaders again
    def collect(loader):
        ys, ps = [], []
        with torch.no_grad():
            for ids, attn, labels in loader:
                ids, attn = ids.to(device), attn.to(device)
                logits = model(ids, attn)
                prob = torch.softmax(logits, dim=-1)[:,1]
                ys.append(labels)
                ps.append(prob.cpu())
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        return y_true, y_prob
    yv, pv = collect(val_loader)
    yt, pt = collect(test_loader)
    save_curves(yv, pv, output_dir, "bilstm_val")
    save_confusion(yv, (pv>=0.5).astype(int), output_dir, "bilstm_val")
    save_curves(yt, pt, output_dir, "bilstm_test")
    save_confusion(yt, (pt>=0.5).astype(int), output_dir, "bilstm_test")

    for split, res in [("val", res_val), ("test", res_test)]:
        with open(output_dir / f"bilstm_{split}_metrics.json", "w") as f:
            json.dump(res.__dict__, f, indent=2)


def run_bert(train, val, test, output_dir: Path, logger, model_name: str = "mental/mental-bert-base-uncased", max_len: int = 256):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    device = device_auto()
    model, tokenizer = build_model_and_tokenizer(type("Cfg", (), {"model_name": model_name, "num_labels": 2}))

    train_ds = TextDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_ds = TextDataset(X_val, y_val, tokenizer, max_len=max_len)
    test_ds = TextDataset(X_test, y_test, tokenizer, max_len=max_len)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        save_total_limit=2,
    )

    def hf_metrics(eval_pred):
        import numpy as np

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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics_val = trainer.evaluate(eval_dataset=val_ds)
    preds = trainer.predict(test_ds)
    y_prob = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    res_test = compute_metrics(np.array(y_test), y_prob)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "bert_val_metrics.json", "w") as f:
        json.dump(metrics_val, f, indent=2)
    with open(output_dir / "bert_test_metrics.json", "w") as f:
        json.dump(res_test.__dict__, f, indent=2)

    # Plots for BERT
    y_val_prob = trainer.predict(val_ds).predictions
    y_val_prob = torch.softmax(torch.tensor(y_val_prob), dim=-1)[:,1].numpy()
    y_val_true = np.array(y_val)
    save_curves(y_val_true, y_val_prob, output_dir, "bert_val")
    save_confusion(y_val_true, (y_val_prob>=0.5).astype(int), output_dir, "bert_val")
    save_curves(np.array(y_test), y_prob, output_dir, "bert_test")
    save_confusion(np.array(y_test), (y_prob>=0.5).astype(int), output_dir, "bert_test")


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
    args = ap.parse_args()

    logger = get_logger("train")
    set_global_seed(42)

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

    data_path = Path(args.data_path)
    df = load_dataset_secure(data_path)

    # Ensure binary labels 0/1
    if df["label"].dtype != int and df["label"].dtype != bool:
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"].astype(str))

    # If explicit splits provided, use them; otherwise create splits
    if args.train_path and args.val_path and args.test_path:
        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.val_path)
        test_df = pd.read_csv(args.test_path)
        train = (train_df["text"].values, train_df["label"].values)
        val = (val_df["text"].values, val_df["label"].values)
        test = (test_df["text"].values, test_df["label"].values)
    else:
        train, val, test = prepare_splits(
            df,
            temporal=args.temporal_split,
            timestamp_col=args.timestamp_col,
        )

    output_dir = Path(args.output_dir)

    if ml_enabled:
        with mlflow.start_run(run_name=args.model):
            mlflow.log_params({"model": args.model})
            if args.temporal_split:
                mlflow.log_param("temporal_split", True)
                mlflow.log_param("timestamp_col", args.timestamp_col)
            # Execute training and log metrics after each split
            if args.model == "svm":
                run_svm(train, val, test, output_dir, logger)
                # Log metrics if produced
                for split in ["val", "test"]:
                    fp = output_dir / f"svm_{split}_metrics.json"
                    if fp.exists():
                        import json as _json
                        m = _json.loads(fp.read_text())
                        for k, v in m.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"{split}_{k}", float(v))
            elif args.model == "bilstm":
                run_bilstm(train, val, test, output_dir, logger)
                for split in ["val", "test"]:
                    fp = output_dir / f"bilstm_{split}_metrics.json"
                    if fp.exists():
                        import json as _json
                        m = _json.loads(fp.read_text())
                        for k, v in m.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"{split}_{k}", float(v))
            elif args.model == "bert":
                run_bert(train, val, test, output_dir, logger)
                for split in ["val", "test"]:
                    fp = output_dir / f"bert_{split}_metrics.json"
                    if fp.exists():
                        import json as _json
                        m = _json.loads(fp.read_text())
                        # HuggingFace eval writes keys differently; log standard ones if present
                        for k, v in (m.items() if isinstance(m, dict) else []):
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"{split}_{k}", float(v))
    else:
        if args.model == "svm":
            run_svm(train, val, test, output_dir, logger)
        elif args.model == "bilstm":
            run_bilstm(train, val, test, output_dir, logger)
        elif args.model == "bert":
            run_bert(train, val, test, output_dir, logger)


if __name__ == "__main__":
    main()

