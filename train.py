"""
train.py — Entrenamiento y evaluación del modelo de rotación de empleados (IBM HR Attrition)
con seguimiento de experimentos en MLflow.
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre desde la terminal

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
DEFAULT_TRACKING = "sqlite:///" + os.path.join(HERE, "mlflow.db")
DEFAULT_EXPERIMENT = "TC5061-attrition-logreg-hiperparametros"

# Solver compatible con cada penalización de la regresión logística
SOLVER_BY_PENALTY = {"l2": "lbfgs", "l1": "liblinear", "elasticnet": "saga"}


# ---------------------------------------------------------------------------
# 1. Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Define los hiperparámetros configurables. Todos tienen valor por defecto."""
    p = argparse.ArgumentParser(description="Entrena una regresión logística para predecir Attrition y la registra en MLflow.")
    # Hiperparámetros del modelo
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización (menor = más regularización). Default: 1.0")
    p.add_argument("--penalty", choices=["l2", "l1", "elasticnet"], default="l2",
                   help="Tipo de regularización. Default: l2")
    p.add_argument("--l1-ratio", type=float, default=0.5,
                   help="Mezcla L1/L2, solo se usa con --penalty elasticnet. Default: 0.5")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="'balanced' compensa el desbalance de clases (16%% positivos). Default: none")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Umbral de probabilidad para predecir Attrition=1. Default: 0.5")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Iteraciones máximas del optimizador. Default: 1000")
    # Datos y reproducibilidad
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba. Default: 0.2")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global. Default: 42")
    p.add_argument("--data-path", default=DEFAULT_DATA, help="Ruta al CSV del dataset.")
    # MLflow
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT, help="Experimento de MLflow.")
    p.add_argument("--run-name", default=None, help="Nombre de la corrida (se genera uno si se omite).")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING, help="Backend de MLflow. Default: ./mlflow.db")
    return p.parse_args(argv)


def set_seed(seed):
    """Fija las semillas de las fuentes de aleatoriedad usadas (Python, NumPy, hash)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 2. Carga y limpieza (mismas decisiones del Paso 3 del notebook)
# ---------------------------------------------------------------------------
ORDINAL_RANGES = {
    "Education": (1, 5), "EnvironmentSatisfaction": (1, 4), "JobInvolvement": (1, 4),
    "JobLevel": (1, 5), "JobSatisfaction": (1, 4), "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4), "StockOptionLevel": (0, 3), "WorkLifeBalance": (1, 4),
}


def load_data(path):
    """Lee el CSV. 'utf-8-sig' quita el BOM de la primera columna."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_data(df_raw):
    """Elimina constantes e ID, convierte el objetivo a 0/1 y valida los rangos ordinales."""
    df = df_raw.copy()
    constants = [c for c in df.columns if df[c].nunique() == 1]
    dropped = constants + ["EmployeeNumber"]
    df = df.drop(columns=dropped)

    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    df[cat_cols] = df[cat_cols].astype("category")

    for col, (lo, hi) in ORDINAL_RANGES.items():
        assert df[col].between(lo, hi).all(), f"{col} fuera de rango"
    return df, dropped


def split_data(df, test_size, seed):
    """Partición estratificada en entrenamiento y prueba."""
    X = df.drop(columns="Attrition")
    y = df["Attrition"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------------------
# 3. Modelo
# ---------------------------------------------------------------------------
def build_pipeline(X, args):
    """Pipeline = escalado + one-hot (ajustados solo con train) + regresión logística."""
    num_features = X.select_dtypes("number").columns.tolist()
    cat_features = X.select_dtypes("category").columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])
    clf_kwargs = dict(
        C=args.C,
        penalty=args.penalty,
        solver=SOLVER_BY_PENALTY[args.penalty],
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    if args.penalty == "elasticnet":
        clf_kwargs["l1_ratio"] = args.l1_ratio
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(**clf_kwargs)),
    ])


def train(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# 4. Evaluación
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_score, threshold):
    """Métricas de clasificación binaria a partir de probabilidades y un umbral."""
    y_hat = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "roc_auc": roc_auc_score(y_true, y_score),
    }, y_hat


def plot_evaluation(y_test, y_pred, y_score, path):
    """Matriz de confusión y curva ROC lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No", "Yes"],
                                            cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Matriz de confusión (test)")
    RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[1], name="Regresión Logística")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.5)")
    axes[1].legend()
    axes[1].set_title("Curva ROC (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def coefficients_table(pipeline):
    """Coeficientes del modelo ordenados por magnitud."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"feature": names, "coef": coefs})
    return table.reindex(table["coef"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# 5. Orquestación + MLflow
# ---------------------------------------------------------------------------
def default_run_name(args):
    name = f"C={args.C}-{args.penalty}-cw={args.class_weight}-thr={args.threshold}"
    return name if args.seed == 42 else f"{name}-seed={args.seed}"


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    # Datos
    df, dropped = clean_data(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    # Entrenamiento
    model = train(build_pipeline(X_train, args), X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    # Evaluación
    score_test = model.predict_proba(X_test)[:, 1]
    score_train = model.predict_proba(X_train)[:, 1]
    m_test, y_pred = compute_metrics(y_test, score_test, args.threshold)
    m_train, _ = compute_metrics(y_train, score_train, args.threshold)
    m_dummy, _ = compute_metrics(y_test, dummy.predict_proba(X_test)[:, 1], 0.5)
    # Huella de las predicciones: dos corridas con los mismos argumentos deben coincidir
    pred_hash = hashlib.sha256(np.round(score_test, 10).tobytes()).hexdigest()[:16]

    # Registro en MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name or default_run_name(args)) as run:
        # Hiperparámetros (log_param uno por uno)
        for key in ["C", "penalty", "l1_ratio", "class_weight", "threshold",
                    "max_iter", "test_size", "seed"]:
            value = getattr(args, key)
            if key == "l1_ratio" and args.penalty != "elasticnet":
                continue
            mlflow.log_param(key, value)
        mlflow.log_param("solver", SOLVER_BY_PENALTY[args.penalty])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Métricas (log_metric una por una)
        for k, v in m_test.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in m_train.items():
            mlflow.log_metric(f"train_{k}", v)
        mlflow.log_metric("dummy_test_accuracy", m_dummy["accuracy"])
        n_nonzero = int((model.named_steps["classifier"].coef_ != 0).sum())
        mlflow.log_metric("n_coef_nonzero", n_nonzero)

        mlflow.set_tags({"curso": "TC5061", "autor": "Lizeth Ruiz",
                         "modelo": "LogisticRegression", "predictions_sha256": pred_hash})

        # Artefactos: gráfica de evaluación, archivo de métricas y coeficientes
        with tempfile.TemporaryDirectory() as tmp:
            eval_png = os.path.join(tmp, "evaluacion.png")
            plot_evaluation(y_test, y_pred, score_test, eval_png)
            mlflow.log_artifact(eval_png, artifact_path="plots")

            metrics_json = os.path.join(tmp, "metrics.json")
            with open(metrics_json, "w") as f:
                json.dump({"args": vars(args), "test": m_test, "train": m_train,
                           "dummy_test": m_dummy, "predictions_sha256": pred_hash,
                           "dropped_columns": dropped}, f, indent=2, default=str)
            mlflow.log_artifact(metrics_json, artifact_path="metrics")

            coefs_csv = os.path.join(tmp, "coeficientes.csv")
            coefficients_table(model).to_csv(coefs_csv, index=False)
            mlflow.log_artifact(coefs_csv, artifact_path="model_info")

        # Modelo entrenado (pipeline completo con firma)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.compose._column_transformer._RemainderColsList"],
        )

        print(f"Run: {run.info.run_name}  ({run.info.run_id})")
        print("Test: " + "  ".join(f"{k}={v:.4f}" for k, v in m_test.items()))
        print(f"predictions_sha256={pred_hash}")


if __name__ == "__main__":
    main()
