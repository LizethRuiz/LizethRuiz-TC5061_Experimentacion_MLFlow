# TC5061 · Experimentación con MLflow

Predicción de rotación de empleados (`Attrition`) con regresión logística sobre el dataset
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
El entrenamiento está en un script parametrizado (`train.py`), y cada corrida queda registrada en MLflow.

**Autora:** Lizeth Ruiz (A01840361)

## Contenido

| Archivo | Descripción |
|---|---|
| `train.py` | Carga, limpieza, entrenamiento, evaluación y registro en MLflow |
| `requirements.txt` | Dependencias con las versiones usadas |
| `run_experiments.sh` | Reproduce las 9 corridas del experimento |
| `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` | Dataset |
| `TC5061_EDA_RuizLizeth.ipynb` | Notebook de exploración de datos (solo EDA) |

## Instalación

Probado con Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Con los valores por defecto, que corresponden al baseline:

```bash
python train.py
```

Cambiando hiperparámetros sin editar el código:

```bash
python train.py --C 0.1 --penalty l2 --class-weight balanced --threshold 0.5 --seed 42
```

Para ver todas las opciones, usa `python train.py --help`.

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización |
| `--penalty` | `l2` | `l2`, `l1` o `elasticnet` (el *solver* se elige automáticamente) |
| `--l1-ratio` | `0.5` | Mezcla L1/L2, solo con `elasticnet` |
| `--class-weight` | `none` | `none` o `balanced` |
| `--threshold` | `0.5` | Umbral de probabilidad para predecir `Attrition = 1` |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--seed` | `42` | Semilla aleatoria |
| `--experiment-name` | `TC5061-attrition-logreg-hiperparametros` | Experimento de MLflow |
| `--run-name` | automático | Nombre de la corrida |
| `--tracking-uri` | `sqlite:///mlflow.db` | Backend de MLflow |

Para reproducir las 9 corridas del reporte:

```bash
bash run_experiments.sh
```

## Ver resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abre http://localhost:5000, entra al experimento `TC5061-attrition-logreg-hiperparametros`, selecciona
las corridas y usa **Compare**. En macOS el puerto 5000 puede estar ocupado por AirPlay; si es así, usa otro, como `--port 5055`.

Cada corrida registra:

- **Parámetros** (`log_param`): C, penalty, class_weight, threshold, max_iter, test_size, seed y solver.
- **Métricas** (`log_metric`): accuracy, precision, recall, F1 y ROC-AUC en prueba y en entrenamiento, más la accuracy del `DummyClassifier`.
- **Artefactos** (`log_artifact`): `plots/evaluacion.png` (matriz de confusión y curva ROC), `metrics/metrics.json` y `model_info/coeficientes.csv`.
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo, con firma y ejemplo de entrada.

## Reproducibilidad

La semilla fija `random`, `numpy` y `PYTHONHASHSEED`, y se pasa como `random_state` a la partición y al modelo.
Cada corrida guarda el tag `predictions_sha256`, un hash de las probabilidades predichas. Dos ejecuciones con los mismos
argumentos producen el mismo hash y las mismas métricas.
