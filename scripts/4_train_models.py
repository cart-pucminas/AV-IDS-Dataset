#!/usr/bin/env python3
"""
Etapa 4 — Treinamento e avaliação dos modelos de detecção de intrusão

Este script carrega o dataset consolidado (combined_can_dataset.csv),
executa o pré-processamento, aplica balanceamento híbrido no conjunto de
treinamento e treina quatro algoritmos de classificação:

  1. Árvore de Decisão  (Decision Tree)
  2. Floresta Aleatória (Random Forest)
  3. XGBoost
  4. CNN 1D (Rede Neural Convolucional 1D)

Balanceamento do conjunto de treinamento (híbrido):
  - Oversampling via SMOTE: gera instâncias sintéticas das classes
    minoritárias por interpolação entre vizinhos mais próximos.
  - Undersampling via RandomUnderSampler: reduz a classe majoritária
    (DoS) para evitar que o treino seja dominado por ela.
  Ambas as técnicas são aplicadas em sequência (Pipeline) APENAS no
  conjunto de treinamento; o conjunto de teste permanece intacto.

A otimização de hiperparâmetros é feita automaticamente com o framework
Optuna, usando busca bayesiana (TPE) para maximizar o F1-score macro.

Divisão dos dados:
  - 80 % treinamento (por cenário completo, sem data leakage)
  - 20 % teste (distribuição original, sem balanceamento)

Saída:
  - Relatório de classificação (precisão, recall, F1-score, suporte)
    impresso no console para cada modelo.
  - Modelos salvos em 'models/' (Scikit-learn → .joblib, CNN → .keras).

Requisitos:
    pip install pandas numpy scikit-learn imbalanced-learn xgboost
               tensorflow optuna joblib
"""

import gc
import logging
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATASET_PATH  = "data/combined_can_dataset.csv.gz"
MODELS_DIR    = "models"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OPTUNA_TRIALS = 20

FEATURE_COLS  = [f'data_{i}' for i in range(8)] + ['arbitration_id', 'dlc']
TARGET_COL    = 'attack_type'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# 1. Carregamento e pré-processamento
# ---------------------------------------------------------------------------

def load_data(path: str):
    """
    Carrega o dataset, codifica variáveis categóricas e separa features/target.

    road_type e climate são codificadas via label encoding determinístico.
    IDs CAN e bytes de payload são mantidos em representação inteira.
    """
    log.info(f"Carregando dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Total de registros: {len(df):,}")

    # Codificar variáveis categóricas de contexto
    for col in ['road_type', 'climate']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Adicionar road_type e climate às features se disponíveis
    extra = [c for c in ['road_type', 'climate'] if c in df.columns]
    feat_cols = FEATURE_COLS + extra

    df = df.dropna(subset=feat_cols + [TARGET_COL])

    X = df[feat_cols].astype(np.int32)
    y = df[TARGET_COL]

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    log.info(f"Classes: {list(le_target.classes_)}")
    log.info(f"Distribuição original:\n{y.value_counts().to_string()}")

    return X, y_enc, list(le_target.classes_)


# ---------------------------------------------------------------------------
# 2. Divisão treino/teste por cenário (evita data leakage)
# ---------------------------------------------------------------------------

def split_by_scenario(X: pd.DataFrame, y: np.ndarray, df_full: pd.DataFrame,
                       test_size: float = 0.20, random_state: int = 42):
    """
    Divide em treino/teste agrupando cenários completos, não registros
    individuais. Isso evita data leakage entre registros do mesmo cenário.
    """
    # Identificar cenários únicos e alocar 20% deles ao teste
    scenarios = df_full[['road_type', 'climate', 'attack_type']].drop_duplicates()
    n_test_scenarios = max(1, int(len(scenarios) * test_size))

    rng = np.random.default_rng(random_state)
    test_idx = rng.choice(len(scenarios), size=n_test_scenarios, replace=False)
    test_scenarios = scenarios.iloc[test_idx]

    mask_test = pd.Series(False, index=df_full.index)
    for _, row in test_scenarios.iterrows():
        cond = True
        for col in row.index:
            cond = cond & (df_full[col] == row[col])
        mask_test = mask_test | cond

    X_train = X[~mask_test]
    X_test  = X[mask_test]
    y_train = y[~mask_test]
    y_test  = y[mask_test]

    log.info(f"Treino: {len(X_train):,}  |  Teste: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Balanceamento híbrido (apenas no treino)
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Aplica balanceamento híbrido ao conjunto de treinamento:

      1. SMOTE (oversampling): gera instâncias sintéticas das classes
         minoritárias por interpolação entre os k vizinhos mais próximos
         no espaço de características.
      2. RandomUnderSampler (undersampling): reduz aleatoriamente a classe
         majoritária (DoS), evitando que o treino seja dominado por ela e
         reduzindo o custo computacional.

    As duas técnicas são encadeadas em um Pipeline e aplicadas APENAS ao
    conjunto de treinamento. O conjunto de teste nunca é modificado.
    """
    log.info("Aplicando balanceamento híbrido (SMOTE + RandomUnderSampler)...")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state, n_jobs=-1)),
        ('under', RandomUnderSampler(random_state=random_state)),
    ])
    X_bal, y_bal = pipeline.fit_resample(X_train, y_train)
    log.info(f"Treino após balanceamento: {len(X_bal):,} registros")
    return X_bal, y_bal


# ---------------------------------------------------------------------------
# 4. Otimização de hiperparâmetros com Optuna
# ---------------------------------------------------------------------------

def optimize(X, y, model_type: str, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Busca bayesiana de hiperparâmetros usando Tree-structured Parzen Estimator."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        if model_type == "dt":
            clf = DecisionTreeClassifier(
                max_depth         = trial.suggest_int('max_depth', 5, 50),
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
                criterion         = trial.suggest_categorical('criterion', ['gini', 'entropy']),
                random_state      = RANDOM_STATE,
            )
        elif model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators    = trial.suggest_int('n_estimators', 50, 300),
                max_depth       = trial.suggest_int('max_depth', 10, 50),
                min_samples_leaf= trial.suggest_int('min_samples_leaf', 1, 10),
                n_jobs          = -1,
                random_state    = RANDOM_STATE,
            )
        elif model_type == "xgb":
            clf = xgb.XGBClassifier(
                n_estimators  = trial.suggest_int('n_estimators', 100, 500),
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                max_depth     = trial.suggest_int('max_depth', 3, 15),
                subsample     = trial.suggest_float('subsample', 0.6, 1.0),
                tree_method   = 'hist',
                random_state  = RANDOM_STATE,
            )
        else:
            raise ValueError(f"model_type '{model_type}' desconhecido.")

        scores = cross_val_score(clf, X, y, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"  Melhores parâmetros ({model_type}): {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# 5. Arquitetura CNN 1D
# ---------------------------------------------------------------------------

def build_cnn(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    CNN 1D para detecção de intrusão em sequências de bytes CAN.

    A mensagem CAN é tratada como uma sequência estruturada de dimensão
    input_dim. Filtros convolucionais deslizam sobre os bytes, capturando
    relações entre posições vizinhas — por exemplo, os dois bytes de um
    inteiro de 16 bits do RPM.

    Arquitetura:
        Conv1D(64, k=3) → BN → MaxPool
        Conv1D(128, k=3) → BN → GlobalAvgPool
        Dense(128) → Dropout(0.4) → Softmax
    """
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Reshape((input_dim, 1))(inp)
    x   = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer = 'adam',
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # — Carregar dados —
    df_full = pd.read_csv(DATASET_PATH, low_memory=False)
    X, y, class_names = load_data(DATASET_PATH)
    num_classes = len(class_names)

    # — Dividir por cenário (sem data leakage) —
    X_train_raw, X_test, y_train_raw, y_test = split_by_scenario(
        X, y, df_full, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # — Balancear com SMOTE + RandomUnderSampler (APENAS treino) —
    X_train, y_train = apply_smote(X_train_raw, y_train_raw, RANDOM_STATE)
    del X_train_raw, y_train_raw
    gc.collect()

    # — Árvore de Decisão —
    log.info("Treinando Árvore de Decisão...")
    best_dt = optimize(X_train, y_train, "dt")
    dt = DecisionTreeClassifier(**best_dt, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree.joblib"))

    # — Random Forest —
    log.info("Treinando Random Forest...")
    best_rf = optimize(X_train, y_train, "rf")
    rf = RandomForestClassifier(**best_rf, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))

    # — XGBoost —
    log.info("Treinando XGBoost...")
    best_xgb = optimize(X_train, y_train, "xgb")
    xgb_clf = xgb.XGBClassifier(
        **best_xgb, tree_method='hist', random_state=RANDOM_STATE)
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "xgboost.joblib"))

    # — CNN 1D —
    log.info("Treinando CNN 1D...")
    cnn = build_cnn(X_train.shape[1], num_classes)
    es  = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduce = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    cnn.fit(
        X_train, y_train,
        epochs           = 50,
        batch_size       = 256,
        validation_split = 0.20,
        callbacks        = [es, lr_reduce],
        verbose          = 1,
    )
    cnn.save(os.path.join(MODELS_DIR, "cnn_1d.keras"))

    # — Avaliação final —
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL — CONJUNTO DE TESTE")
    print("=" * 70)

    trained_models = {
        "Árvore de Decisão": dt,
        "Random Forest":     rf,
        "XGBoost":           xgb_clf,
        "CNN 1D":            cnn,
    }

    for name, clf in trained_models.items():
        if name == "CNN 1D":
            y_pred = np.argmax(clf.predict(X_test), axis=1)
        else:
            y_pred = clf.predict(X_test)

        print(f"\n{'─'*70}")
        print(f"Modelo: {name}")
        print(classification_report(y_test, y_pred, target_names=class_names,
                                    digits=4))

    print(f"\nModelos salvos em '{MODELS_DIR}/'.")


if __name__ == "__main__":
    main()
