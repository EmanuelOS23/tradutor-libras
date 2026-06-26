"""
=============================================================================
TRADUTOR DE LIBRAS - TREINAMENTO DO MODELO
=============================================================================
Treina um classificador Random Forest com os dados coletados em dataset.csv
e salva o modelo treinado para uso no tradutor em tempo real.

COMO USAR:
  1. Colete dados com: python coleta_dados.py
  2. Execute: python treinar_modelo.py
  3. O modelo será salvo em modelo/modelo_libras.pkl

SAÍDA:
  - modelo/modelo_libras.pkl  → modelo treinado
  - modelo/classes.pkl        → lista de classes (letras)
  - Relatório de avaliação no terminal
  - Gráfico da matriz de confusão
=============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Configurações ──────────────────────────────────────────────────────────────
DATASET_PATH    = "dados/dataset.csv"
MODELO_PATH     = "modelo/modelo_libras.pkl"
CLASSES_PATH    = "modelo/classes.pkl"
RELATORIO_PATH  = "modelo/relatorio_treinamento.png"

# Hiperparâmetros do Random Forest
N_ESTIMATORS   = 200   # número de árvores
MAX_DEPTH      = None  # profundidade máxima (None = sem limite)
MIN_SAMPLES    = 2     # amostras mínimas para dividir nó
TEST_SIZE      = 0.20  # 20% para teste
RANDOM_STATE   = 42


def carregar_dataset():
    """Carrega e valida o dataset CSV."""
    if not os.path.exists(DATASET_PATH):
        print(f"\nERRO: Dataset nao encontrado em '{DATASET_PATH}'")
        print("   Execute 'python coleta_dados.py' primeiro para coletar dados.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n  Dataset carregado: {len(df)} amostras, {df['label'].nunique()} classes")

    # Verificar amostras por classe
    contagens = df["label"].value_counts().sort_index()
    print("\n  Amostras por letra:")
    print("  " + "-" * 40)
    for letra, count in contagens.items():
        barra = "|" * min(count // 5, 40)
        status = "OK" if count >= 50 else "!" if count >= 20 else "X"
        print(f"  [{status}] {letra}: {count:4d}  {barra}")

    classes_insuficientes = contagens[contagens < 20].index.tolist()
    if classes_insuficientes:
        print(f"\n  AVISO: Letras com poucas amostras (< 20): {classes_insuficientes}")
        print("    Recomenda-se coletar mais dados para essas letras.")

    return df


def preparar_features(df):
    """Separa features (X) e rotulos (y) do dataset."""
    colunas_features = [c for c in df.columns if c != "label"]
    X = df[colunas_features].values.astype(np.float32)
    y = df["label"].values
    return X, y


def treinar(X_treino, y_treino):
    """Treina o modelo Random Forest."""
    print("\n  Treinando Random Forest...")
    print(f"  -> {N_ESTIMATORS} arvores | profundidade maxima: {'ilimitada' if MAX_DEPTH is None else MAX_DEPTH}")

    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,         # usar todos os nucleos do processador
        verbose=0
    )

    inicio = time.time()
    modelo.fit(X_treino, y_treino)
    tempo = time.time() - inicio

    print(f"  -> Treinamento concluido em {tempo:.2f}s")
    return modelo


def avaliar(modelo, X_treino, X_teste, y_treino, y_teste, classes):
    """Avalia o modelo e imprime metricas."""
    print("\n" + "=" * 60)
    print("  AVALIACAO DO MODELO")
    print("=" * 60)

    # Acuracia nos conjuntos de treino e teste
    acc_treino = accuracy_score(y_treino, modelo.predict(X_treino))
    y_pred = modelo.predict(X_teste)
    acc_teste = accuracy_score(y_teste, y_pred)

    print(f"\n  Acuracia no treino : {acc_treino*100:.2f}%")
    print(f"  Acuracia no teste  : {acc_teste*100:.2f}%")

    if acc_treino - acc_teste > 0.10:
        print("  AVISO: Possivel overfitting (diferenca > 10%). Colete mais dados.")
    else:
        print("  OK: Modelo generaliza bem (sem overfitting significativo).")

    # Relatorio por classe
    print("\n  Relatorio por letra:")
    print(classification_report(y_teste, y_pred, target_names=classes, zero_division=0))

    return y_pred, acc_teste


def salvar_graficos(y_teste, y_pred, classes, acc_teste):
    """Gera e salva grafico com matriz de confusao e importancias."""
    print("\n  Gerando graficos...")

    # Paleta de cores dark
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("#0f1116")

    # -- Grafico 1: Matriz de confusao --
    ax1 = fig.add_subplot(1, 2, 1)
    cm = confusion_matrix(y_teste, y_pred, labels=classes)
    mask_zero = cm == 0

    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues",
        linewidths=0.5, linecolor="#1a1d24",
        ax=ax1,
        mask=mask_zero,
        cbar_kws={"shrink": 0.8}
    )
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap=sns.color_palette(["#1a1d24"], as_cmap=True),
        linewidths=0.5, linecolor="#1a1d24",
        ax=ax1,
        mask=~mask_zero,
        cbar=False, alpha=0.3
    )

    ax1.set_title(f"Matriz de Confusao\nAcuracia: {acc_teste*100:.1f}%",
                  color="white", fontsize=14, pad=15)
    ax1.set_xlabel("Predito", color="#80c4ff", fontsize=11)
    ax1.set_ylabel("Real", color="#80c4ff", fontsize=11)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.set_facecolor("#0f1116")

    # -- Grafico 2: Acuracia por letra --
    ax2 = fig.add_subplot(1, 2, 2)
    cm_norm = cm.astype(float)
    with np.errstate(invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0)
    acc_por_classe = np.diag(cm_norm) * 100

    cores = ["#00e676" if a >= 90 else "#ffab40" if a >= 70 else "#ef5350"
             for a in acc_por_classe]
    bars = ax2.bar(classes, acc_por_classe, color=cores, edgecolor="#1a1d24", linewidth=0.5)

    ax2.axhline(90, color="#00e676", linestyle="--", alpha=0.5, linewidth=1, label="90%")
    ax2.axhline(70, color="#ffab40", linestyle="--", alpha=0.5, linewidth=1, label="70%")
    ax2.set_title("Acuracia por Letra", color="white", fontsize=14, pad=15)
    ax2.set_xlabel("Letra", color="#80c4ff", fontsize=11)
    ax2.set_ylabel("Acuracia (%)", color="#80c4ff", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.set_facecolor("#0f1116")
    ax2.spines["bottom"].set_color("#404050")
    ax2.spines["left"].set_color("#404050")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    legenda = [
        mpatches.Patch(color="#00e676", label=">= 90% (otimo)"),
        mpatches.Patch(color="#ffab40", label="70-89% (bom)"),
        mpatches.Patch(color="#ef5350", label="< 70% (melhorar)"),
    ]
    ax2.legend(handles=legenda, loc="lower right",
               facecolor="#1a1d24", edgecolor="#404050",
               labelcolor="white", fontsize=9)

    plt.tight_layout(pad=2.5)
    plt.savefig(RELATORIO_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Grafico salvo em '{RELATORIO_PATH}'")


def salvar_modelo(modelo, classes):
    """Salva o modelo e as classes em disco."""
    os.makedirs("modelo", exist_ok=True)
    with open(MODELO_PATH, "wb") as f:
        pickle.dump(modelo, f)
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(classes, f)
    print(f"  Modelo salvo em '{MODELO_PATH}'")
    print(f"  Classes salvas em '{CLASSES_PATH}'")


def main():
    print("=" * 60)
    print("  TREINAMENTO DO MODELO - ALFABETO DE LIBRAS")
    print("=" * 60)

    # 1. Carregar dados
    df = carregar_dataset()
    X, y = preparar_features(df)
    classes = sorted(df["label"].unique())

    print(f"\n  Features por amostra: {X.shape[1]} (21 landmarks x 3 coordenadas)")
    print(f"  Classes: {len(classes)} letras: {' '.join(classes)}")

    # 2. Dividir treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # garante proporção igual por classe
    )
    print(f"\n  Divisao: {len(X_treino)} treino | {len(X_teste)} teste")

    # 3. Treinar
    modelo = treinar(X_treino, y_treino)

    # 4. Avaliar
    y_pred, acc_teste = avaliar(modelo, X_treino, X_teste, y_treino, y_teste, classes)

    # 5. Validacao cruzada (extra)
    print("\n  Validacao cruzada (5-fold)...")
    scores_cv = cross_val_score(modelo, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  Acuracias por fold: {[f'{s*100:.1f}%' for s in scores_cv]}")
    print(f"  Media: {scores_cv.mean()*100:.2f}% +/- {scores_cv.std()*100:.2f}%")

    # 6. Salvar gráficos
    salvar_graficos(y_teste, y_pred, classes, acc_teste)

    # 7. Salvar modelo
    salvar_modelo(modelo, classes)

    print("\n" + "=" * 60)
    print("  TREINAMENTO CONCLUIDO!")
    print(f"  Acuracia final: {acc_teste*100:.2f}%")
    print("  Agora execute: python tradutor.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
