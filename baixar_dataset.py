"""
=============================================================================
DOWNLOAD DO DATASET DE LIBRAS (Kaggle)
=============================================================================
Baixa o dataset "Libras Landmark Dataset A-Z" do Kaggle e converte para
o formato do nosso projeto (dataset.csv com landmarks normalizados).

COMO USAR:
  Opção A — Com conta Kaggle (recomendado, mais dados):
    1. Crie uma conta em kaggle.com (grátis)
    2. Va em kaggle.com → seu perfil → Settings → API → Create New Token
    3. Isso vai baixar um arquivo "kaggle.json"
    4. Coloque o kaggle.json na pasta do projeto
    5. Execute: python baixar_dataset.py

  Opção B — Sem conta Kaggle (dataset embutido menor):
    Execute: python baixar_dataset.py --offline
    Isso gera um dataset sintético baseado nas posições reais do alfabeto
    de LIBRAS para testes imediatos. Depois substitua pelos dados reais.

=============================================================================
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd

DATASET_PATH  = "dados/dataset.csv"
KAGGLE_DATASET = "heitorccf/librasign"

# ── Posições aproximadas dos landmarks para cada letra de LIBRAS ─────────────
# Estas são posições NORMALIZADAS (relativas ao pulso) baseadas no alfabeto
# manual brasileiro. Permitem testar o sistema antes de coletar dados reais.
# Cada letra tem uma configuração característica dos dedos.
#
# Convenção:  landmark[0] = pulso (sempre 0,0,0 após normalização)
#             landmarks[1-4]  = polegar
#             landmarks[5-8]  = indicador
#             landmarks[9-12] = médio
#             landmarks[13-16]= anelar
#             landmarks[17-20]= mindinho
#
# Os valores foram derivados estudando as configurações típicas do DACTILEMA.

POSES_LIBRAS = {
    # Mão fechada - polegar para fora
    "A": {
        "polegar":   [(0.12, -0.15, 0), (0.20, -0.25, 0), (0.25, -0.28, 0), (0.28, -0.30, 0)],
        "indicador": [(0.10, -0.40, 0), (0.10, -0.55, 0), (0.10, -0.65, 0), (0.10, -0.70, 0)],
        "medio":     [(0.00, -0.42, 0), (0.00, -0.58, 0), (0.00, -0.68, 0), (0.00, -0.73, 0)],
        "anelar":    [(-0.10,-0.40, 0),(-0.10,-0.55, 0),(-0.10,-0.64, 0),(-0.10,-0.70, 0)],
        "mindinho":  [(-0.20,-0.35, 0),(-0.20,-0.48, 0),(-0.20,-0.57, 0),(-0.20,-0.62, 0)],
        "dobrado": [1,2,3,4],  # dedos dobrados (indicador, medio, anelar, mindinho)
    },
    # Mão em C aberto
    "B": {
        "polegar":   [(0.10, -0.20, 0), (0.08, -0.30, 0), (0.06, -0.35, 0), (0.05, -0.38, 0)],
        "indicador": [(0.12, -0.40, 0), (0.12, -0.62, 0), (0.12, -0.78, 0), (0.12, -0.88, 0)],
        "medio":     [(0.00, -0.42, 0), (0.00, -0.65, 0), (0.00, -0.80, 0), (0.00, -0.90, 0)],
        "anelar":    [(-0.12,-0.40, 0),(-0.12,-0.62, 0),(-0.12,-0.77, 0),(-0.12,-0.87, 0)],
        "mindinho":  [(-0.22,-0.35, 0),(-0.22,-0.55, 0),(-0.22,-0.68, 0),(-0.22,-0.78, 0)],
    },
}

def gerar_landmarks_base(letra):
    """
    Gera landmarks base para cada letra do alfabeto LIBRAS.
    Cada letra retorna um vetor de 63 valores (21 pontos x,y,z normalizados).
    As posições foram modeladas a partir das configurações reais do datilograma.
    """
    import math

    # ─── Tabela de landmarks por letra ────────────────────────────────────────
    # Formato: lista de 21 tuplas (x, y, z) — landmark 0 = pulso = (0,0,0)
    # Convenção de sinal: y negativo = subindo (para cima da mão)
    #                     x positivo = para o lado do polegar

    def pulso():
        return (0.0, 0.0, 0.0)

    # Helpers para posições típicas
    # Dedo estendido: ponta longe do pulso em y negativo
    # Dedo dobrado: ponta próxima da palma

    def segmentos_dedo(base_x, base_y, ext, curva_x=0.0, curva_y=0.0):
        """Gera 4 pontos de um dedo com extensão ext (0=fechado, 1=aberto)."""
        comprimentos = [0.12, 0.10, 0.09, 0.08]
        pts = []
        x, y = base_x, base_y
        for i, comp in enumerate(comprimentos):
            if ext >= 0.7:  # estendido
                nx = x + curva_x * comp
                ny = y - comp * ext
            elif ext <= 0.3:  # dobrado (curva para palma)
                nx = x + comp * 0.3 * (1 - ext)
                ny = y - comp * 0.25
            else:  # semi-dobrado
                nx = x + curva_x * comp * 0.5
                ny = y - comp * ext * 0.7
            pts.append((round(nx, 4), round(ny, 4), 0.0))
            x, y = nx, ny
        return pts

    # ── Configurações detalhadas por letra ────────────────────────────────────
    # Cada entrada: (polegar_x, polegar_y, [ext_t, ext_i, ext_m, ext_a, ext_mi],
    #                separacao_iv, separacao_mv, angulo_polegar)
    # + ajustes finos de posição para distinguir letras similares

    LETRAS = {
        # A: punho fechado, polegar ao lado (lateral, acima dos dedos)
        "A": dict(t_base=(0.18,-0.08), t_ext=0.65, t_cx=0.05,
                  i_base=(0.10,-0.28), i_ext=0.18,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.18),

        # B: 4 dedos esticados juntos, polegar dobrado sobre a palma
        "B": dict(t_base=(0.05,-0.18), t_ext=0.15,
                  i_base=(0.10,-0.28), i_ext=0.95,
                  m_base=(0.00,-0.30), m_ext=0.95,
                  a_base=(-0.10,-0.28), a_ext=0.92,
                  mi_base=(-0.18,-0.24), mi_ext=0.88),

        # C: todos os dedos curvados formando letra C
        "C": dict(t_base=(0.20,-0.10), t_ext=0.55, t_cx=0.12,
                  i_base=(0.12,-0.28), i_ext=0.50,
                  m_base=(0.02,-0.30), m_ext=0.50,
                  a_base=(-0.08,-0.28), a_ext=0.48,
                  mi_base=(-0.16,-0.24), mi_ext=0.44),

        # D: indicador esticado, outros fechados, polegar toca o médio
        "D": dict(t_base=(0.08,-0.22), t_ext=0.40, t_cx=0.08,
                  i_base=(0.12,-0.28), i_ext=0.95,
                  m_base=(0.02,-0.30), m_ext=0.20,
                  a_base=(-0.08,-0.28), a_ext=0.18,
                  mi_base=(-0.16,-0.24), mi_ext=0.15),

        # E: todos os dedos semi-dobrados, pontas tocando a palma
        "E": dict(t_base=(0.06,-0.14), t_ext=0.20,
                  i_base=(0.10,-0.28), i_ext=0.22,
                  m_base=(0.00,-0.30), m_ext=0.22,
                  a_base=(-0.10,-0.28), a_ext=0.20,
                  mi_base=(-0.18,-0.24), mi_ext=0.18,
                  # E: pontas curvadas para dentro (distingue de A,M,N,S)
                  _offset_iy=-0.05, _offset_my=-0.05),

        # F: polegar+indicador fazem O, outros 3 abertos
        "F": dict(t_base=(0.12,-0.18), t_ext=0.45, t_cx=0.08,
                  i_base=(0.10,-0.28), i_ext=0.22,
                  m_base=(0.00,-0.30), m_ext=0.90,
                  a_base=(-0.10,-0.28), a_ext=0.88,
                  mi_base=(-0.18,-0.24), mi_ext=0.85),

        # G: polegar e indicador horizontais apontando para frente
        "G": dict(t_base=(0.18,-0.08), t_ext=0.80, t_cx=0.15,
                  i_base=(0.12,-0.24), i_ext=0.90, i_cx=0.12,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.16,
                  mi_base=(-0.18,-0.24), mi_ext=0.14),

        # H: indicador+médio esticados horizontalmente (distingue de U/V/R)
        "H": dict(t_base=(0.06,-0.16), t_ext=0.18,
                  i_base=(0.10,-0.26), i_ext=0.90, i_cx=0.10,
                  m_base=(0.00,-0.26), m_ext=0.88, m_cx=0.08,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15,
                  _horizontal=True),

        # I: só mindinho esticado
        "I": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.10,-0.28), i_ext=0.18,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.95),

        # J: igual I mas mindinho mais inclinado (pose inicial do J)
        "J": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.10,-0.28), i_ext=0.18,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.20,-0.22), mi_ext=0.92, mi_cx=-0.05),

        # K: indicador+médio esticados em V, polegar entre eles
        "K": dict(t_base=(0.12,-0.22), t_ext=0.60, t_cx=0.05,
                  i_base=(0.12,-0.28), i_ext=0.92, i_cx=0.04,
                  m_base=(0.00,-0.30), m_ext=0.88, m_cx=-0.04,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15),

        # L: polegar+indicador abertos em ângulo reto (L)
        "L": dict(t_base=(0.18,-0.08), t_ext=0.92, t_cx=0.12,
                  i_base=(0.12,-0.28), i_ext=0.95,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.16,
                  mi_base=(-0.18,-0.24), mi_ext=0.14),

        # M: punho fechado, polegar sob 3 dedos (indicador+médio+anelar sobre polegar)
        "M": dict(t_base=(0.04,-0.20), t_ext=0.15,
                  i_base=(0.10,-0.28), i_ext=0.25,
                  m_base=(0.00,-0.30), m_ext=0.25,
                  a_base=(-0.10,-0.28), a_ext=0.22,
                  mi_base=(-0.18,-0.24), mi_ext=0.18,
                  # M: 3 dedos dobrados sobre o polegar (distinto de A,E,N,S)
                  _offset_ix=0.04, _offset_mx=0.04, _offset_ax=0.02),

        # N: igual M mas polegar sob 2 dedos
        "N": dict(t_base=(0.04,-0.20), t_ext=0.15,
                  i_base=(0.10,-0.28), i_ext=0.25,
                  m_base=(0.00,-0.30), m_ext=0.22,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.18,
                  _offset_ix=0.04, _offset_mx=0.02),

        # O: todos os dedos curvados formando círculo com polegar
        "O": dict(t_base=(0.16,-0.12), t_ext=0.42, t_cx=0.10,
                  i_base=(0.12,-0.28), i_ext=0.38,
                  m_base=(0.02,-0.30), m_ext=0.38,
                  a_base=(-0.08,-0.28), a_ext=0.36,
                  mi_base=(-0.16,-0.24), mi_ext=0.32),

        # P: indicador apontando para baixo, médio também
        "P": dict(t_base=(0.10,-0.18), t_ext=0.50,
                  i_base=(0.12,-0.28), i_ext=0.90, i_cx=0.0,
                  m_base=(0.02,-0.28), m_ext=0.25,
                  a_base=(-0.08,-0.26), a_ext=0.18,
                  mi_base=(-0.16,-0.22), mi_ext=0.15,
                  _apontando_baixo=True),

        # Q: similar a G mas apontando para baixo
        "Q": dict(t_base=(0.16,-0.10), t_ext=0.70, t_cx=0.08,
                  i_base=(0.12,-0.26), i_ext=0.85, i_cx=0.06,
                  m_base=(0.00,-0.30), m_ext=0.20,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15),

        # R: indicador+médio cruzados (distingue de H,U,V)
        "R": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.10,-0.28), i_ext=0.88, i_cx=0.03,
                  m_base=(0.04,-0.30), m_ext=0.85, m_cx=-0.03,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15,
                  _cruzado=True),

        # S: punho fechado, polegar SOBRE os dedos na frente
        "S": dict(t_base=(0.10,-0.22), t_ext=0.35, t_cx=0.06,
                  i_base=(0.10,-0.28), i_ext=0.20,
                  m_base=(0.00,-0.30), m_ext=0.20,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.16),

        # T: polegar entre indicador e médio (acima do indicador)
        "T": dict(t_base=(0.08,-0.24), t_ext=0.50, t_cx=0.04,
                  i_base=(0.10,-0.28), i_ext=0.22,
                  m_base=(0.00,-0.30), m_ext=0.22,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.16,
                  _t_entre_dedos=True),

        # U: indicador+médio juntos e verticais (distingue de H=horizontais, V=separados, R=cruzados)
        "U": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.09,-0.28), i_ext=0.92,
                  m_base=(0.01,-0.30), m_ext=0.90,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15),

        # V: indicador+médio separados em V (distingue de U=juntos)
        "V": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.13,-0.28), i_ext=0.92, i_cx=0.04,
                  m_base=(-0.03,-0.30), m_ext=0.90, m_cx=-0.04,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.15),

        # W: 3 dedos (indicador+médio+anelar) abertos e separados
        "W": dict(t_base=(0.06,-0.16), t_ext=0.20,
                  i_base=(0.14,-0.28), i_ext=0.92, i_cx=0.05,
                  m_base=(0.00,-0.30), m_ext=0.92,
                  a_base=(-0.12,-0.28), a_ext=0.88, a_cx=-0.05,
                  mi_base=(-0.18,-0.24), mi_ext=0.18),

        # X: indicador curvado em gancho, outros fechados
        "X": dict(t_base=(0.10,-0.16), t_ext=0.30,
                  i_base=(0.10,-0.28), i_ext=0.55, i_cx=0.06,
                  m_base=(0.00,-0.30), m_ext=0.20,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.16),

        # Y: polegar+mindinho abertos, outros fechados
        "Y": dict(t_base=(0.18,-0.08), t_ext=0.90, t_cx=0.10,
                  i_base=(0.10,-0.28), i_ext=0.20,
                  m_base=(0.00,-0.30), m_ext=0.18,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.20,-0.22), mi_ext=0.92),

        # Z: indicador esticado (pose inicial do Z)
        "Z": dict(t_base=(0.08,-0.18), t_ext=0.30,
                  i_base=(0.12,-0.28), i_ext=0.95,
                  m_base=(0.00,-0.30), m_ext=0.22,
                  a_base=(-0.10,-0.28), a_ext=0.18,
                  mi_base=(-0.18,-0.24), mi_ext=0.16),
    }

    cfg = LETRAS.get(letra, LETRAS["A"])

    lm = [pulso()]

    def pega(cfg, prefixo, default_x, default_y, default_ext, default_cx=0.0):
        bx = cfg.get(f"{prefixo}_base", (default_x, default_y))[0]
        by = cfg.get(f"{prefixo}_base", (default_x, default_y))[1]
        ex = cfg.get(f"{prefixo}_ext", default_ext)
        cx = cfg.get(f"{prefixo}_cx", default_cx)
        off_x = cfg.get(f"_offset_{prefixo}x", 0.0)
        off_y = cfg.get(f"_offset_{prefixo}y", 0.0)
        pts = segmentos_dedo(bx + off_x, by + off_y, ex, cx)
        return pts

    lm += pega(cfg, "t",  0.15, -0.10, 0.5)
    lm += pega(cfg, "i",  0.10, -0.28, 0.5)
    lm += pega(cfg, "m",  0.00, -0.30, 0.5)
    lm += pega(cfg, "a", -0.10, -0.28, 0.5)
    lm += pega(cfg, "mi",-0.18, -0.24, 0.5)

    return [coord for pt in lm for coord in pt]


def gerar_dataset_sintetico(n_por_letra=150, ruido=0.025):
    """
    Gera dataset sintético com ruído gaussiano para simular variações naturais.
    NOTA: Este dataset é APROXIMADO. Para melhor acurácia, use dados reais coletados.
    """
    print("\n  Gerando dataset sintetico baseado nas configuracoes do alfabeto LIBRAS...")
    print(f"  {n_por_letra} amostras por letra + ruido gaussiano (std={ruido})")

    colunas = [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]]
    colunas.append("label")

    linhas = []
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for letra in letras:
        base = np.array(gerar_landmarks_base(letra))
        for _ in range(n_por_letra):
            # Adiciona ruído gaussiano para simular variações naturais da mão
            amostra = base + np.random.normal(0, ruido, size=base.shape)
            linhas.append(list(amostra) + [letra])

        print(f"  {letra}: {n_por_letra} amostras geradas")

    df = pd.DataFrame(linhas, columns=colunas)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # embaralha
    return df


def baixar_kaggle(kaggle_json_path=None):
    """Tenta baixar o dataset via API do Kaggle."""
    try:
        import kaggle
    except ImportError:
        print("  Instalando kaggle...")
        os.system("pip install kaggle -q")
        import kaggle

    # Configura credenciais
    if kaggle_json_path and os.path.exists(kaggle_json_path):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        destino = os.path.expanduser("~/.kaggle/kaggle.json")
        shutil.copy(kaggle_json_path, destino)
        os.chmod(destino, 0o600)
        print(f"  Credenciais configuradas de: {kaggle_json_path}")
    elif os.path.exists("kaggle.json"):
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        destino = os.path.expanduser("~/.kaggle/kaggle.json")
        shutil.copy("kaggle.json", destino)
        os.chmod(destino, 0o600)
        print("  Credenciais kaggle.json encontradas na pasta do projeto.")

    print(f"\n  Baixando dataset: {KAGGLE_DATASET}")
    print("  Aguarde...")

    os.makedirs("dados/kaggle_raw", exist_ok=True)
    os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p dados/kaggle_raw --unzip')

    # Verifica se baixou
    arquivos_csv = []
    for raiz, dirs, arquivos in os.walk("dados/kaggle_raw"):
        for arq in arquivos:
            if arq.endswith(".csv"):
                arquivos_csv.append(os.path.join(raiz, arq))

    return arquivos_csv


def converter_formato_kaggle(arquivos_csv):
    """
    Converte o formato do dataset Kaggle (um CSV por letra) para
    o nosso formato unificado (um CSV com coluna 'label').
    """
    print(f"\n  Convertendo {len(arquivos_csv)} arquivos CSV...")

    dfs = []
    colunas_target = [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]]

    for caminho in sorted(arquivos_csv):
        nome = os.path.basename(caminho)
        # Tenta descobrir a letra pelo nome do arquivo (ex: A.csv, letra_A.csv)
        letra = None
        base = os.path.splitext(nome)[0].upper()
        if len(base) == 1 and base.isalpha():
            letra = base
        elif "_" in base:
            partes = base.split("_")
            for p in partes:
                if len(p) == 1 and p.isalpha():
                    letra = p
                    break

        if not letra:
            print(f"  Ignorando {nome} (nao identificou letra)")
            continue

        df_letra = pd.read_csv(caminho)
        print(f"  {nome} -> Letra {letra}: {len(df_letra)} amostras, colunas: {list(df_letra.columns[:5])}...")

        # Verifica formato: se tem 63 colunas numéricas, usa direto
        colunas_num = [c for c in df_letra.columns if df_letra[c].dtype in [np.float64, np.float32, np.int64]]

        if len(colunas_num) >= 63:
            # Pega as primeiras 63 colunas numéricas
            dados = df_letra[colunas_num[:63]].values
            df_convertido = pd.DataFrame(dados, columns=colunas_target)
            df_convertido["label"] = letra
            dfs.append(df_convertido)
        else:
            print(f"  AVISO: {nome} nao tem 63 colunas numericas (tem {len(colunas_num)}). Ignorando.")

    if not dfs:
        return None

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_final


def main():
    parser = argparse.ArgumentParser(description="Download/gera dataset de LIBRAS")
    parser.add_argument("--offline", action="store_true",
                        help="Gera dataset sintetico sem precisar do Kaggle")
    parser.add_argument("--kaggle-json", type=str, default=None,
                        help="Caminho para o arquivo kaggle.json")
    parser.add_argument("--n-amostras", type=int, default=150,
                        help="Amostras por letra no modo sintetico (padrao: 150)")
    args = parser.parse_args()

    os.makedirs("dados", exist_ok=True)

    print("=" * 60)
    print("  DOWNLOAD / GERACAO DE DATASET DE LIBRAS")
    print("=" * 60)

    if args.offline:
        # ── Modo offline: dataset sintético ───────────────────────────────
        print("\n  MODO OFFLINE - Gerando dataset sintetico")
        print("  NOTA: Para melhores resultados, use dados reais coletados")
        print("        com coleta_dados.py ou baixados do Kaggle.")

        df = gerar_dataset_sintetico(n_por_letra=args.n_amostras)

    else:
        # ── Modo online: baixar do Kaggle ──────────────────────────────────
        print("\n  MODO KAGGLE - Baixando dataset real de LIBRAS")
        print("\n  Para usar este modo, voce precisa:")
        print("  1. Criar conta em kaggle.com (gratis)")
        print("  2. Ir em Settings > API > 'Create New Token'")
        print("  3. Colocar o kaggle.json baixado na pasta do projeto")
        print("  4. Executar: python baixar_dataset.py")

        # Verifica se tem credenciais
        tem_credenciais = (
            os.path.exists("kaggle.json") or
            os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or
            (args.kaggle_json and os.path.exists(args.kaggle_json))
        )

        if not tem_credenciais:
            print("\n  ERRO: Nenhum arquivo kaggle.json encontrado!")
            print("\n  Opcoes:")
            print("  A) Baixe manualmente em: https://www.kaggle.com/datasets/heitorccf/librasign")
            print("     Extraia os CSVs em dados/kaggle_raw/ e rode novamente")
            print("  B) Use o modo offline:  python baixar_dataset.py --offline")
            print("\n  Usando modo offline como fallback...")
            df = gerar_dataset_sintetico(n_por_letra=args.n_amostras)
        else:
            try:
                arquivos_csv = baixar_kaggle(args.kaggle_json)
                if arquivos_csv:
                    df = converter_formato_kaggle(arquivos_csv)
                    if df is None:
                        print("  Falha na conversao. Usando modo offline.")
                        df = gerar_dataset_sintetico(n_por_letra=args.n_amostras)
                else:
                    print("  Nenhum CSV encontrado. Usando modo offline.")
                    df = gerar_dataset_sintetico(n_por_letra=args.n_amostras)
            except Exception as e:
                print(f"  Erro ao baixar do Kaggle: {e}")
                print("  Usando modo offline como fallback.")
                df = gerar_dataset_sintetico(n_por_letra=args.n_amostras)

    # ── Salvar dataset ──────────────────────────────────────────────────────
    # Se já existe dataset coletado manualmente, combina com ele
    if os.path.exists(DATASET_PATH):
        print(f"\n  Dataset existente encontrado em {DATASET_PATH}")
        df_existente = pd.read_csv(DATASET_PATH)
        print(f"  Combinando: {len(df_existente)} amostras existentes + {len(df)} novas")
        df = pd.concat([df_existente, df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(DATASET_PATH, index=False)

    print(f"\n  Dataset salvo em '{DATASET_PATH}'")
    print(f"  Total de amostras: {len(df)}")
    print(f"  Letras cobertas: {df['label'].nunique()}")
    print("\n  Distribuicao por letra:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\n  Agora execute: python treinar_modelo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
