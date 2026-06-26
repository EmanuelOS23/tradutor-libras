"""
=============================================================================
TRADUTOR DE LIBRAS - COLETA DE DADOS
=============================================================================
Script para coletar amostras de treinamento do alfabeto de LIBRAS.

COMO USAR:
  1. Execute: python coleta_dados.py
  2. A janela da câmera será aberta
  3. Posicione sua mão no quadro e faça o gesto de uma letra
  4. Pressione a TECLA correspondente à letra (ex: 'a' para A, 'b' para B...)
  5. Cada pressionamento salva UMA amostra
  6. Meta: pelo menos 100 amostras por letra
  7. Pressione 'q' para sair e salvar o dataset

DICA: Cada membro da equipe pode rodar o script e coletar amostras.
      Os dados são ACUMULADOS no arquivo CSV (não sobrescritos).
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

# ── Configurações ──────────────────────────────────────────────────────────────
DATASET_PATH = "dados/dataset.csv"
LETRAS_VALIDAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
META_POR_LETRA = 100  # Amostras desejadas por letra

# ── Cores (BGR) ────────────────────────────────────────────────────────────────
COR_VERDE     = (0, 230, 100)
COR_AZUL      = (60, 140, 255)
COR_VERMELHO  = (0, 60, 220)
COR_AMARELO   = (0, 210, 255)
COR_BRANCO    = (255, 255, 255)
COR_PRETO     = (0, 0, 0)
COR_FUNDO     = (15, 17, 22)
COR_DESTAQUE  = (80, 200, 255)

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


def normalizar_landmarks(landmarks, largura, altura):
    """
    Extrai e normaliza os 21 landmarks da mão.
    Normalização:
      - Coordenadas convertidas para pixels
      - Subtraídas do pulso (ponto 0) → invariante à posição
      - Divididas pela distância máxima → invariante ao tamanho
    Retorna um vetor de 63 valores (21 pontos × 3 coordenadas).
    """
    pontos = []
    for lm in landmarks.landmark:
        pontos.append([lm.x * largura, lm.y * altura, lm.z * largura])
    pontos = np.array(pontos)  # shape (21, 3)

    # Translação: centralizar no pulso (landmark 0)
    pulso = pontos[0].copy()
    pontos -= pulso

    # Escala: normalizar pela distância máxima do pulso
    escala = np.max(np.abs(pontos))
    if escala > 0:
        pontos /= escala

    return pontos.flatten().tolist()  # 63 valores


def carregar_contagens():
    """Carrega contagem atual de amostras por letra do CSV."""
    contagens = {letra: 0 for letra in LETRAS_VALIDAS}
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        if "label" in df.columns:
            for letra, contagem in df["label"].value_counts().items():
                if letra in contagens:
                    contagens[letra] = int(contagem)
    return contagens


def desenhar_painel_info(frame, contagens, ultima_letra, flash_timer):
    """Desenha o painel lateral de informações."""
    h, w = frame.shape[:2]

    # Painel lateral direito (fundo escuro)
    painel_w = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - painel_w, 0), (w, h), (20, 22, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Título
    cv2.putText(frame, "COLETA DE DADOS", (w - painel_w + 8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_DESTAQUE, 1, cv2.LINE_AA)
    cv2.putText(frame, "LIBRAS - Alfabeto", (w - painel_w + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COR_BRANCO, 1, cv2.LINE_AA)
    cv2.line(frame, (w - painel_w, 58), (w, 58), COR_DESTAQUE, 1)

    # Progresso por letra (mini grade)
    cols = 4
    cell_w = painel_w // cols
    start_y = 70
    cell_h = 20

    for i, letra in enumerate(LETRAS_VALIDAS):
        col = i % cols
        row = i // cols
        x = (w - painel_w) + col * cell_w + 2
        y = start_y + row * cell_h

        count = contagens[letra]
        progresso = min(count / META_POR_LETRA, 1.0)

        # Cor baseada no progresso
        if progresso >= 1.0:
            cor_bg = (30, 120, 40)
            cor_txt = COR_BRANCO
        elif progresso > 0.5:
            cor_bg = (20, 80, 130)
            cor_txt = COR_BRANCO
        elif progresso > 0:
            cor_bg = (20, 50, 80)
            cor_txt = (180, 180, 180)
        else:
            cor_bg = (30, 30, 35)
            cor_txt = (90, 90, 90)

        cv2.rectangle(frame, (x, y), (x + cell_w - 3, y + cell_h - 2), cor_bg, -1)
        cv2.putText(frame, f"{letra}:{count}", (x + 2, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, cor_txt, 1, cv2.LINE_AA)

    # Total coletado
    total = sum(contagens.values())
    total_meta = len(LETRAS_VALIDAS) * META_POR_LETRA
    sep_y = start_y + (len(LETRAS_VALIDAS) // cols + 1) * cell_h + 5
    cv2.line(frame, (w - painel_w, sep_y), (w, sep_y), (60, 60, 70), 1)
    cv2.putText(frame, f"Total: {total}/{total_meta}", (w - painel_w + 8, sep_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COR_AMARELO, 1, cv2.LINE_AA)

    # Flash de confirmação ao salvar
    if flash_timer > 0 and ultima_letra:
        alpha = min(flash_timer / 15.0, 1.0)
        cor_flash = tuple(int(c * alpha) for c in COR_VERDE)
        cv2.putText(frame, f"✓ '{ultima_letra}' salva!", (w - painel_w + 8, sep_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_flash, 1, cv2.LINE_AA)

    # Instrução na base
    cv2.putText(frame, "Tecla = salvar amostra", (w - painel_w + 4, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q = sair e salvar", (w - painel_w + 4, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 160), 1, cv2.LINE_AA)


def desenhar_overlay_mao(frame, resultado, largura, altura):
    """Desenha os landmarks da mão com estilo personalizado."""
    if not resultado.multi_hand_landmarks:
        return

    for hand_landmarks in resultado.multi_hand_landmarks:
        # Conexões
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(60, 180, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 120, 220), thickness=2)
        )


def main():
    os.makedirs("dados", exist_ok=True)
    
    print("=" * 60)
    print("  COLETOR DE DADOS — ALFABETO DE LIBRAS")
    print("=" * 60)
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Meta: {META_POR_LETRA} amostras por letra")
    print(f"  Pressione a tecla da letra para salvar")
    print(f"  Pressione 'Q' para sair")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    amostras = []
    contagens = carregar_contagens()
    ultima_letra = ""
    flash_timer = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            painel_w = 200  # largura do painel lateral
            area_w = w - painel_w  # largura da área de captura

            # Processar apenas a área da câmera (sem o painel)
            frame_rgb = cv2.cvtColor(frame[:, :area_w], cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            resultado = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Fundo levemente escurecido
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (area_w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Retângulo guia para posição da mão
            guia_x1, guia_y1 = area_w // 4, h // 5
            guia_x2, guia_y2 = 3 * area_w // 4, 4 * h // 5
            cv2.rectangle(frame, (guia_x1, guia_y1), (guia_x2, guia_y2),
                          (60, 60, 80), 1)
            cv2.putText(frame, "Posicione a mao aqui", (guia_x1 + 10, guia_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 120), 1, cv2.LINE_AA)

            # Landmarks da mão
            desenhar_overlay_mao(frame, resultado, area_w, h)

            # Status de detecção
            if resultado.multi_hand_landmarks:
                cv2.putText(frame, "MAO DETECTADA", (12, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_VERDE, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Aguardando mao...", (12, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 100), 1, cv2.LINE_AA)

            # Painel lateral
            desenhar_painel_info(frame, contagens, ultima_letra, flash_timer)

            if flash_timer > 0:
                flash_timer -= 1

            cv2.imshow("LIBRAS - Coleta de Dados", frame)

            # Captura de tecla
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q') or tecla == ord('Q'):
                break

            letra_pressionada = chr(tecla).upper()
            if letra_pressionada in LETRAS_VALIDAS:
                if resultado.multi_hand_landmarks:
                    lm = resultado.multi_hand_landmarks[0]
                    features = normalizar_landmarks(lm, area_w, h)
                    amostras.append(features + [letra_pressionada])
                    contagens[letra_pressionada] += 1
                    ultima_letra = letra_pressionada
                    flash_timer = 20
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Letra '{letra_pressionada}' → {contagens[letra_pressionada]} amostras")
                else:
                    print(f"  ⚠ Nenhuma mão detectada! Posicione a mão antes de pressionar.")

    cap.release()
    cv2.destroyAllWindows()

    # Salvar no CSV
    if amostras:
        colunas = []
        for i in range(21):
            colunas += [f"x{i}", f"y{i}", f"z{i}"]
        colunas.append("label")

        df_novo = pd.DataFrame(amostras, columns=colunas)

        if os.path.exists(DATASET_PATH):
            df_existente = pd.read_csv(DATASET_PATH)
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_csv(DATASET_PATH, index=False)
        print(f"\n✓ {len(amostras)} novas amostras salvas em '{DATASET_PATH}'")
        print(f"✓ Total no dataset: {len(df_final)} amostras")
        print("\nDistribuição por letra:")
        print(df_final["label"].value_counts().sort_index().to_string())
    else:
        print("\nNenhuma amostra nova coletada.")


if __name__ == "__main__":
    main()
