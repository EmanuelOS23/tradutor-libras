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
META_POR_LETRA = 100

# Caminho do modelo da nova API do MediaPipe
MODEL_PATH = "modelos/hand_landmarker.task"

# ── Cores (BGR) ────────────────────────────────────────────────────────────────
COR_VERDE     = (0, 230, 100)
COR_AZUL      = (60, 140, 255)
COR_VERMELHO  = (0, 60, 220)
COR_AMARELO   = (0, 210, 255)
COR_BRANCO    = (255, 255, 255)
COR_PRETO     = (0, 0, 0)
COR_FUNDO     = (15, 17, 22)
COR_DESTAQUE  = (80, 200, 255)

# ── MediaPipe Tasks ────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexões entre os 21 pontos da mão
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def criar_detector_maos():
    """Cria o detector de mãos usando a API nova do MediaPipe."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nModelo não encontrado: {MODEL_PATH}\n\n"
            "Baixe o modelo com o comando:\n\n"
            "mkdir -p modelos\n"
            "curl -L -o modelos/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task\n"
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    return HandLandmarker.create_from_options(options)


def normalizar_landmarks(hand_landmarks, largura, altura):
    """
    Extrai e normaliza os 21 landmarks da mão.

    Normalização:
      - Coordenadas convertidas para pixels
      - Subtraídas do pulso, ponto 0
      - Divididas pela maior distância absoluta

    Retorna um vetor de 63 valores:
      21 pontos × 3 coordenadas.
    """
    pontos = []

    for lm in hand_landmarks:
        pontos.append([lm.x * largura, lm.y * altura, lm.z * largura])

    pontos = np.array(pontos)

    pulso = pontos[0].copy()
    pontos -= pulso

    escala = np.max(np.abs(pontos))
    if escala > 0:
        pontos /= escala

    return pontos.flatten().tolist()


def carregar_contagens():
    """Carrega contagem atual de amostras por letra do CSV."""
    contagens = {letra: 0 for letra in LETRAS_VALIDAS}

    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)

            if "label" in df.columns:
                for letra, contagem in df["label"].value_counts().items():
                    if letra in contagens:
                        contagens[letra] = int(contagem)

        except pd.errors.EmptyDataError:
            pass

    return contagens


def desenhar_painel_info(frame, contagens, ultima_letra, flash_timer):
    """Desenha o painel lateral de informações."""
    h, w = frame.shape[:2]

    painel_w = 200
    overlay = frame.copy()

    cv2.rectangle(overlay, (w - painel_w, 0), (w, h), (20, 22, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(
        frame,
        "COLETA DE DADOS",
        (w - painel_w + 8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COR_DESTAQUE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "LIBRAS - Alfabeto",
        (w - painel_w + 8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COR_BRANCO,
        1,
        cv2.LINE_AA
    )

    cv2.line(frame, (w - painel_w, 58), (w, 58), COR_DESTAQUE, 1)

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

        cv2.rectangle(
            frame,
            (x, y),
            (x + cell_w - 3, y + cell_h - 2),
            cor_bg,
            -1
        )

        cv2.putText(
            frame,
            f"{letra}:{count}",
            (x + 2, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            cor_txt,
            1,
            cv2.LINE_AA
        )

    total = sum(contagens.values())
    total_meta = len(LETRAS_VALIDAS) * META_POR_LETRA

    sep_y = start_y + (len(LETRAS_VALIDAS) // cols + 1) * cell_h + 5

    cv2.line(frame, (w - painel_w, sep_y), (w, sep_y), (60, 60, 70), 1)

    cv2.putText(
        frame,
        f"Total: {total}/{total_meta}",
        (w - painel_w + 8, sep_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COR_AMARELO,
        1,
        cv2.LINE_AA
    )

    if flash_timer > 0 and ultima_letra:
        alpha = min(flash_timer / 15.0, 1.0)
        cor_flash = tuple(int(c * alpha) for c in COR_VERDE)

        cv2.putText(
            frame,
            f"'{ultima_letra}' salva!",
            (w - painel_w + 8, sep_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cor_flash,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "Tecla = salvar amostra",
        (w - painel_w + 4, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (140, 140, 160),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Q = sair e salvar",
        (w - painel_w + 4, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (140, 140, 160),
        1,
        cv2.LINE_AA
    )


def desenhar_overlay_mao(frame, resultado, largura, altura):
    """Desenha os landmarks da mão usando a API nova do MediaPipe."""
    if not resultado.hand_landmarks:
        return

    for hand_landmarks in resultado.hand_landmarks:
        pontos = []

        for lm in hand_landmarks:
            x = int(lm.x * largura)
            y = int(lm.y * altura)
            pontos.append((x, y))

        for inicio, fim in HAND_CONNECTIONS:
            if inicio < len(pontos) and fim < len(pontos):
                cv2.line(
                    frame,
                    pontos[inicio],
                    pontos[fim],
                    (0, 120, 220),
                    2
                )

        for x, y in pontos:
            cv2.circle(
                frame,
                (x, y),
                4,
                (60, 180, 255),
                -1
            )


def main():
    os.makedirs("dados", exist_ok=True)

    print("=" * 60)
    print("  COLETOR DE DADOS — ALFABETO DE LIBRAS")
    print("=" * 60)
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Meta: {META_POR_LETRA} amostras por letra")
    print("  Pressione a tecla da letra para salvar")
    print("  Pressione 'Q' para sair")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    amostras = []
    contagens = carregar_contagens()
    ultima_letra = ""
    flash_timer = 0

    ultimo_timestamp_ms = 0

    with criar_detector_maos() as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erro: não foi possível capturar frame da câmera.")
                break

            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            painel_w = 200
            area_w = w - painel_w

            # Processar apenas a área de captura, sem o painel lateral
            frame_rgb = cv2.cvtColor(frame[:, :area_w], cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            timestamp_ms = int(time.time() * 1000)

            # O MediaPipe em modo VIDEO exige timestamp crescente
            if timestamp_ms <= ultimo_timestamp_ms:
                timestamp_ms = ultimo_timestamp_ms + 1

            ultimo_timestamp_ms = timestamp_ms

            resultado = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Fundo levemente escurecido na área da câmera
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (area_w, h),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.15,
                frame,
                0.85,
                0,
                frame
            )

            # Retângulo guia para posição da mão
            guia_x1, guia_y1 = area_w // 4, h // 5
            guia_x2, guia_y2 = 3 * area_w // 4, 4 * h // 5

            cv2.rectangle(
                frame,
                (guia_x1, guia_y1),
                (guia_x2, guia_y2),
                (60, 60, 80),
                1
            )

            cv2.putText(
                frame,
                "Posicione a mao aqui",
                (guia_x1 + 10, guia_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 100, 120),
                1,
                cv2.LINE_AA
            )

            # Desenhar landmarks da mão
            desenhar_overlay_mao(frame, resultado, area_w, h)

            # Status de detecção
            if resultado.hand_landmarks:
                cv2.putText(
                    frame,
                    "MAO DETECTADA",
                    (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COR_VERDE,
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame,
                    "Aguardando mao...",
                    (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 80, 100),
                    1,
                    cv2.LINE_AA
                )

            # Painel lateral
            desenhar_painel_info(
                frame,
                contagens,
                ultima_letra,
                flash_timer
            )

            if flash_timer > 0:
                flash_timer -= 1

            cv2.imshow("LIBRAS - Coleta de Dados", frame)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q") or tecla == ord("Q"):
                break

            letra_pressionada = chr(tecla).upper()

            if letra_pressionada in LETRAS_VALIDAS:
                if resultado.hand_landmarks:
                    landmarks_mao = resultado.hand_landmarks[0]

                    features = normalizar_landmarks(
                        landmarks_mao,
                        area_w,
                        h
                    )

                    amostras.append(features + [letra_pressionada])
                    contagens[letra_pressionada] += 1
                    ultima_letra = letra_pressionada
                    flash_timer = 20

                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"Letra '{letra_pressionada}' → "
                        f"{contagens[letra_pressionada]} amostras"
                    )
                else:
                    print(
                        "  ⚠ Nenhuma mão detectada! "
                        "Posicione a mão antes de pressionar."
                    )

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
            try:
                df_existente = pd.read_csv(DATASET_PATH)
                df_final = pd.concat(
                    [df_existente, df_novo],
                    ignore_index=True
                )
            except pd.errors.EmptyDataError:
                df_final = df_novo
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