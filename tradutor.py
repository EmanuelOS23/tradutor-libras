"""
=============================================================================
TRADUTOR DE LIBRAS - APLICAÇÃO PRINCIPAL (TEMPO REAL)
=============================================================================
Usa a câmera para detectar gestos do alfabeto de LIBRAS em tempo real.

PRÉ-REQUISITOS:
  1. Coletou dados:   python coleta_dados.py
  2. Treinou modelo:  python treinar_modelo.py

COMO USAR:
  python tradutor.py

CONTROLES:
  Q ou ESC  → Sair
  H         → Mostrar/ocultar painel de ajuda
  S         → Mostrar/ocultar histórico de letras detectadas
  C         → Limpar histórico de letras
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import time
from collections import deque

# ── Configurações ──────────────────────────────────────────────────────────────
MODELO_PATH    = "modelo/modelo_libras.pkl"
CLASSES_PATH   = "modelo/classes.pkl"

CONFIANCA_MIN  = 0.70   # confiança mínima para exibir predição
HISTORICO_MAX  = 8      # últimas N letras diferentes no histórico
SUAVIZACAO_N   = 7      # frames para suavização da predição (evita piscadas)

# Letras com movimento real (J e Z) — serão marcadas com asterisco
LETRAS_COM_MOVIMENTO = {"J", "Z"}

# ── Paleta de Cores (BGR) ──────────────────────────────────────────────────────
COR_FUNDO_PAINEL = (15, 17, 22)
COR_AZUL_CLARO   = (255, 180, 60)   # BGR → azul claro
COR_VERDE        = (80, 230, 100)
COR_AMARELO      = (30, 210, 255)
COR_VERMELHO     = (60, 60, 220)
COR_BRANCO       = (255, 255, 255)
COR_CINZA        = (140, 140, 160)
COR_DESTAQUE     = (255, 160, 60)   # laranja-azul
COR_SOMBRA       = (0, 0, 0)

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def carregar_modelo():
    """Carrega o modelo e as classes salvas."""
    if not os.path.exists(MODELO_PATH):
        print(f"\n❌ ERRO: Modelo não encontrado em '{MODELO_PATH}'")
        print("   Execute 'python treinar_modelo.py' primeiro.")
        sys.exit(1)
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    print(f"✓ Modelo carregado | {len(classes)} letras: {' '.join(classes)}")
    return modelo, classes


def normalizar_landmarks(landmarks, largura, altura):
    """
    Extrai e normaliza os 21 landmarks da mão.
    DEVE ser idêntica à função em coleta_dados.py.
    """
    pontos = []
    for lm in landmarks.landmark:
        pontos.append([lm.x * largura, lm.y * altura, lm.z * largura])
    pontos = np.array(pontos)

    pulso = pontos[0].copy()
    pontos -= pulso

    escala = np.max(np.abs(pontos))
    if escala > 0:
        pontos /= escala

    return pontos.flatten().reshape(1, -1)


class SuavizadorPred:
    """Suaviza predições ao longo de N frames para reduzir flickering."""
    def __init__(self, n=SUAVIZACAO_N):
        self.historico = deque(maxlen=n)

    def atualizar(self, letra, confianca):
        self.historico.append((letra, confianca))

    def obter(self):
        if not self.historico:
            return None, 0.0
        # Conta votos por letra
        votos = {}
        confiancas = {}
        for letra, conf in self.historico:
            votos[letra] = votos.get(letra, 0) + 1
            confiancas[letra] = max(confiancas.get(letra, 0), conf)
        # Letra com mais votos
        letra_final = max(votos, key=votos.get)
        return letra_final, confiancas[letra_final]


def desenhar_texto_sombra(frame, texto, pos, fonte, escala, cor, espessura=1):
    """Desenha texto com sombra sutil para melhor legibilidade."""
    x, y = pos
    cv2.putText(frame, texto, (x + 1, y + 1), fonte, escala,
                (0, 0, 0), espessura + 1, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), fonte, escala, cor, espessura, cv2.LINE_AA)


def desenhar_painel_letra(frame, letra, confianca, tem_mao, w, h):
    """Painel principal — exibe a letra detectada em destaque."""
    painel_h = 160
    painel_w = 340

    # Posição: centro superior
    px = (w - painel_w) // 2
    py = 15

    # Fundo com borda arredondada (aproximado com retângulo)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + painel_w, py + painel_h),
                  (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (px, py), (px + painel_w, py + painel_h),
                  (50, 80, 120), 1)

    if tem_mao and letra and confianca >= CONFIANCA_MIN:
        # Cor baseada na confiança
        if confianca >= 0.90:
            cor_letra = COR_VERDE
            cor_conf  = COR_VERDE
        elif confianca >= 0.75:
            cor_letra = COR_AMARELO
            cor_conf  = COR_AMARELO
        else:
            cor_letra = (120, 150, 200)
            cor_conf  = COR_CINZA

        # Letra grande centralizada
        texto_letra = letra
        if letra in LETRAS_COM_MOVIMENTO:
            texto_letra = letra + "*"

        (tw, th), _ = cv2.getTextSize(texto_letra, cv2.FONT_HERSHEY_DUPLEX, 3.5, 3)
        tx = px + (painel_w - tw) // 2
        ty = py + painel_h // 2 + th // 2 - 5

        # Glow (múltiplas camadas)
        for espessura, alpha in [(12, 0.08), (7, 0.15), (4, 0.25)]:
            overlay2 = frame.copy()
            cv2.putText(overlay2, texto_letra, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 3.5, cor_letra, espessura, cv2.LINE_AA)
            cv2.addWeighted(overlay2, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, texto_letra, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 3.5, cor_letra, 3, cv2.LINE_AA)

        # Barra de confiança
        barra_x = px + 20
        barra_y = py + painel_h - 22
        barra_w = painel_w - 40
        barra_h = 8

        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + barra_w, barra_y + barra_h), (40, 45, 55), -1)
        preenchido = int(barra_w * confianca)
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + preenchido, barra_y + barra_h), cor_conf, -1)

        texto_conf = f"Confianca: {confianca*100:.0f}%"
        cv2.putText(frame, texto_conf, (barra_x, barra_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COR_CINZA, 1, cv2.LINE_AA)

    elif tem_mao:
        # Mão detectada mas confiança baixa
        desenhar_texto_sombra(frame, "Ajuste o gesto...",
                              (px + 50, py + painel_h // 2 + 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.65, COR_CINZA)
    else:
        # Sem mão
        desenhar_texto_sombra(frame, "Posicione a mao",
                              (px + 55, py + painel_h // 2 + 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 75, 95))
        desenhar_texto_sombra(frame, "na frente da camera",
                              (px + 38, py + painel_h // 2 + 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 65, 85))


def desenhar_historico(frame, historico_letras, w, h):
    """Painel inferior: últimas letras detectadas formando palavras."""
    painel_h = 55
    painel_w = w - 40
    px, py = 20, h - painel_h - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + painel_w, py + painel_h),
                  (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (px, py), (px + painel_w, py + painel_h),
                  (40, 50, 70), 1)

    cv2.putText(frame, "Historico:", (px + 10, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COR_CINZA, 1, cv2.LINE_AA)

    texto = " ".join(historico_letras) if historico_letras else "—"
    cv2.putText(frame, texto, (px + 100, py + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, COR_AZUL_CLARO, 1, cv2.LINE_AA)

    cv2.putText(frame, "C=limpar", (px + painel_w - 80, py + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 75, 90), 1, cv2.LINE_AA)


def desenhar_landmarks_custom(frame, resultado, area_w, h):
    """Renderiza os landmarks da mão com estilo customizado."""
    if not resultado.multi_hand_landmarks:
        return

    CONEXOES = mp_hands.HAND_CONNECTIONS
    COR_CONEXAO  = (60, 140, 255)
    COR_NO       = (0, 220, 255)
    COR_PONTA    = (0, 255, 160)

    PONTAS = {4, 8, 12, 16, 20}  # pontas dos dedos

    for hand_lm in resultado.multi_hand_landmarks:
        pontos = {}
        for idx, lm in enumerate(hand_lm.landmark):
            x = int(lm.x * area_w)
            y = int(lm.y * h)
            pontos[idx] = (x, y)

        # Conexões
        for conn in CONEXOES:
            p1, p2 = conn
            if p1 in pontos and p2 in pontos:
                cv2.line(frame, pontos[p1], pontos[p2], COR_CONEXAO, 2, cv2.LINE_AA)

        # Nós
        for idx, (x, y) in pontos.items():
            cor = COR_PONTA if idx in PONTAS else COR_NO
            raio = 5 if idx in PONTAS else 3
            cv2.circle(frame, (x, y), raio + 1, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), raio, cor, -1)


def desenhar_hud(frame, fps, w, h):
    """HUD mínimo: FPS e teclas de atalho."""
    # FPS (canto superior esquerdo)
    cor_fps = COR_VERDE if fps >= 25 else COR_AMARELO if fps >= 15 else COR_VERMELHO
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_fps, 1, cv2.LINE_AA)

    # Título (canto superior direito)
    titulo = "TRADUTOR DE LIBRAS"
    (tw, _), _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, titulo, (w - tw - 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_DESTAQUE, 1, cv2.LINE_AA)

    # Teclas de atalho (rodapé esquerdo)
    atalhos = "Q/ESC: Sair  |  C: Limpar historico"
    cv2.putText(frame, atalhos, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 75, 90), 1, cv2.LINE_AA)

    # Nota sobre letras com movimento
    nota = "* Letras J e Z possuem movimento — mostrada pose inicial"
    cv2.putText(frame, nota, (12, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60, 65, 80), 1, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("  TRADUTOR DE LIBRAS — TEMPO REAL")
    print("=" * 60)

    modelo, classes = carregar_modelo()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    suavizador = SuavizadorPred(n=SUAVIZACAO_N)
    historico_letras = deque(maxlen=HISTORICO_MAX)

    ultima_letra_hist = None
    tempo_estavel = 0           # frames que a mesma letra se mantém
    FRAMES_PARA_HISTORICO = 20  # frames estáveis antes de adicionar ao histórico

    fps_contador = 0
    fps_timer = time.time()
    fps_atual = 0.0

    print("\n  Câmera iniciada. Faça gestos do alfabeto de LIBRAS!")
    print("  Pressione Q ou ESC para sair.\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.60,
        model_complexity=1
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # FPS
            fps_contador += 1
            agora = time.time()
            if agora - fps_timer >= 1.0:
                fps_atual = fps_contador / (agora - fps_timer)
                fps_contador = 0
                fps_timer = agora

            # Processar frame com MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            resultado = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            tem_mao = bool(resultado.multi_hand_landmarks)
            letra_atual, confianca_atual = None, 0.0

            if tem_mao:
                lm = resultado.multi_hand_landmarks[0]
                features = normalizar_landmarks(lm, w, h)

                # Predição com probabilidades
                probs = modelo.predict_proba(features)[0]
                idx_pred = np.argmax(probs)
                letra_raw = classes[idx_pred]
                conf_raw = probs[idx_pred]

                suavizador.atualizar(letra_raw, conf_raw)
                letra_atual, confianca_atual = suavizador.obter()

                # Adicionar ao histórico após estabilidade
                if letra_atual == ultima_letra_hist:
                    tempo_estavel += 1
                else:
                    tempo_estavel = 0
                    ultima_letra_hist = letra_atual

                if (tempo_estavel == FRAMES_PARA_HISTORICO
                        and confianca_atual >= CONFIANCA_MIN):
                    if not historico_letras or historico_letras[-1] != letra_atual:
                        historico_letras.append(letra_atual)

            else:
                suavizador.historico.clear()
                tempo_estavel = 0
                ultima_letra_hist = None

            # ── Renderização ──────────────────────────────────────────────────
            desenhar_landmarks_custom(frame, resultado, w, h)
            desenhar_painel_letra(frame, letra_atual, confianca_atual, tem_mao, w, h)
            desenhar_historico(frame, list(historico_letras), w, h)
            desenhar_hud(frame, fps_atual, w, h)

            cv2.imshow("Tradutor de LIBRAS", frame)

            # Teclas
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord('q'), ord('Q'), 27):  # Q ou ESC
                break
            elif tecla in (ord('c'), ord('C')):
                historico_letras.clear()
                ultima_letra_hist = None

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Até logo! Tradutor encerrado.")


if __name__ == "__main__":
    main()
