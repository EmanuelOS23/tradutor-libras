# 🤟 Tradutor de LIBRAS — Alfabeto em Tempo Real

Projeto de Visão Computacional que reconhece o **alfabeto de LIBRAS** (Língua Brasileira de Sinais) em tempo real usando a câmera do computador.

**Disciplina**: Visão Computacional — Atividade III 2026.2  

---

## 📋 Estrutura do Projeto

```
Tradutor Libras/
├── coleta_dados.py       # 1️⃣ Coleta amostras de treinamento
├── treinar_modelo.py     # 2️⃣ Treina o classificador Random Forest
├── tradutor.py           # 3️⃣ Tradutor em tempo real (app principal)
├── requirements.txt      # Dependências Python
├── dados/
│   └── dataset.csv       # Dataset gerado na coleta
└── modelo/
    ├── modelo_libras.pkl # Modelo treinado
    └── classes.pkl       # Lista de classes (letras)
```

---

## ⚙️ Instalação

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar (Passo a Passo)

### Passo 1 — Coletar dados de treinamento
```bash
python coleta_dados.py
```
- Posicione a mão e faça o gesto de uma letra
- Pressione a **tecla da letra** para salvar a amostra (ex: `A` para a letra A)
- **Meta**: pelo menos 100 amostras por letra
- Pressione `Q` para sair e salvar

> **Dica para equipes**: cada membro pode rodar o script e coletar. Os dados são acumulados no CSV automaticamente!

### Passo 2 — Treinar o modelo
```bash
python treinar_modelo.py
```
- Treina o Random Forest com os dados coletados
- Exibe acurácia, matriz de confusão e gráfico por letra
- Salva o modelo em `modelo/`

### Passo 3 — Usar o tradutor
```bash
python tradutor.py
```
- Abre a câmera em tempo real
- Faça os gestos do alfabeto de LIBRAS
- A letra detectada aparece em destaque na tela

---

## ⌨️ Controles do Tradutor

| Tecla | Ação |
|-------|------|
| `Q` ou `ESC` | Sair |
| `C` | Limpar histórico de letras |

---

## 🧠 Técnicas Utilizadas

| Componente | Tecnologia |
|------------|-----------|
| Detecção da mão | **MediaPipe Hands** (Google) |
| Extração de landmarks | 21 pontos 3D por mão = 63 features |
| Normalização | Translação pelo pulso + escala |
| Classificador | **Random Forest** (scikit-learn, 200 árvores) |
| Interface | **OpenCV** |

### Pipeline
```
Câmera → MediaPipe → 21 Landmarks 3D → Normalização → Random Forest → Letra
```

---

## ⚠️ Notas Importantes

- **Letras J e Z**: possuem movimento no alfabeto real. Nesta versão, são tratadas como poses estáticas (posição final do gesto).
- **Iluminação**: prefira ambientes bem iluminados para melhor detecção
- **Fundo**: fundo neutro melhora a acurácia do MediaPipe

---

## 📦 Dependências

```
opencv-python  ≥ 4.8
mediapipe      ≥ 0.10
scikit-learn   ≥ 1.3
numpy          ≥ 1.24
pandas         ≥ 2.0
matplotlib     ≥ 3.7
seaborn        ≥ 0.12
```
