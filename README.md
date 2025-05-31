# Clasificador de Sentimiento con Embeddings Preentrenados

Este proyecto implementa un clasificador de sentimiento binario (positivo/negativo) utilizando embeddings de palabras preentrenados (Word2Vec de Mikolov) y dos arquitecturas: una red neuronal feedforward (FNN) y una red recurrente (RNN/LSTM).

## Estructura del Proyecto

- `src/`: Código fuente para preprocesamiento, modelo y entrenamiento.
- `notebooks/`: Experimentación interactiva en Jupyter.
- `data/`: Dataset de reviews de texto (no incluido por peso).
- `docs/`: Visualizaciones, métricas y análisis.
- `main.py`: Script principal para entrenamiento.

## Requisitos

- Python
- Paquetes especificados en `requirements.txt`

Instalación:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
