# Predicción de Solubilidad Molecular con LSTM y GCN

**TFG — Trabajo de Fin de Grado**

Comparativa de dos arquitecturas de deep learning para predecir la solubilidad acuosa (LogS) de moléculas orgánicas usando el dataset ESOL (Delaney, 1128 moléculas).

---

## Estructura del repositorio

```
├── data/
│   └── delaney-processed.csv    # Dataset ESOL (1128 moléculas)
├── LSTM_random.ipynb            # LSTM con partición aleatoria
├── LSTM_scaffold.ipynb          # LSTM con partición scaffold
├── GCN_random.ipynb             # GCN con partición aleatoria
├── GCN_scaffold.ipynb           # GCN con partición scaffold
└── requirements.txt
```

---

## Dataset

El proyecto usa el dataset **ESOL (Delaney)**, incluido en `data/delaney-processed.csv`.

| Columna | Descripción |
|---------|-------------|
| `smiles` | Representación SMILES de la molécula |
| `measured log solubility in mols per litre` | LogS experimental (variable objetivo) |

DeepChem carga el dataset automáticamente con `dc.molnet.load_delaney()`. El fichero CSV en `data/` permite su consulta directa sin conexión.

---

## Modelos

### LSTM — Representación Secuencial
- Tokenización química personalizada de la cadena SMILES (regex consciente de química)
- Capa de embedding (dim = 64) + LSTM multicapa
- Cabeza densa: hidden → 64 → 1

### GCN — Representación Topológica (Grafos)
- Conversión SMILES → grafo molecular con RDKit
- Nodos: vector de 20 características atómicas (tipo de átomo, hibridación, carga formal, H implícitos, anillo, aromaticidad)
- Aristas: vector de 5 características del enlace (tipo, pertenece a anillo)
- Capas `GCNConv` + `GlobalMeanPool` + cabeza densa
- Interpretabilidad con **GNNExplainer**

---

## Reproducibilidad

Ejecutar los notebooks en el siguiente orden:

1. `LSTM_random.ipynb`
2. `LSTM_scaffold.ipynb`
3. `GCN_random.ipynb`
4. `GCN_scaffold.ipynb`

> ⚠️ **Los notebooks deben ejecutarse en Google Colab para reproducir exactamente los resultados del TFG.** Ejecutarlos en un entorno local puede dar resultados distintos debido a diferencias en la versión de PyTorch, uso de CPU en lugar de GPU, y variaciones numéricas entre plataformas (Windows vs Linux). Incluso con la misma semilla aleatoria (`seed=42`), estas diferencias hacen que los valores de RMSE no sean idénticos a los reportados.

---

## Instalación

```bash
pip install rdkit deepchem torch pandas numpy matplotlib pillow
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

---

## Hiperparámetros seleccionados

Elegidos mediante grid search exhaustivo sobre 27 combinaciones (código comentado en cada notebook):

| Modelo | Split    | hidden\_size | num\_layers | dropout |
|--------|----------|-------------|------------|---------|
| LSTM   | random   | 256         | 1          | 0.1     |
| LSTM   | scaffold | 256         | 3          | 0.1     |
| GCN    | random   | 256         | 3          | 0.3     |
| GCN    | scaffold | 256         | 3          | 0.2     |

Configuración común: Adam (lr = 0.001), hasta 100 épocas, early stopping con paciencia 10.

---

## Salidas generadas

Cada notebook produce:

- `*_learning_curve.png` — Curva train/val RMSE por época
- Scatter plot predicciones vs. valores reales (LogS)
- **Solo GCN scaffold**: `explain_<smiles>.png` — Mapas de importancia atómica (GNNExplainer) para las moléculas más insoluble, más soluble e intermedia del test
- **Solo GCN random**: `gcn_random_split.pth` — Pesos del modelo y constantes de normalización
- **Solo GCN scaffold**: `gcn_scaffold_split.pth` — Pesos del modelo y constantes de normalización
