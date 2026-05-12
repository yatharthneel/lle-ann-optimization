# 🧪 ANN-Based Optimization of Liquid-Liquid Extraction

> **AI Minor Course Project** — Department of Chemical / Process Engineering, 2025–2026

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Overview

This project applies **Artificial Neural Networks (ANN)** to model and optimise the **reactive liquid-liquid extraction (LLE)** of itaconic acid. The ANN model achieves **R² = 0.993**, significantly outperforming traditional Response Surface Methodology (RSM, R² = 0.970).

### System
- **Solute:** Itaconic Acid
- **Extractant:** Tri-n-octylamine (TOA)
- **Modifier:** Dichloromethane (DCM)
- **Output:** Extraction Efficiency (% Y)

---

## 🗂️ Project Structure

```
lle-ann-optimization/
│
├── src/
│   └── ann_pipeline.py        # Full ANN pipeline (train → optimise → visualise)
│
├── docs/
│   ├── LLE_ANN_Report.pdf     # Full technical report
│   └── LLE_ANN_Presentation.pptx  # Course presentation slides
│
├── results/
│   └── lle_ann_results.png    # Output plots (generated on run)
│
├── notebooks/
│   └── (Jupyter notebooks, if added later)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yatharthneel/lle-ann-optimization.git
cd lle-ann-optimization
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

```bash
python src/ann_pipeline.py
```

This will:
1. Load the 15-experiment Box-Behnken dataset
2. Normalise inputs using Min-Max scaling
3. Train an ANN (architecture: 3 → 10 → 1)
4. Run 5-fold cross-validation
5. Optimise using Differential Evolution
6. Generate and save 6 diagnostic plots to `results/lle_ann_results.png`

---

## 📊 Results

| Metric | RSM | ANN (This Model) |
|--------|-----|-----------------|
| R² | 0.970 | **0.993** |
| Handles nonlinearity | Partially | ✅ Fully |
| Coupled with optimiser | Limited | ✅ Differential Evolution |

### Optimal Conditions Found

| Variable | Optimum | Range |
|----------|---------|-------|
| Acid Concentration | 0.072 mol/L | 0.036 – 0.108 mol/L |
| TOA % (v/v) | 16.075% | 5 – 30% |
| DCM % (v/v) | 62.15% | 30 – 90% |
| **Predicted % Y** | **~98–100%** | — |

---

## 🧠 ANN Architecture

```
Input Layer     Hidden Layer      Output Layer
(3 neurons)  →  (10 neurons)   →  (1 neuron)
Acid Conc.      tanh activation    % Extraction
TOA %
DCM %
```

- **Training algorithm:** LBFGS (quasi-Newton, equivalent to Levenberg-Marquardt for small datasets)
- **Optimiser:** Differential Evolution (global search)
- **Dataset:** 15 experiments (Box-Behnken Design)

---

## 📁 Documentation

- 📄 Full report: [`docs/LLE_ANN_Report.pdf`](docs/LLE_ANN_Report.pdf)
- 📊 Presentation slides: [`docs/LLE_ANN_Presentation.pptx`](docs/LLE_ANN_Presentation.pptx)

---

## 🔭 Future Work

- [ ] Hybrid Physics-Informed ANN (NRTL + ANN)
- [ ] Transfer learning across different LLE systems
- [ ] Digital twin of extraction column
- [ ] Autonomous self-driving lab integration

---

## 📚 Key Reference

Kumar, S., Datta, D., & Wasewar, K. L. (2021). *Statistical modeling and optimization of itaconic acid reactive extraction using RSM and ANN.* Bioresource Technology Reports, 16, 100851.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
