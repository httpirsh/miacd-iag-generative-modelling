# TP1 ArtBench-10 Generative Modeling Project

## 📁 Project Structure

```
TP1-alunos-src-only/
├── src/                              # Python utilities
│   └── artbench_local_dataset.py     # Dataset loading helper
│
├── notebooks/                        # Jupyter notebooks
│   └── ArtBench10_Student_Start_Pack.ipynb  # Main project notebook
│
├── data/                             # ⭐ Dataset & Training Splits
│   ├── artbench-10-python/
│   │   └── artbench-10-batches-py/   # Full ArtBench-10 (Python pickle format)
│   ├── artbench-10-binary/
│   │   └── artbench-10-batches-bin/  # Binary format
│   ├── ArtBench-10.csv               # Dataset metadata and labels
│   └── training_20_percent.csv       # 20% training subset definition
│
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🚀 Getting Started

### 1. Setup Environment
```bash
cd TP1-alunos-src-only/
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### 3. Data Already Available

The full ArtBench-10 dataset (363MB) is already included in the `data/` folder:
```
data/
├── artbench-10-python/artbench-10-batches-py/  (used by the notebook)
├── artbench-10-binary/artbench-10-batches-bin/
├── ArtBench-10.csv
└── training_20_percent.csv                    (training split)
```

**No additional setup needed** — the notebook will automatically load from `./data/`

### 4. Run Notebook
```bash
cd notebooks/
jupyter notebook ArtBench10_Student_Start_Pack.ipynb
```

## 📋 Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| **src/** | Reusable Python utilities for data loading and processing |
| **notebooks/** | Jupyter notebooks for experimentation and analysis |
| **data/** | ⭐ Full ArtBench-10 dataset (363MB) + training split definitions |

## 🔧 Key Components

### `src/artbench_local_dataset.py`
Provides the `load_kaggle_artbench10_splits()` function to load ArtBench-10 data from local directories.

### `notebooks/ArtBench10_Student_Start_Pack.ipynb`
Complete training pipeline including:
- ✅ Data loading and preprocessing
- ✅ PyTorch DataLoader creation
- ✅ Visualization utilities
- ✅ VAE, DCGAN, and Diffusion model implementations
- ✅ Training loops
- ✅ Evaluation metrics (FID, KID)
- ✅ Sample generation

### `data/training_20_percent.csv`
Defines the 20% training subset for rapid model development and testing.

## 📝 Project Workflow

### Phase 1: Development (on 20% subset)
1. Load data using `train_loader_from_csv`
2. Train all three models (VAE, DCGAN, Diffusion)
3. Generate samples and compute FID/KID
4. Select best model based on metrics

### Phase 2: Final Evaluation (on 100% dataset)
1. Retrain best model with ≥10 random seeds
2. Aggregate FID/KID scores (mean ± std)
3. Generate final report with visualizations

## 🎯 Expected Outcomes

For each model on full dataset (≥10 seeds):
- **FID (Fréchet Inception Distance)**: Mean ± Std
- **KID (Kernel Inception Distance)**: Mean ± Std  
- **Sample visualizations**: Grid of generated artworks
- **Training curves**: Loss progression over epochs

## 📚 References

- PyTorch: https://pytorch.org/
- ArtBench-10: https://www.kaggle.com/datasets/alexanderliao/artbench10
- FID Paper: https://arxiv.org/abs/1706.08500
- Diffusion Models: https://arxiv.org/abs/2006.11239

## ✅ Checklist

- [ ] Data directory configured with ArtBench-10
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Notebook runs without errors
- [ ] Phase 1 (20% subset) training completed
- [ ] Best model selected
- [ ] Phase 2 (100% dataset) training completed
- [ ] Final report generated

---

**Last Updated**: 2026-03-23
