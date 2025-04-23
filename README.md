# 📊 SVM Optimization Assignment

This project demonstrates the optimization of Support Vector Machines (SVM) using a multi-class dataset from the UCI repository. It involves sampling, hyperparameter tuning over multiple iterations, and visualizing convergence.

---

## 📁 Dataset
- **Name**: Wine Quality Dataset (Red)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
- **Classes**: Wine quality score (0 to 10)
- **Size**: ~1.6k rows × 11 features

---

## ⚙️ Methodology

### 1. **Preprocessing**
- Loaded dataset using `pandas`
- Normalized the features using `StandardScaler`

### 2. **Sampling**
- Created 10 random samples using `train_test_split` with a 70-30 train-test ratio
- Used different random states (0 to 9) to ensure variety

### 3. **Optimization**
- SVM models trained over 100 iterations per sample
- Kernels used: `linear`, `poly`, `rbf`, `sigmoid`
- For each iteration, random values for:
  - `C` (Nu): [0.1, 10]
  - `gamma` (Epsilon): [0.01, 10]
- Best accuracy and associated hyperparameters tracked per sample

### 4. **Evaluation**
- Identified the sample with highest accuracy
- Plotted convergence using `learning_curve` on that best-performing SVM model

---

## 📈 Results

### Table 1: Comparative Performance of Optimized SVM Samples

| Sample # | Best Accuracy | Best Kernel | Best Nu (C) | Best Epsilon (Gamma) |
|----------|----------------|--------------|-------------|-----------------------|
| S1–S10   | Auto-filled based on run | Kernel with highest performance | Optimized `C` | Optimized `gamma` |

> 📁 Saved at: `results/table1_best_accuracies.csv`

---

### 📉 Convergence Graph

![Convergence Plot](results/figure1_convergence_plot.png)

- **X-axis**: Iterations (Training Set Size %)
- **Y-axis**: Accuracy
- **Curves**:
  - Training accuracy
  - Cross-validation accuracy
- **Purpose**: Shows how the best model's accuracy evolves with increasing training data

---

## 💻 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
