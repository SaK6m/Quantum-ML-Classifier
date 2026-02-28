# Hybrid Quantum-Classical MNIST Classifier

This project implements a **Variational Quantum Classifier (VQC)** to distinguish between handwritten digits using a hybrid quantum-classical pipeline. By combining classical dimensionality reduction with quantum entangling circuits, the model achieves high accuracy on a 4-qubit system.

## 📊 Data Preprocessing
To make the MNIST dataset compatible with a near-term quantum simulator, I developed a preprocessing pipeline:
* **Binary Filtering**: Focused the model on distinguishing between digits **0** and **1**.
* **Standardization**: Used `StandardScaler` to center pixel values, ensuring optimal performance for dimensionality reduction.
* **PCA (Dimensionality Reduction)**: Reduced **784-pixel** images down to **4 principal components**.
* **Quantum Mapping**: Normalized features to a **[0, 1]** range using `MinMaxScaler` to represent quantum gate rotation angles (mapped to $[0, \pi]$).

## 📈 Visualizing the Latent Space
After reducing the data to 4 dimensions, the first two components show clear clustering, proving the data is separable before being fed into the quantum circuit:

![MNIST PCA Clusters](assets/mnist_pca_plot.png)

---

## ⚛️ Quantum Architecture
The model uses a 4-qubit circuit designed in Qiskit, consisting of two main functional blocks:

### 1. The Feature Map (`ZZFeatureMap`)
* **Role**: Encodes classical data into quantum states using data-dependent rotations and entangling gates.
* **Configuration**: `reps=2` with linear entanglement.

### 2. The Ansatz (`RealAmplitudes`)
* **Role**: The "trainable" layers of the Quantum Neural Network.
* **Configuration**: 4 qubits with `reps=3`, resulting in **16 trainable weights ($\theta$)**.
* **Optimization**: The COBYLA optimizer tunes these weights to maximize the separation between digit classes in the Hilbert space.

---

## 🚀 Experimental Results

### **Phase 1: Initial Benchmark**
* **Training Accuracy**: 75.0%
* **Data Scale**: 100 samples
* **Iterations**: 50

### **Phase 2: High-Fidelity Training**
By scaling the training set and increasing optimizer iterations, the model achieved near-state-of-the-art performance for a 4-qubit VQC:
* **Training Accuracy**: **92.90%**
* **Data Scale**: 1,000 samples
* **Optimizer**: COBYLA (100 iterations)
* **Result**: Successfully captured the underlying distribution of the MNIST latent space.



---

## 🧠 Inference & Lessons Learned
One of the key technical challenges solved in this project was **Label Mapping Consistency**.

* **The Problem**: During standalone inference, the model initially showed 0% accuracy due to a discrepancy in how bitstrings (e.g., `1011`) were interpreted as classical labels.
* **The Solution**: Implemented a deterministic **Parity Function** ($f: \{0,1\}^n \to \{0,1\}$) to map quantum measurements to binary classes. 
* **Model Persistence**: Optimized weights are serialized into `vqc_weights.json`, allowing for instant inference without retraining.

---

## 🛠️ Project Structure
* `notebooks/`: Lab environment for training, debugging, and visualization.
* `src/`: Core Python modules for circuit logic and standalone inference.
* `models/`: Exported model weights (JSON) for production-style reuse.
* `data/`: Preprocessed datasets (Local only).

## 💻 Usage
To run a prediction on a random sample using the trained "Quantum Brain":
```bash
python src/inference.py