# Hybrid Quantum-Classical MNIST Classifier

This project implements a Variational Quantum Classifier (VQC) to distinguish between handwritten digits using a hybrid approach.

## 📊 Data Preprocessing
To make the MNIST dataset compatible with a 4-qubit quantum simulator, I developed a preprocessing pipeline:
* **Binary Filtering**: Focused the model on distinguishing between digits **0** and **1**.
* **Standardization**: Used `StandardScaler` to center pixel values, ensuring optimal performance for dimensionality reduction.
* **PCA (Dimensionality Reduction)**: Reduced the **784-pixel** images down to **4 principal components**.
* **Quantum Mapping**: Normalized features to a **[0, 1]** range using `MinMaxScaler` to represent quantum gate rotation angles.

## 📈 Visualizing the Latent Space
After reducing the data to 4 dimensions, the first two components show clear clustering, proving the data is separable before being fed into the quantum circuit:

![MNIST PCA Clusters](assets/mnist_pca_plot.png)

---

## ⚛️ Quantum Circuit Architecture
The model uses a 4-qubit circuit designed in Qiskit, consisting of two main functional blocks:

### 1. The Feature Map (`ZZFeatureMap`)
* **Role**: Encodes classical data into quantum states.
* **Configuration**: Uses `reps=2` and linear entanglement to capture complex relationships between the 4 input features.
* **Parameters**: 0 (This block is fixed by the input data).

### 2. The Ansatz (`RealAmplitudes`)
* **Role**: The "trainable" part of the circuit (the quantum equivalent of neural network layers).
* **Configuration**: 4 qubits with `reps=3`.
* **Parameters**: **16 trainable weights** ($\theta$). These are the "knobs" the classical optimizer turns to learn the difference between digits.

---

## 🛠️ Project Structure
* `notebooks/`: Exploratory data analysis and circuit visualization.
* `data/`: (Local only) Preprocessed `.npz` files.
* `src/`: Core logic for circuit construction and model training.

-----
## 🚀 Results
* **Training Accuracy**: 75% (using 100 samples from 14,780).
* **Optimizer**: COBYLA (50 iterations).
* **Quantum Architecture**: 4 Qubits
* **Ansatz Depth(reps)**: 3 (16 trainable parameters)

# 🚀 Result Phase 2: 
* **Training Accuracy**: 92.90% (using 1000 samples from 14,780).
* **Optimizer**: COBYLA (100 iterations).
* **Quantum Architecture**: 4 Qubits
* **Ansatz Depth(reps)**: 3 (16 trainable parameters)