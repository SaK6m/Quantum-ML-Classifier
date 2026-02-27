import numpy as np
import os

from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
# from qiskit.primitives import Sampler
from qiskit.primitives import StatevectorSampler as Sampler # for local CPU rather than a real quantum ahrdware provider

from circuits import get_quantum_Circuits

algorithm_globals.random_seed = 42

def train_model():
    # Load
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data/mnist_01_quantum.npz')

    data = np.load(data_path)
    # X with 4 values (PCA) and Y with 0 or 1 (1 value)
    X, y = data['X'], data['y']

    X_train, y_train = X[:100], y[:100].astype(int)

    # get Quantum Architecture
    feature_map, ansatz = get_quantum_Circuits(num_qubits=4, reps=3)

    # Setup the Optimizer and Sampler
    # COBYLA does not need to know the 'slope' of the math
    optimizer = COBYLA(maxiter=50)
    sampler = Sampler()

    vqc = VQC(feature_map=feature_map,
              ansatz=ansatz,
              optimizer=optimizer,
              sampler=sampler)
    
    print("Starting training...")
    vqc.fit(X_train, y_train)
    print("Training completed!")

    score = vqc.score(X_train, y_train)
    print(f"Training Accuracy: {score * 100:.2f}%")

if __name__ == "__main__":
    train_model()
