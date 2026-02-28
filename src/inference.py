import json
import os
import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from circuits import get_quantum_circuits

def parity(x):
    return "{:b}".format(x).count("1") % 2

def run_inference():
    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, '..', 'models', 'vqc_weights_from_ipynb.json')
    data_path = os.path.join(current_dir, '..', 'data', 'mnist_01_quantum.npz')

    # 2. Load the trained weights
    with open(weights_path, 'r') as f:
        weights = np.array(json.load(f))

    # 3. Reconstruct the Quantum Architecture
    # Must use same reps as training!
    feature_map, ansatz = get_quantum_circuits(num_qubits=4, reps=3)
    
    # Initialize VQC with the weights
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        sampler=StatevectorSampler(),
        interpret=parity,
        output_shape=2,
        warm_start=True # This allows us to inject weights
    )
    
    # Manually set the internal weights
    vqc._fit_result = type('obj', (object,), {'x': weights})
    vqc._is_fitted = True

    # 4. Load Data and pick a random sample
    data = np.load(data_path)
    X, y = data['X'], data['y']
    
    # Let's pick 5 random images from the end of the dataset
    random_indices = np.random.randint(1000, len(X), size=5)
    samples = X[random_indices]
    actual_labels = y[random_indices].astype(int)

    # 5. Predict!
    print("Quantum Brain is thinking...")
    pred_value = vqc.predict(samples)

    predictions = 1 - pred_value  # Invert predictions to match original labels (0 and 1) if needed

    # 6. Show Results
    print("\n--- Inference Results ---")
    for i in range(5):
        # We use .item() or [0] to get the single number out of the array
        pred_value = predictions[i][0] if isinstance(predictions[i], np.ndarray) else predictions[i]
        label_value = actual_labels[i]
        
        status = "CORRECT!!" if pred_value == label_value else "WRONG!!"
        print(f"Sample {i+1}: Predicted {pred_value} | Actual {label_value} | {status}")

if __name__ == "__main__":
    run_inference()