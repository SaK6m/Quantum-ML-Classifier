from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

def get_quantum_Circuits(num_qubits=4, reps=3):
    """ 
    Creates a hybrid quantum circuit for binary classification.
    """

    #Mapping 4PCS to 4 qubits
    # This 'ZZFeatureMAp' is great at capturig relationshops between features
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement = 'linear')

    #'RealAmplitudes" uses rotation gates (Ry) and CNOT to create entanglement
    ansatz = RealAmplitudes(num_qubits, reps=reps)
    
    return feature_map, ansatz

if __name__ == "__main__":

    fm,ans = get_quantum_Circuits()
    print(f"Feature Map Qubits: {fm.num_qubits}, Ansatz Parameters: {ans.num_parameters}")
 
    circuit = fm.compose(ans)
    print("Circuit successfully initialized!")