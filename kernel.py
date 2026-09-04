import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

iris = load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

scaler = StandardScaler()
X = scaler.fit_transform(X)

training_sizes = [10, 20, 30, 40, 50]

classical_accuracies = []
quantum_accuracies = []

classical_times = []
quantum_times = []

for size in training_sizes:

    print("\n==============================")
    print(f"Training Size: {size}")
    print("==============================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=size,
        stratify=y,
        random_state=42
    )

    start_time = time.time()

    classical_model = SVC(kernel="rbf", gamma="scale")
    classical_model.fit(X_train, y_train)

    classical_time = time.time() - start_time
    classical_times.append(classical_time)

    y_pred_classical = classical_model.predict(X_test)
    classical_acc = accuracy_score(y_test, y_pred_classical)
    classical_accuracies.append(classical_acc)

    print(f"Classical Accuracy: {classical_acc:.4f}")
    print(f"Classical Training Time: {classical_time:.4f} sec")

    start_time = time.time()

    feature_map = zz_feature_map(
        feature_dimension=X.shape[1],
        reps=2,
        entanglement="linear"
    )
   
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    quantum_model = SVC(kernel=quantum_kernel.evaluate)
    quantum_model.fit(X_train, y_train)
    
    quantum_time = time.time() - start_time
    quantum_times.append(quantum_time)

    y_pred_quantum = quantum_model.predict(X_test)
    quantum_acc = accuracy_score(y_test, y_pred_quantum)
    quantum_accuracies.append(quantum_acc)

    print(f"Quantum Accuracy: {quantum_acc:.4f}")
    print(f"Quantum Training Time: {quantum_time:.4f} sec")

for i in range(len(training_sizes)):
    print(f"  Training Size: {training_sizes[i]}")
    print(f"  Classical Accuracy: {classical_accuracies[i]:.4f}")
    print(f"  Quantum Accuracy:   {quantum_accuracies[i]:.4f}")
    print(f"  Classical Time:     {classical_times[i]:.4f} sec")
    print(f"  Quantum Time:       {quantum_times[i]:.4f} sec")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_sizes, classical_accuracies, marker='o')
plt.plot(training_sizes, quantum_accuracies, marker='s')
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend(["Classical RBF", "Quantum Kernel"])

plt.subplot(1, 2, 2)
plt.plot(training_sizes, classical_times, marker='o')
plt.plot(training_sizes, quantum_times, marker='s')
plt.xlabel("Training Size")
plt.ylabel("Training Time (seconds)")
plt.title("Execution Time Comparison")
plt.legend(["Classical RBF", "Quantum Kernel"])

plt.tight_layout()
plt.show()
