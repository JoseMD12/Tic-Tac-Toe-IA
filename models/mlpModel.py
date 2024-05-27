from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_mlp(X_train, y_train, hidden_layers=(100,)):
    # Create and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_mlp(mlp, X_test, y_test):
    # Evaluate the model
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Acurácia do MLP: {accuracy}")
    return accuracy
