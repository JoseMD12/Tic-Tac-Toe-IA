from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X_train, y_train, n_neighbors=3):
    # Create and train the k-NN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_knn(knn, X_test, y_test):
    # Evaluate the model
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Acurácia do k-NN: {accuracy}")
    return accuracy
