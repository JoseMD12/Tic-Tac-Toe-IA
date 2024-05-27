from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train, y_train, max_depth=None):
    # Create and train the Decision Tree model
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt

def evaluate_decision_tree(dt, X_test, y_test):
    # Evaluate the model
    predictions = dt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Acurácia da Árvore de Decisão: {accuracy}")
    return accuracy
