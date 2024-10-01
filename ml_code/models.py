from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:

    @staticmethod
    def get_model(model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=500)
            # ROC curve = 0.72
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier()
            # ROC curve = 0.59
        elif model_type == 'random_forest':
            return RandomForestClassifier()
            # ROC curve = 0.78
        else:
            raise ValueError(f"Unknown model type: {model_type}")