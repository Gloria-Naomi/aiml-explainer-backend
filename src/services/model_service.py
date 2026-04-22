import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import src.models.ml_model as state


def preprocess_data(df, target_column):
    """Preprocess the uploaded dataset"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode categorical target
    if y.dtype == 'object':
        state.label_encoder = LabelEncoder()
        y = state.label_encoder.fit_transform(y)
    else:
        state.label_encoder = None

    # Fill missing values
    X = X.fillna(X.mean(numeric_only=True))

    return X, y


def train_on_custom_dataset(target_column):
    """Train RandomForest on the uploaded dataset"""
    if state.dataset_info is None:
        raise ValueError("No dataset uploaded")

    df = pd.read_csv(state.dataset_info['filepath'])
    X, y = preprocess_data(df, target_column)

    state.feature_names = X.columns.tolist()
    state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    state.model = RandomForestClassifier(n_estimators=100, random_state=42)
    state.model.fit(state.X_train, state.y_train)

    accuracy = state.model.score(state.X_test, state.y_test)

    return {
        'accuracy': float(accuracy),
        'n_samples': len(state.X_train),
        'n_features': len(state.feature_names),
        'features': state.feature_names,
        'n_classes': len(np.unique(y))
    }


def train_on_iris():
    """Train RandomForest on the built-in Iris dataset"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    state.feature_names = iris.feature_names

    state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    state.model = RandomForestClassifier(n_estimators=100, random_state=42)
    state.model.fit(state.X_train, state.y_train)

    accuracy = state.model.score(state.X_test, state.y_test)

    return {
        'accuracy': float(accuracy),
        'n_samples': len(state.X_train),
        'n_features': len(state.feature_names)
    }


def get_performance_metrics():
    """Return confusion matrix, accuracy, and per-class metrics"""
    if state.model is None:
        raise ValueError("No model trained")

    y_pred = state.model.predict(state.X_test)
    accuracy = accuracy_score(state.y_test, y_pred)
    conf_matrix = confusion_matrix(state.y_test, y_pred)
    class_report = classification_report(state.y_test, y_pred, output_dict=True)

    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(len(np.unique(state.y_test))),
                yticklabels=range(len(np.unique(state.y_test))))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close()

    metrics_by_class = {
        f'Class {k}': {
            'precision': f"{v['precision']:.3f}",
            'recall': f"{v['recall']:.3f}",
            'f1-score': f"{v['f1-score']:.3f}",
            'support': int(v['support'])
        }
        for k, v in class_report.items()
        if k not in ['accuracy', 'macro avg', 'weighted avg']
    }

    return {
        'confusion_matrix_img': img_str,
        'accuracy': float(accuracy),
        'metrics_by_class': metrics_by_class,
        'macro_avg': {
            'precision': f"{class_report['macro avg']['precision']:.3f}",
            'recall': f"{class_report['macro avg']['recall']:.3f}",
            'f1-score': f"{class_report['macro avg']['f1-score']:.3f}"
        }
    }
