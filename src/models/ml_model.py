# Global model state — single source of truth across the app

model = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = None
dataset_info = None
label_encoder = None

def reset():
    """Reset all model state"""
    global model, X_train, X_test, y_train, y_test
    global feature_names, dataset_info, label_encoder
    model = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    feature_names = None
    dataset_info = None
    label_encoder = None