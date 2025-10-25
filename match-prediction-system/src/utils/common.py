def load_config(config_file):
    import yaml
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)

def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def calculate_accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)