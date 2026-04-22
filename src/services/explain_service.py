import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import io
import base64
import src.models.ml_model as state


def _plot_to_base64():
    """Convert current matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    return img_str


def generate_shap_summary():
    """Generate SHAP summary plot and feature importance"""
    if state.model is None:
        raise ValueError("No model trained")

    explainer = shap.TreeExplainer(state.model)
    shap_values = explainer.shap_values(state.X_test)

    shap_values_for_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, state.X_test,
                      feature_names=state.feature_names,
                      show=False, max_display=state.X_test.shape[1])
    img_str = _plot_to_base64()

    # Feature importance
    if isinstance(shap_values, list):
        feature_importance = np.array([
            np.abs(sv).mean(axis=0) for sv in shap_values
        ]).mean(axis=0)
    else:
        feature_importance = np.abs(shap_values_for_plot).mean(axis=0)

    if feature_importance.ndim > 1:
        feature_importance = feature_importance.flatten()

    # Align length with feature names
    n = len(state.feature_names)
    if len(feature_importance) > n:
        feature_importance = feature_importance[:n]
    elif len(feature_importance) < n:
        feature_importance = np.pad(feature_importance, (0, n - len(feature_importance)))

    importance_data = sorted([
        {'feature': str(state.feature_names[i]), 'importance': float(feature_importance[i])}
        for i in range(n)
    ], key=lambda x: x['importance'], reverse=True)

    return {
        'plot_image': img_str,
        'feature_importance': importance_data,
        'n_samples_explained': len(state.X_test)
    }


def explain_instance_shap(instance_idx):
    """Generate SHAP force plot for a single instance"""
    if state.model is None:
        raise ValueError("No model trained")
    if instance_idx >= len(state.X_test):
        raise IndexError(f"Instance index out of range. Max: {len(state.X_test) - 1}")

    instance = state.X_test.iloc[instance_idx:instance_idx + 1]
    explainer = shap.TreeExplainer(state.model)
    shap_values_obj = explainer(instance)

    plt.figure(figsize=(12, 4))
    if hasattr(shap_values_obj, 'values') and len(shap_values_obj.values.shape) > 2:
        shap.plots.force(shap_values_obj[0, :, 0], matplotlib=True, show=False)
    else:
        shap.plots.force(shap_values_obj[0], matplotlib=True, show=False)
    img_str = _plot_to_base64()

    return {
        'plot_image': img_str,
        'prediction': int(state.model.predict(instance)[0]),
        'prediction_proba': state.model.predict_proba(instance)[0].tolist(),
        'instance_data': instance.to_dict('records')[0]
    }


def generate_waterfall(instance_idx):
    """Generate SHAP waterfall plot for a single instance"""
    if state.model is None:
        raise ValueError("No model trained")
    if instance_idx >= len(state.X_test):
        raise IndexError(f"Instance index out of range. Max: {len(state.X_test) - 1}")

    instance = state.X_test.iloc[instance_idx:instance_idx + 1]
    explainer = shap.TreeExplainer(state.model)
    shap_values = explainer(instance)

    plt.figure(figsize=(10, 6))
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
    img_str = _plot_to_base64()

    return {
        'plot_image': img_str,
        'prediction': int(state.model.predict(instance)[0]),
        'prediction_proba': state.model.predict_proba(instance)[0].tolist(),
        'instance_idx': instance_idx
    }


def generate_lime_summary():
    """Generate LIME explanations across multiple instances"""
    if state.model is None:
        raise ValueError("No model trained")

    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        state.X_train.values,
        feature_names=state.feature_names,
        class_names=[str(i) for i in range(len(np.unique(state.y_train)))],
        mode='classification',
        random_state=42
    )

    num_instances = min(5, len(state.X_test))
    lime_explanations = []

    for i in range(num_instances):
        instance = state.X_test.iloc[i].values
        exp = explainer.explain_instance(instance, state.model.predict_proba,
                                         num_features=len(state.feature_names))
        lime_explanations.append({
            'instance_idx': i,
            'prediction': int(state.model.predict([instance])[0]),
            'features': [{'feature': f, 'weight': float(w)} for f, w in exp.as_list()]
        })

    # Aggregate importance
    feature_importance = {f: 0 for f in state.feature_names}
    for exp in lime_explanations:
        for item in exp['features']:
            feat_name = item['feature'].split()[0]
            if feat_name in feature_importance:
                feature_importance[feat_name] += abs(item['weight'])

    total = sum(feature_importance.values())
    if total > 0:
        feature_importance = {k: v / total for k, v in feature_importance.items()}

    importance_data = sorted([
        {'feature': k, 'importance': float(v)}
        for k, v in feature_importance.items()
    ], key=lambda x: x['importance'], reverse=True)

    return {
        'feature_importance': importance_data,
        'sample_explanations': lime_explanations[:3],
        'n_instances_explained': num_instances
    }


def explain_instance_lime(instance_idx):
    """Generate LIME explanation for a single instance"""
    if state.model is None:
        raise ValueError("No model trained")
    if instance_idx >= len(state.X_test):
        raise IndexError(f"Instance index out of range. Max: {len(state.X_test) - 1}")

    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        state.X_train.values,
        feature_names=state.feature_names,
        class_names=[str(i) for i in range(len(np.unique(state.y_train)))],
        mode='classification',
        random_state=42
    )

    instance = state.X_test.iloc[instance_idx].values
    exp = explainer.explain_instance(instance, state.model.predict_proba,
                                     num_features=len(state.feature_names))

    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)

    return {
        'plot_image': img_str,
        'prediction': int(state.model.predict([instance])[0]),
        'prediction_proba': state.model.predict_proba([instance])[0].tolist(),
        'explanation': [{'feature': f, 'weight': float(w)} for f, w in exp.as_list()],
        'instance_data': state.X_test.iloc[instance_idx].to_dict()
    }