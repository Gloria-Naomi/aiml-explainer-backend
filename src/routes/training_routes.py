from flask import Blueprint, request, jsonify
from src.services import model_service

training_bp = Blueprint('training', __name__)


@training_bp.route('/train_model', methods=['POST'])
def train_model():
    try:
        result = model_service.train_on_iris()
        return jsonify({'success': True, 'message': 'Model trained successfully', **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/train_custom_model', methods=['POST'])
def train_custom_model():
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column not specified'}), 400

        result = model_service.train_on_custom_dataset(target_column)
        return jsonify({'success': True, 'message': 'Model trained on custom dataset', **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/get_performance_metrics', methods=['POST'])
def get_performance_metrics():
    try:
        result = model_service.get_performance_metrics()
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500