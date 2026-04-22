from flask import Blueprint, request, jsonify
from src.services import explain_service
import traceback

explain_bp = Blueprint('explain', __name__)


@explain_bp.route('/generate_shap', methods=['POST'])
def generate_shap():
    try:
        result = explain_service.generate_shap_summary()
        return jsonify({'success': True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/explain_instance', methods=['POST'])
def explain_instance():
    try:
        data = request.get_json()
        idx = int(data.get('instance_idx', 0))
        result = explain_service.explain_instance_shap(idx)
        return jsonify({'success': True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/generate_waterfall', methods=['POST'])
def generate_waterfall():
    try:
        data = request.get_json()
        idx = int(data.get('instance_idx', 0))
        result = explain_service.generate_waterfall(idx)
        return jsonify({'success': True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/generate_lime', methods=['POST'])
def generate_lime():
    try:
        result = explain_service.generate_lime_summary()
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@explain_bp.route('/explain_instance_lime', methods=['POST'])
def explain_instance_lime():
    try:
        data = request.get_json()
        idx = int(data.get('instance_idx', 0))
        result = explain_service.explain_instance_lime(idx)
        return jsonify({'success': True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500