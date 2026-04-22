from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import os
import src.models.ml_model as state
from src.utils.file_utils import allowed_file, validate_csv

upload_bp = Blueprint('upload', __name__)

UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@upload_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading CSV: {str(e)}'}), 400

        is_valid, errors = validate_csv(df)
        if not is_valid:
            return jsonify({'success': False, 'error': '; '.join(errors)}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        state.dataset_info = {
            'filename': filename,
            'filepath': filepath,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }

        file.seek(0)
        file.save(filepath)

        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset_info': state.dataset_info
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500