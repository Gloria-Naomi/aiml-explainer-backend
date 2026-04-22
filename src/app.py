"""
SP-105 AIML Explainer - Main Flask Application
A web-based application for visualizing black-box ML model explanations
"""

from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import os
from src.config import config

# Initialize extensions (without app, for app factory pattern)
db = SQLAlchemy()

def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    Session(app)

    # Register blueprints
    from src.routes.upload_routes import upload_bp
    from src.routes.training_routes import training_bp
    from src.routes.explain_routes import explain_bp
    from src.routes.report_routes import report_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(explain_bp)
    app.register_blueprint(report_bp)

    # Core routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health_check():
        return jsonify({'status': 'healthy', 'message': 'AIML Explainer is running'})

    @app.route('/info')
    def info():
        return jsonify({
            'project': 'AIML Explainer',
            'author': 'Gloria Kouam',
            'description': 'Backend system for ML data processing and explainability',
            'version': '2.0.0'
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return render_template('error.html', error='Page not found'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', error='Internal server error'), 500

    return app


if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        db.create_all()
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )