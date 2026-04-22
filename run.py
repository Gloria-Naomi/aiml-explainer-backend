import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import create_app

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        from src.app import db
        db.create_all()
app.run(host='0.0.0.0', port=8080)