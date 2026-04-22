import pandas as pd

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv(df):
    """Validate uploaded CSV file"""
    errors = []

    if df.empty:
        errors.append("CSV file is empty")
        return False, errors

    if len(df) < 10:
        errors.append("Dataset must have at least 10 rows")

    if len(df.columns) < 2:
        errors.append("Dataset must have at least 2 columns")

    if df.columns.isnull().any():
        errors.append("All columns must have names")

    return len(errors) == 0, errors