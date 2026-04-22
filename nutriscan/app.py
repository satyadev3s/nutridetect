import os
from pathlib import Path
from datetime import timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import Flask

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

from flask_mail import Mail

try:
    from models import db, login_manager
    import routes as routes_module
except ImportError:
    # Support both "python app.py" and package-style imports.
    from .models import db, login_manager
    from . import routes as routes_module


def create_app():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    # Load env files if present (local first, then mail folder fallback)
    load_dotenv(base_dir / '.env')
    load_dotenv(base_dir / 'mail' / 'mail.env')
    load_dotenv(project_root / 'mail.env')

    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=str(base_dir / 'static'),
        instance_path=str(base_dir / 'instance'),
        instance_relative_config=False,
    )

    secret_key = os.getenv('SECRET_KEY', 'nutridetect-dev-secret')
    app.config['SECRET_KEY'] = secret_key

    db_path = base_dir / 'instance' / 'nutridetect.db'
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path.as_posix()}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').strip().lower() == 'true'
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

    dataset_dir = project_root / 'dataset'
    app.config['MALNUTRITION_MODEL_PATH'] = str(dataset_dir / 'malnutrition_model.h5')
    app.config['MALNUTRITION_MODEL_CANDIDATE_PATHS'] = [
        str(dataset_dir / 'malnutrition_model.h5'),
        str(dataset_dir / 'malnutrition_model.keras'),
    ]

    app.config['HUMAN_MODEL_PATH'] = str(dataset_dir / 'human_model.h5')
    app.config['HUMAN_MODEL_CANDIDATE_PATHS'] = [
        str(dataset_dir / 'human_model.h5'),
        str(dataset_dir / 'human_model.keras'),
        str(dataset_dir / 'human_model_savedmodel'),
    ]

    app.config['MODEL_THRESHOLDS_PATHS'] = [str(dataset_dir / 'model_thresholds.json')]
    app.config['MODEL_LABELS_PATHS'] = [str(dataset_dir / 'malnutrition_labels.json')]

    app.config['UPLOAD_FOLDER'] = str(base_dir / 'static' / 'uploads')
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    app.config['LEGACY_DB_PATHS'] = [
        str(base_dir / 'nutridetect.db'),
        str(project_root / 'nutridetect.db'),
        str(project_root / 'instance' / 'nutridetect.db'),
    ]

    tz_name = os.getenv('APP_TIMEZONE', 'Asia/Kolkata')
    try:
        app.config['APP_TIMEZONE'] = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        app.config['APP_TIMEZONE'] = timezone.utc

    app.config['FORCE_PREDICTION_RESULT'] = None

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login_page'

    mail = Mail(app)
    routes_module.mail = mail

    app.register_blueprint(routes_module.bp)

    with app.app_context():
        routes_module.ensure_app_storage()

    return app


app = create_app()


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5002'))
    debug_enabled = os.getenv('FLASK_DEBUG', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
    app.run(debug=debug_enabled, host='0.0.0.0', port=port)
