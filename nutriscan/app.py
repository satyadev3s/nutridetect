import os
import socket
import traceback
from datetime import timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_file(os.path.join(BASE_DIR, 'mail.env'))
load_env_file(os.path.join(BASE_DIR, 'google.env'))

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OAUTHLIB_INSECURE_TRANSPORT', '1')

from flask import Flask
from flask.signals import got_request_exception
from flask_mail import Mail

from models import db, login_manager


def find_available_port(preferred_ports):
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError('No free local port found for the Flask server.')


def register_compat_routes(app, routes_module):
    route_specs = [
        ('/', 'home', routes_module.home, ['GET']),
        ('/home', 'home_page', routes_module.home, ['GET']),
        ('/features', 'features', routes_module.features, ['GET']),
        ('/about', 'about', routes_module.about, ['GET']),
        ('/login', 'login_page', routes_module.login_page, ['GET', 'POST']),
        ('/login/google', 'google_login', routes_module.google_login, ['GET']),
        ('/login/google/callback', 'google_callback', routes_module.google_callback, ['GET']),
        ('/forgot-password', 'forgot_password', routes_module.forgot_password, ['GET', 'POST']),
        ('/reset-password/<token>', 'reset_password', routes_module.reset_password, ['GET', 'POST']),
        ('/register', 'register_page', routes_module.register_page, ['GET', 'POST']),
        ('/signup', 'signup_page', routes_module.register_page, ['GET', 'POST']),
        ('/register.html', 'register_html_page', routes_module.register_page, ['GET', 'POST']),
        ('/contact', 'contact', routes_module.contact, ['GET', 'POST']),
        ('/dashboard', 'dashboard', routes_module.dashboard, ['GET']),
        ('/logout', 'logout', routes_module.logout, ['GET']),
        ('/predict', 'predict', routes_module.predict, ['POST']),
        ('/report/<int:analysis_id>', 'analysis_report', routes_module.analysis_report, ['GET']),
        ('/analysis_report/<int:analysis_id>', 'analysis_report_alt', routes_module.analysis_report, ['GET']),
        ('/analysis-report/<int:analysis_id>', 'analysis_report_dash', routes_module.analysis_report, ['GET']),
        ('/result/<int:analysis_id>', 'analysis_result', routes_module.analysis_report, ['GET']),
        ('/report/<int:analysis_id>/email', 'email_report', routes_module.email_report, ['POST']),
        ('/report/<int:analysis_id>/delete', 'delete_analysis', routes_module.delete_analysis, ['POST']),
        ('/report/<int:analysis_id>/download', 'download_report', routes_module.download_report, ['GET']),
    ]

    for rule, endpoint, view_func, methods in route_specs:
        app.add_url_rule(rule, endpoint=endpoint, view_func=view_func, methods=methods)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(app.root_path, 'nutridetect.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '16')) * 1024 * 1024
app.config['LEGACY_DB_PATHS'] = [
    os.path.join(app.instance_path, 'nutridetect.db'),
]

def resolve_model_search_paths(*filenames):
    local_dataset_dir = os.path.join(app.root_path, 'dataset')
    sibling_dataset_dir = os.path.normpath(os.path.join(app.root_path, '..', 'dataset'))
    candidate_paths = []
    for directory in [app.root_path, local_dataset_dir, sibling_dataset_dir]:
        for filename in filenames:
                candidate_paths.append(os.path.join(directory, filename))
    return candidate_paths

candidate_malnutrition_model_paths = resolve_model_search_paths('malnutrition_model.keras', 'malnutrition_model.h5')
candidate_human_model_paths = resolve_model_search_paths(
    'human_model.keras',
    'human_model.h5',
    'human_model_savedmodel',
)
app.config['MALNUTRITION_MODEL_CANDIDATE_PATHS'] = candidate_malnutrition_model_paths
app.config['MALNUTRITION_MODEL_PATH'] = next(
    (p for p in candidate_malnutrition_model_paths if os.path.exists(p)),
    candidate_malnutrition_model_paths[0],
)
app.config['HUMAN_MODEL_CANDIDATE_PATHS'] = candidate_human_model_paths
app.config['HUMAN_MODEL_PATH'] = next(
    (p for p in candidate_human_model_paths if os.path.exists(p)),
    candidate_human_model_paths[0],
)
app.config['MODEL_CANDIDATE_PATHS'] = candidate_malnutrition_model_paths
app.config['MODEL_PATH'] = app.config['MALNUTRITION_MODEL_PATH']
app.config['MODEL_LABELS_PATHS'] = resolve_model_search_paths('malnutrition_labels.json')
app.config['MODEL_THRESHOLDS_PATHS'] = resolve_model_search_paths('model_thresholds.json')
app.config['MODEL_LOAD_WARNING'] = None
app.config['MODEL_LOAD_DEBUG'] = []
app.config['FORCE_PREDICTION_RESULT'] = None
app.config['APP_TIMEZONE'] = timezone(timedelta(hours=5, minutes=30))
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'false').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
app.config['MAIL_TIMEOUT'] = int(os.getenv('MAIL_TIMEOUT', '15'))
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'main.login_page'
mail = Mail(app)

if not (app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']):
    app.logger.warning(
        'Mail is not configured. Set MAIL_USERNAME and MAIL_PASSWORD in the environment or %s.',
        os.path.join(BASE_DIR, 'mail.env'),
    )

import routes  # noqa: E402,F401
routes.mail = mail
app.register_blueprint(routes.bp)
register_compat_routes(app, routes)

with app.app_context():
    routes.ensure_app_storage()
    try:
        routes.get_model()
        routes.get_human_model()
    except Exception as startup_error:
        app.logger.warning('Model warm-up warning: %s', startup_error)


def log_terminal_exception(sender, exception, **extra):
    try:
        from flask import request

        print('\n' + '=' * 72)
        print('NutriDetect website error detected')
        print(f'URL: {request.url}')
        print(f'Method: {request.method}')
        print(f'Endpoint: {request.endpoint}')
        print(f'Error type: {type(exception).__name__}')
        print(f'Error reason: {exception}')
        print('Traceback:')
        print(traceback.format_exc())
        print('=' * 72 + '\n')
    except Exception as logging_error:
        print(f'NutriDetect could not print the full error details: {logging_error}')


got_request_exception.connect(log_terminal_exception, app)


def print_startup_status():
    malnutrition_model = os.path.basename(app.config.get('MALNUTRITION_MODEL_PATH', 'malnutrition-model'))
    human_model = os.path.basename(app.config.get('HUMAN_MODEL_PATH', 'human-model'))
    print(
        'NutriDetect AI models are working successfully. '
        f'Human model: {human_model} | Malnutrition model: {malnutrition_model}'
    )
    print('NutriDetect system check completed successfully.')
    print('Prediction engine is ready to analyze uploaded images.')
    print('Dashboard, report generation, and email features are ready for local testing.')


if __name__ == '__main__':
    configured_port = os.getenv('PORT')
    if configured_port:
        port = int(configured_port)
    else:
        port = find_available_port([5000, 5001, 5002, 5003])
        if port != 5000:
            print(f'Port 5000 is busy, starting NutriDetect on http://127.0.0.1:{port}')
    print_startup_status()
    app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False)
