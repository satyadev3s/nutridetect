import os
import shutil
import json
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
import requests
import tensorflow as tf
import cv2
from PIL import Image
from flask import Blueprint, abort, current_app, flash, make_response, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_mail import Message
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from models import Analysis, Contact, User, db, get_authenticated_user

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

bp = Blueprint('main', __name__)
mail = None
model = None
human_model = None
mal_class_indices = None
threshold_config = None
GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
GOOGLE_USERINFO_URL = 'https://www.googleapis.com/oauth2/v3/userinfo'

DEFAULT_THRESHOLDS = {
    'human_threshold': 0.5,
    'face_fallback_threshold': 0.15,
    'malnutrition_threshold': 0.5,
    'malnutrition_uncertain_margin': 0.12,
}


def format_mail_error(error_message):
    if not error_message:
        return 'Email delivery failed. Please try again later.'

    normalized = str(error_message).lower()
    if 'username and password not accepted' in normalized or 'badcredentials' in normalized:
        return (
            'Email login failed. Update MAIL_USERNAME and MAIL_PASSWORD with a valid Gmail app password, '
            'then restart the app.'
        )
    if 'timed out' in normalized:
        return 'Email delivery timed out. Please try again in a moment.'
    if 'connection unexpectedly closed' in normalized:
        return 'Email connection closed unexpectedly. Please try again.'
    return str(error_message)


def get_model():
    global model
    if model is None:
        load_errors = []
        requested_model_path = current_app.config['MALNUTRITION_MODEL_PATH']
        for model_path in current_app.config.get(
            'MALNUTRITION_MODEL_CANDIDATE_PATHS',
            [current_app.config['MALNUTRITION_MODEL_PATH']],
        ):
            if not os.path.exists(model_path):
                continue
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                current_app.logger.info('Loaded model from %s', model_path)
                current_app.config['MALNUTRITION_MODEL_PATH'] = model_path
                current_app.config['MODEL_PATH'] = model_path
                if model_path != requested_model_path:
                    current_app.config['MODEL_LOAD_WARNING'] = (
                        f'Primary malnutrition model could not be loaded, so NutriDetect used the fallback model '
                        f'"{os.path.basename(model_path)}".'
                    )
                break
            except Exception as exc:
                load_errors.append(f'{model_path}: {exc}')
        if model is None:
            warning = 'AI image model is unavailable right now, so NutriDetect is using BMI-based analysis.'
            existing_warning = current_app.config.get('MODEL_LOAD_WARNING')
            current_app.config['MODEL_LOAD_WARNING'] = (
                f'{existing_warning} {warning}'.strip() if existing_warning else warning
            )
            current_app.logger.warning('Malnutrition model load failed: %s', ' | '.join(load_errors))
            model = False
    return model


def get_human_model():
    global human_model
    if human_model is None:
        load_errors = []
        requested_model_path = current_app.config['HUMAN_MODEL_PATH']
        for model_path in current_app.config.get(
            'HUMAN_MODEL_CANDIDATE_PATHS',
            [current_app.config['HUMAN_MODEL_PATH']],
        ):
            if not os.path.exists(model_path):
                continue
            try:
                human_model = tf.keras.models.load_model(model_path, compile=False)
                current_app.logger.info(
                    'Loaded human model from %s using %s',
                    model_path,
                    type(human_model).__name__,
                )
                current_app.config['HUMAN_MODEL_PATH'] = model_path
                if model_path != requested_model_path and os.path.basename(model_path) != 'human_model_savedmodel':
                    warning = (
                        f'Primary human-detection model could not be loaded, so NutriDetect used the fallback model '
                        f'"{os.path.basename(model_path)}".'
                    )
                    existing_warning = current_app.config.get('MODEL_LOAD_WARNING')
                    current_app.config['MODEL_LOAD_WARNING'] = (
                        f'{existing_warning} {warning}'.strip() if existing_warning else warning
                    )
                break
            except Exception as exc:
                load_errors.append(f'{model_path}: {exc}')
        if human_model is None:
            warning = 'Human image verification is limited right now, so NutriDetect is using face-detection fallback.'
            existing_warning = current_app.config.get('MODEL_LOAD_WARNING')
            current_app.config['MODEL_LOAD_WARNING'] = (
                f'{existing_warning} {warning}'.strip() if existing_warning else warning
            )
            current_app.logger.warning('Human model load failed: %s', ' | '.join(load_errors))
            human_model = False
    return human_model


def predict_human_probability(loaded_human_model, img_array):
    img_array = np.asarray(img_array, dtype=np.float32)

    if hasattr(loaded_human_model, 'predict'):
        raw_output = loaded_human_model.predict(img_array, verbose=0)
        return safe_probability(np.asarray(raw_output).reshape(-1)[0])

    serving_fn = getattr(loaded_human_model, 'signatures', {}).get('serving_default')
    if serving_fn is None:
        raise TypeError('Human model does not expose a predict method or serving_default signature.')

    output_map = serving_fn(tf.constant(img_array))
    first_tensor = next(iter(output_map.values()))
    return safe_probability(np.asarray(first_tensor).reshape(-1)[0])


def load_json_file(candidate_paths, default):
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        except Exception:
            current_app.logger.exception('Unable to load JSON file from %s', path)
    return default


def get_threshold_config():
    global threshold_config
    if threshold_config is None:
        loaded = load_json_file(current_app.config.get('MODEL_THRESHOLDS_PATHS', []), {})
        threshold_config = {**DEFAULT_THRESHOLDS, **loaded}
    return threshold_config


def get_malnutrition_class_indices():
    global mal_class_indices
    if mal_class_indices is None:
        loaded = load_json_file(current_app.config.get('MODEL_LABELS_PATHS', []), {})
        mal_class_indices = loaded.get('class_indices', {})
    return mal_class_indices


def verify_user_password(user, password):
    stored_password = (user.password_hash or '').strip()
    if not stored_password:
        return False

    if stored_password.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_password, password)
    return stored_password == password


def is_google_login_configured():
    return bool(
        current_app.config.get('GOOGLE_CLIENT_ID') and current_app.config.get('GOOGLE_CLIENT_SECRET')
    )


def build_google_redirect_uri():
    return url_for('main.google_callback', _external=True)


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def safe_probability(value):
    return float(np.clip(value, 0.0, 1.0))


def detect_face_count(image_path):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame = cv2.imread(image_path)
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def get_malnutrition_probability(raw_probability):
    class_indices = get_malnutrition_class_indices()
    malnutrition_index = class_indices.get('malnutrition', 0)
    if malnutrition_index == 1:
        return raw_probability
    return 1.0 - raw_probability


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def get_bmi_category(bmi):
    if bmi is None:
        return 'Not Provided'
    if bmi < 18.5:
        return 'Underweight'
    if bmi < 25:
        return 'Normal'
    if bmi < 30:
        return 'Overweight'
    return 'Obese'


def derive_score_from_bmi(bmi):
    if bmi is None:
        return None
    if bmi < 16:
        return 0.95
    if bmi < 17:
        return 0.85
    if bmi < 18.5:
        return 0.68
    if bmi < 25:
        return 0.14
    if bmi < 30:
        return 0.24
    return 0.34


def combine_prediction_scores(image_score, bmi):
    bmi_score = derive_score_from_bmi(bmi)
    if bmi_score is None:
        return image_score, 'Uploaded photo screening model'

    # Blend the visual model with BMI so both inputs influence the final result.
    combined_score = (image_score * 0.6) + (bmi_score * 0.4)
    return combined_score, 'Combined image screening and BMI analysis'


def guidance_from_result(score, bmi):
    is_malnourished = score > 0.5
    risk_level = 'High' if score > 0.75 else 'Moderate' if score > 0.5 else 'Low'

    if bmi is None and is_malnourished:
        protein_recommendation = 'Increase protein intake to about 65-80 g/day with dal, eggs, milk, curd, paneer, soy, fish, or chicken.'
        calorie_recommendation = 'Increase calories to around 2200-2600 kcal/day using 3 meals and 2-3 nutritious snacks.'
        precautions = [
            'The analysis suggests undernutrition risk, but body measurements were not provided.',
            'Enter height and weight next time for a more accurate plan.',
            'Avoid skipping meals and long gaps between eating.',
        ]
        foods_to_eat = [
            'Choose energy-dense foods like bananas, peanut butter, khichdi, curd rice, nuts, milk, and eggs.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Consult a doctor or dietitian if you have recent weight loss, weakness, or poor appetite.',
        ]
        return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice

    if bmi is None and risk_level == 'Moderate':
        protein_recommendation = 'Keep protein around 55-70 g/day with balanced meals and one extra protein snack.'
        calorie_recommendation = 'Aim for roughly 2000-2300 kcal/day until a more accurate assessment is available.'
        precautions = [
            'This result is borderline and based mainly on the screening photo.',
            'Use a clearer image and add height and weight for a more accurate assessment.',
        ]
        foods_to_eat = [
            'Add milk, curd, sprouts, dal, eggs, fruit, and nuts to regular meals.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Get medical advice if tiredness, weakness, or appetite loss continues.',
        ]
        return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice

    if bmi is None:
        protein_recommendation = 'Maintain protein intake around 50-60 g/day with regular balanced meals.'
        calorie_recommendation = 'Maintain calorie intake around 1800-2200 kcal/day depending on daily activity.'
        precautions = [
            'This result is based on image screening only because BMI details were not provided.',
            'For a more reliable nutrition assessment, enter height and weight along with the photo.',
        ]
        foods_to_eat = [
            'Follow regular balanced meals with cereals, pulses, vegetables, fruit, milk, and enough fluids.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Consult a doctor if you have visible weight loss, weakness, appetite changes, or repeated illness.',
        ]
        return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice

    return None


def build_prediction_details(score, bmi=None, analysis_status=None, assessment_basis=None):
    forced_result = current_app.config.get('FORCE_PREDICTION_RESULT')
    if forced_result == 'Not Malnourished':
        score = 0.10
    elif forced_result == 'Malnourished':
        score = 0.90

    confidence = round(max(score, 1 - score) * 100, 1)
    if analysis_status == 'non_human':
        result = 'Image does not appear to contain a person'
        risk_level = 'Retake'
        risk_emoji = '🟠'
        nutritional_status = 'Not Assessed'
        bmi_category = get_bmi_category(bmi)
        bmi_value = round(bmi, 1) if bmi is not None else 'Not Available'
        protein_recommendation = 'Upload a clear photo with one person visible from head to toe for analysis.'
        calorie_recommendation = 'No calorie recommendation is available until a valid person photo is uploaded.'
        precautions = [
            'The uploaded image did not pass the human-presence check.',
            'Avoid group photos, blurred images, or pictures of objects only.',
        ]
        foods_to_eat = [
            'Retake the photo in good lighting with the full body visible.',
        ]
        doctor_advice = [
            'If you need nutrition support right away, consult a doctor or dietitian directly instead of relying on image screening.',
        ]
        recommendations = [*precautions, *foods_to_eat, *doctor_advice]
        return {
            'result': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'nutritional_status': nutritional_status,
            'bmi_category': bmi_category,
            'bmi_value': bmi_value,
            'assessment_basis': assessment_basis or 'Human-detection screening',
            'protein_recommendation': protein_recommendation,
            'calorie_recommendation': calorie_recommendation,
            'precautions': precautions,
            'foods_to_eat': foods_to_eat,
            'doctor_advice': doctor_advice,
            'recommendations': recommendations,
        }
    if analysis_status == 'uncertain':
        result = 'Uncertain nutrition result'
        risk_level = 'Retake'
        risk_emoji = '🟠'
        nutritional_status = 'Needs Better Image'
        bmi_category = get_bmi_category(bmi)
        bmi_value = round(bmi, 1) if bmi is not None else 'Not Available'
        protein_recommendation = 'Use BMI details and a clearer full-body image for a more reliable assessment.'
        calorie_recommendation = 'No exact calorie target is generated because the image result is uncertain.'
        precautions = [
            'The model confidence was too close to the decision threshold.',
            'Upload a sharp full-body image with plain background and good lighting.',
        ]
        foods_to_eat = [
            'Continue balanced meals while you retake the screening image.',
        ]
        doctor_advice = [
            'Consult a doctor or dietitian if there is visible weight loss, weakness, or appetite loss.',
        ]
        recommendations = [*precautions, *foods_to_eat, *doctor_advice]
        return {
            'result': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'nutritional_status': nutritional_status,
            'bmi_category': bmi_category,
            'bmi_value': bmi_value,
            'assessment_basis': assessment_basis or 'Image screening model',
            'protein_recommendation': protein_recommendation,
            'calorie_recommendation': calorie_recommendation,
            'precautions': precautions,
            'foods_to_eat': foods_to_eat,
            'doctor_advice': doctor_advice,
            'recommendations': recommendations,
        }

    is_malnourished = score > 0.5
    result = 'Malnourished' if is_malnourished else 'Not Malnourished'
    risk_level = 'High' if score > 0.75 else 'Moderate' if score > 0.5 else 'Low'
    risk_emoji = '🔴' if risk_level == 'High' else '🟡' if risk_level == 'Moderate' else '🟢'
    nutritional_status = 'Needs Attention' if is_malnourished else 'Balanced'
    bmi_category = get_bmi_category(bmi)
    bmi_value = round(bmi, 1) if bmi is not None else (17.5 if is_malnourished else 22.4)
    assessment_basis = assessment_basis or ('BMI and body measurements' if bmi is not None else 'Uploaded photo screening model')

    if bmi is not None and bmi < 16:
        protein_recommendation = 'Increase protein-rich foods to 75-90 g/day with milk, eggs, dal, paneer, fish, or chicken.'
        calorie_recommendation = 'Increase calorie intake to 2400-2700 kcal/day with 3 meals plus 2-3 energy-dense snacks.'
        precautions = [
            'Your BMI suggests severe undernutrition risk.',
            'Do not skip meals and avoid long gaps between meals.',
            'Track weakness, frequent illness, low appetite, or fast weight loss.',
        ]
        foods_to_eat = [
            'Bananas, peanut butter, curd rice, khichdi, nuts, seeds, milk, and healthy oils.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Please consult a doctor or dietitian as soon as possible.',
            'Seek medical help quickly if there is faintness, severe weakness, repeated infections, or rapid weight loss.',
        ]
    elif bmi is not None and bmi < 18.5:
        protein_recommendation = 'Increase protein-rich foods to 65-75 g/day through dal, sprouts, curd, eggs, paneer, soy, fish, or chicken.'
        calorie_recommendation = 'Aim for 2200-2500 kcal/day using balanced meals and one or two nutritious snacks.'
        precautions = [
            'Your BMI is below the normal range, so improving nutrition should be a priority.',
            'Do not miss breakfast or delay meals.',
            'Monitor energy level and body weight each week.',
        ]
        foods_to_eat = [
            'Eat regular meals with rice or roti, dal, vegetables, fruit, milk, nuts, and a protein source each day.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Consult a doctor or registered dietitian if weight stays low.',
            'Get medical advice if you feel weak often, lose appetite, or continue losing weight.',
        ]
    elif bmi is not None and bmi < 25:
        protein_recommendation = 'Maintain protein intake around 50-60 g/day with balanced meals.'
        calorie_recommendation = 'Maintain calorie intake around 1800-2200 kcal/day depending on age and activity.'
        precautions = [
            'Your BMI is in the normal range.',
            'Maintain regular meal timing, sleep, hydration, and physical activity.',
        ]
        foods_to_eat = [
            'Continue a balanced diet with cereals, pulses, vegetables, fruits, milk, and enough water.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Consult a doctor if you still have fatigue, appetite loss, or recent weight loss even with normal BMI.',
        ]
    elif bmi is not None:
        protein_recommendation = 'Keep protein around 55-70 g/day and choose lighter, balanced meals.'
        calorie_recommendation = 'Avoid excess calories and focus on portion control, fiber, and steady physical activity.'
        precautions = [
            'Your BMI is above the normal range. This does not suggest malnutrition, but diet improvement is needed.',
            'Avoid sugary drinks, excess fried foods, and frequent late-night eating.',
        ]
        foods_to_eat = [
            'Prefer fruits, vegetables, whole grains, pulses, and home-cooked meals over sugary or fried foods.',
            protein_recommendation,
            calorie_recommendation,
        ]
        doctor_advice = [
            'Consult a doctor if you have diabetes, thyroid issues, swelling, or sudden weight change.',
        ]
    else:
        (
            protein_recommendation,
            calorie_recommendation,
            precautions,
            foods_to_eat,
            doctor_advice,
        ) = guidance_from_result(score, bmi)
    recommendations = [*precautions, *foods_to_eat, *doctor_advice]
    return {
        'result': result,
        'confidence': confidence,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'nutritional_status': nutritional_status,
        'bmi_category': bmi_category,
        'bmi_value': bmi_value,
        'assessment_basis': assessment_basis,
        'protein_recommendation': protein_recommendation,
        'calorie_recommendation': calorie_recommendation,
        'precautions': precautions,
        'foods_to_eat': foods_to_eat,
        'doctor_advice': doctor_advice,
        'recommendations': recommendations,
    }


def format_display_datetime(value):
    if value is None:
        return ''
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(current_app.config['APP_TIMEZONE']).strftime('%d %b %Y, %I:%M %p')


def serialize_analysis(analysis):
    analysis_status = None
    if analysis.ai_status == 'Image does not appear to contain a person':
        analysis_status = 'non_human'
    elif analysis.ai_status == 'Uncertain nutrition result':
        analysis_status = 'uncertain'

    score = (analysis.confidence or 0) / 100 if analysis.ai_status == 'Malnourished' else 1 - ((analysis.confidence or 0) / 100)
    details = build_prediction_details(score, analysis.bmi, analysis_status=analysis_status)
    details.update({
        'analysis_id': analysis.id,
        'image': analysis.image_path,
        'created_at': format_display_datetime(analysis.timestamp),
        'age': analysis.age,
        'height': analysis.height,
        'weight': analysis.weight,
    })
    return details


def is_mail_configured():
    return bool(current_app.config.get('MAIL_USERNAME') and current_app.config.get('MAIL_PASSWORD'))


def get_reset_serializer():
    return URLSafeTimedSerializer(current_app.config['SECRET_KEY'])


def generate_reset_token(email):
    return get_reset_serializer().dumps(email, salt='password-reset')


def verify_reset_token(token, max_age=3600):
    try:
        email = get_reset_serializer().loads(token, salt='password-reset', max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
    return email


def build_report_email_text(user, analysis):
    details = serialize_analysis(analysis)
    return "\n".join([
        "NutriDetect Analysis Report",
        "",
        f"Hello {user.full_name},",
        "",
        "Your nutrition analysis is complete.",
        f"Report ID: {analysis.id}",
        f"Generated: {details['created_at']}",
        f"AI Result: {details['result']}",
        f"Nutritional Status: {details['nutritional_status']}",
        f"Risk Level: {details['risk_level']}",
        f"Confidence: {details['confidence']}%",
        f"BMI: {details['bmi_value']}",
        f"BMI Category: {details['bmi_category']}",
        f"Protein Guidance: {details['protein_recommendation']}",
        f"Calorie Guidance: {details['calorie_recommendation']}",
        "",
        "Recommendations:",
        *[f"- {item}" for item in details['recommendations']],
        "",
        "Thanks for using NutriDetect.",
    ])


def send_analysis_report_email(mail, user, analysis):
    if not is_mail_configured():
        return False, 'Email is not configured yet.'

    if mail is None:
        return False, 'Mail service is not initialized.'

    message = Message(
        subject=f"NutriDetect Report #{analysis.id}",
        recipients=[user.email],
        body=build_report_email_text(user, analysis),
    )
    try:
        mail.send(message)
        return True, None
    except Exception as exc:
        current_app.logger.exception('Failed to send analysis report email')
        return False, str(exc)


def send_analysis_report_email_async(app, mail, user_id, analysis_id):
    with app.app_context():
        user = User.query.get(user_id)
        analysis = Analysis.query.get(analysis_id)
        if user is None or analysis is None:
            app.logger.warning(
                'Skipping async analysis email because user or analysis was not found. user_id=%s analysis_id=%s',
                user_id,
                analysis_id,
            )
            return
        sent, error_message = send_analysis_report_email(mail, user, analysis)
        if sent:
            app.logger.info('Analysis report emailed successfully to %s', user.email)
        else:
            app.logger.warning('Analysis report email failed for %s: %s', user.email, error_message)


def send_password_reset_email(mail, user, base_url=None):
    if not is_mail_configured():
        return False, 'Email is not configured yet.'

    if mail is None:
        return False, 'Mail service is not initialized.'

    token = generate_reset_token(user.email)
    if base_url:
        reset_url = f"{base_url.rstrip('/')}{url_for('main.reset_password', token=token)}"
    else:
        reset_url = url_for('main.reset_password', token=token, _external=True)
    message = Message(
        subject='NutriDetect Password Reset',
        recipients=[user.email],
        body="\n".join([
            f"Hello {user.full_name},",
            "",
            "We received a request to reset your NutriDetect password.",
            f"Open this link to choose a new password: {reset_url}",
            "",
            "This link will expire in 1 hour.",
            "If you did not request this, you can ignore this email.",
        ]),
    )
    try:
        mail.send(message)
        return True, None
    except Exception as exc:
        current_app.logger.exception('Failed to send password reset email')
        return False, str(exc)


def ensure_app_storage():
    os.makedirs(current_app.instance_path, exist_ok=True)
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    canonical_db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '', 1)
    if not os.path.exists(canonical_db_path):
        for legacy_db_path in current_app.config['LEGACY_DB_PATHS']:
            if os.path.exists(legacy_db_path):
                shutil.copy2(legacy_db_path, canonical_db_path)
                break
    db.create_all()


@bp.app_errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    limit_mb = int(current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024) / (1024 * 1024))
    flash(f'Image is too large. Please upload a file smaller than {limit_mb} MB.', 'danger')
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.home'))


@bp.app_errorhandler(404)
def handle_not_found(error):
    return render_template('404.html'), 404


@bp.route('/')
@bp.route('/home')
def home():
    return render_template('index.html')


@bp.route('/features')
def features():
    return render_template('features.html')


@bp.route('/about')
def about():
    return render_template('about.html')


@bp.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and verify_user_password(user, password):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Login unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')


@bp.route('/login/google')
def google_login():
    if not is_google_login_configured():
        flash('Google login is not configured yet. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.', 'danger')
        return redirect(url_for('main.login_page'))

    state = URLSafeTimedSerializer(current_app.config['SECRET_KEY']).dumps(
        {'source': 'google-login'},
        salt='google-oauth-state',
    )
    session['google_oauth_state'] = state

    query = urlencode({
        'client_id': current_app.config['GOOGLE_CLIENT_ID'],
        'redirect_uri': build_google_redirect_uri(),
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': state,
        'access_type': 'offline',
        'prompt': 'select_account',
    })
    return redirect(f'{GOOGLE_AUTH_URL}?{query}')


@bp.route('/login/google/callback')
def google_callback():
    if not is_google_login_configured():
        flash('Google login is not configured yet.', 'danger')
        return redirect(url_for('main.login_page'))

    error = request.args.get('error')
    if error:
        flash(f'Google sign-in failed: {error}', 'danger')
        return redirect(url_for('main.login_page'))

    state = request.args.get('state')
    expected_state = session.pop('google_oauth_state', None)
    if not state or state != expected_state:
        flash('Google sign-in failed: invalid state.', 'danger')
        return redirect(url_for('main.login_page'))

    code = request.args.get('code')
    if not code:
        flash('Google sign-in failed: missing authorization code.', 'danger')
        return redirect(url_for('main.login_page'))

    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                'code': code,
                'client_id': current_app.config['GOOGLE_CLIENT_ID'],
                'client_secret': current_app.config['GOOGLE_CLIENT_SECRET'],
                'redirect_uri': build_google_redirect_uri(),
                'grant_type': 'authorization_code',
            },
            timeout=15,
        )
        token_response.raise_for_status()
        access_token = token_response.json().get('access_token')
        if not access_token:
            raise ValueError('missing access token')

        profile_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=15,
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    except Exception as exc:
        current_app.logger.exception('Google OAuth callback failed')
        flash(f'Google sign-in failed: {exc}', 'danger')
        return redirect(url_for('main.login_page'))

    email = (profile.get('email') or '').strip().lower()
    if not email:
        flash('Google sign-in failed: email not provided by Google.', 'danger')
        return redirect(url_for('main.login_page'))

    user = User.query.filter_by(email=email).first()
    if user is None:
        full_name = (profile.get('name') or profile.get('given_name') or email.split('@')[0]).strip()
        user = User(
            full_name=full_name,
            email=email,
            password_hash=generate_password_hash(os.urandom(16).hex()),
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    flash(f'Signed in with Google as {user.email}.', 'success')
    return redirect(url_for('main.dashboard'))


@bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if not email:
            flash('Please enter your email address.', 'danger')
            return redirect(url_for('main.forgot_password'))

        user = User.query.filter_by(email=email).first()
        if user is None:
            flash('No account found with that email address.', 'danger')
            return redirect(url_for('main.forgot_password'))

        sent, error_message = send_password_reset_email(mail, user, request.host_url.rstrip('/'))
        if sent:
            flash(f'Password reset link sent to {user.email}. Please check your inbox.', 'success')
            return redirect(url_for('main.login_page'))

        flash(f'Unable to send reset email to {user.email}: {format_mail_error(error_message)}', 'danger')
        return redirect(url_for('main.forgot_password'))

    return render_template('forgot_password.html')


@bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if email is None:
        flash('This password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('main.forgot_password'))

    user = User.query.filter_by(email=email).first()
    if user is None:
        flash('No account found for this reset link.', 'danger')
        return redirect(url_for('main.forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not password:
            flash('Please enter a new password.', 'danger')
            return redirect(url_for('main.reset_password', token=token))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('main.reset_password', token=token))

        user.password_hash = generate_password_hash(password)
        db.session.commit()
        flash('Password updated successfully. Please log in.', 'success')
        return redirect(url_for('main.login_page'))

    return render_template('reset_password.html', token=token, email=email)


@bp.route('/register', methods=['GET', 'POST'])
@bp.route('/signup', methods=['GET', 'POST'])
@bp.route('/register.html', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not full_name or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('main.register_page'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('main.register_page'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('An account with this email already exists', 'danger')
            return redirect(url_for('main.register_page'))

        hashed_password = generate_password_hash(password)
        new_user = User(full_name=full_name, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully', 'success')
        return redirect(url_for('main.login_page'))
    return render_template('register.html')


@bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        dob_str = request.form.get('dob')
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not all([name, email, dob_str, subject, message]):
            flash('Please fill in all contact form fields', 'danger')
            return redirect(url_for('main.contact'))

        dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
        new_contact = Contact(name=name, email=email, dob=dob, subject=subject, message=message)
        db.session.add(new_contact)
        db.session.commit()
        flash('Message sent successfully', 'success')
        return redirect(url_for('main.contact'))
    return render_template('contact.html')


@bp.route('/dashboard')
@login_required
def dashboard():
    user = get_authenticated_user()
    history = (
        Analysis.query
        .filter_by(user_id=user.id)
        .order_by(Analysis.timestamp.desc())
        .all()
    )
    latest_analysis = history[0] if history else None
    return render_template(
        'dashboard.html',
        name=user.full_name,
        user=user,
        history=history,
        history_count=len(history),
        latest_analysis=latest_analysis,
        format_datetime=format_display_datetime,
    )


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.home'))


@bp.route('/predict', methods=['POST'])
@login_required
def predict():
    user = get_authenticated_user()
    if 'file' not in request.files:
        flash('Please choose an image to analyze', 'danger')
        return redirect(url_for('main.dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('Please choose an image to analyze', 'danger')
        return redirect(url_for('main.dashboard'))

    ensure_app_storage()

    filename = secure_filename(file.filename)
    if not filename:
        flash('Invalid file name', 'danger')
        return redirect(url_for('main.dashboard'))

    if not allowed_file(filename):
        flash('Please upload a valid image file (png, jpg, jpeg, gif, webp)', 'danger')
        return redirect(url_for('main.dashboard'))

    unique_filename = f'{uuid.uuid4().hex}{os.path.splitext(filename)[1].lower()}'
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    full_name = request.form.get('full_name', '').strip()
    age = request.form.get('age', type=int)
    height = request.form.get('height', type=float)
    weight = request.form.get('weight', type=float)
    bmi = None
    if height and weight and height > 0:
        bmi = weight / ((height / 100) ** 2)

    img_array = preprocess_image(filepath)
    thresholds = get_threshold_config()
    face_count = detect_face_count(filepath)
    loaded_human_model = get_human_model()
    if loaded_human_model is False:
        human_probability = 1.0 if face_count > 0 else 0.0
        is_human = face_count > 0
    else:
        human_probability = predict_human_probability(loaded_human_model, img_array)
        is_human = human_probability >= float(thresholds['human_threshold']) or (
            face_count > 0 and human_probability >= float(thresholds['face_fallback_threshold'])
        )

    analysis_status = None
    assessment_basis = 'Two-stage screening model with human verification'
    if not is_human:
        prediction = 0.0
        analysis_status = 'non_human'
        confidence = round((1.0 - human_probability) * 100, 1)
        details = build_prediction_details(
            prediction,
            bmi,
            analysis_status=analysis_status,
            assessment_basis=assessment_basis,
        )
        details['confidence'] = confidence
    else:
        loaded_model = get_model()
        if loaded_model is False:
            if bmi is None:
                prediction = 0.5
                analysis_status = 'uncertain'
                assessment_basis = 'Model unavailable and BMI not provided'
            else:
                prediction = derive_score_from_bmi(bmi)
                assessment_basis = 'BMI-only analysis fallback'
        else:
            raw_prediction = safe_probability(loaded_model.predict(img_array, verbose=0)[0][0])
            image_prediction = get_malnutrition_probability(raw_prediction)
            prediction, assessment_basis = combine_prediction_scores(image_prediction, bmi)
            delta = abs(prediction - float(thresholds['malnutrition_threshold']))
            if delta < float(thresholds['malnutrition_uncertain_margin']):
                analysis_status = 'uncertain'
        details = build_prediction_details(
            prediction,
            bmi,
            analysis_status=analysis_status,
            assessment_basis=assessment_basis,
        )

    model_load_warning = current_app.config.get('MODEL_LOAD_WARNING')
    if model_load_warning:
        flash(model_load_warning, 'warning')
        current_app.config['MODEL_LOAD_WARNING'] = None

    analysis = Analysis(
        user_id=user.id,
        age=age,
        height=height,
        weight=weight,
        bmi=round(bmi, 2) if bmi is not None else None,
        bmi_category=details['bmi_category'],
        ai_status=details['result'],
        confidence=details['confidence'],
        image_path=unique_filename,
    )
    try:
        if full_name and full_name != user.full_name:
            user.full_name = full_name
        db.session.add(analysis)
        db.session.commit()
    except Exception:
        db.session.rollback()
        flash('Unable to save analysis history right now. Please try again.', 'danger')
        return redirect(url_for('main.dashboard'))

    if is_mail_configured() and mail is not None:
        sent, error_message = send_analysis_report_email(mail, user, analysis)
        if sent:
            flash(f'Analysis completed successfully. Report emailed to {user.email}.', 'success')
        else:
            flash(
                f'Analysis completed successfully, but the report could not be emailed to {user.email}: '
                f'{format_mail_error(error_message)}',
                'warning',
            )
    else:
        flash('Analysis completed successfully.', 'success')
    return redirect(url_for('main.analysis_report', analysis_id=analysis.id))


@bp.route('/report/<int:analysis_id>')
@bp.route('/analysis_report/<int:analysis_id>')
@bp.route('/analysis-report/<int:analysis_id>')
@bp.route('/result/<int:analysis_id>')
@login_required
def analysis_report(analysis_id):
    user = get_authenticated_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if analysis is None:
        abort(404)
    return render_template('result.html', **serialize_analysis(analysis))


@bp.route('/report/<int:analysis_id>/delete', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    user = get_authenticated_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if analysis is None:
        abort(404)

    db.session.delete(analysis)
    db.session.commit()
    flash('History item removed successfully.', 'success')
    return redirect(url_for('main.dashboard') + '#history')


@bp.route('/report/<int:analysis_id>/download')
@login_required
def download_report(analysis_id):
    user = get_authenticated_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if analysis is None:
        abort(404)

    details = serialize_analysis(analysis)
    report_text = "\n".join([
        "NutriDetect Analysis Report",
        f"Report ID: {analysis.id}",
        f"Generated: {details['created_at']}",
        f"Name: {user.full_name}",
        f"AI Result: {details['result']}",
        f"Nutritional Status: {details['nutritional_status']}",
        f"Risk Level: {details['risk_level']}",
        f"Confidence: {details['confidence']}%",
        f"BMI: {details['bmi_value']}",
        f"BMI Category: {details['bmi_category']}",
        f"Protein Guidance: {details['protein_recommendation']}",
        f"Calorie Guidance: {details['calorie_recommendation']}",
        "",
        "Recommendations:",
        *[f"- {item}" for item in details['recommendations']],
    ])
    response = make_response(report_text)
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename=nutridetect_report_{analysis.id}.txt'
    return response


@bp.route('/report/<int:analysis_id>/email', methods=['POST'])
@login_required
def email_report(analysis_id):
    user = get_authenticated_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if analysis is None:
        abort(404)

    sent, error_message = send_analysis_report_email(mail, user, analysis)
    if sent:
        flash(f'Report emailed successfully to {user.email}.', 'success')
    else:
        flash(f'Unable to send report to {user.email}: {format_mail_error(error_message)}', 'danger')

    next_page = request.form.get('next')
    if next_page == 'dashboard':
        return redirect(url_for('main.dashboard') + '#history')
    return redirect(url_for('main.analysis_report', analysis_id=analysis.id))
