import os
import shutil
import json
import threading
import uuid
import zipfile
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
import requests
import cv2
from PIL import Image
from flask import Blueprint, abort, current_app, flash, make_response, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_mail import Message
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from models import Analysis, Contact, User, db, get_authenticated_user
except ImportError:
    # Support package-style imports (e.g. final_dti.nutriscan.app).
    from .models import Analysis, Contact, User, db, get_authenticated_user

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

bp = Blueprint('main', __name__)
mail = None
model = None
human_model = None
mal_class_indices = None
threshold_config = None
wfh_tables = None
tf = None
tf_import_attempted = False
warmup_started = False
warmup_lock = threading.Lock()
GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
GOOGLE_USERINFO_URL = 'https://www.googleapis.com/oauth2/v3/userinfo'

DEFAULT_THRESHOLDS = {
    'human_threshold': 0.5,
    'face_fallback_threshold': 0.15,
    'malnutrition_threshold': 0.5,
    'malnutrition_uncertain_margin': 0.12,
}
OTP_EXPIRY_SECONDS = 10 * 60
OTP_MAX_ATTEMPTS = 5


def get_tf():
    global tf, tf_import_attempted
    if not tf_import_attempted:
        tf_import_attempted = True
        try:
            # Use TensorFlow's default bundled Keras on deployment (Render/TensorFlow 2.15).
            # Forcing legacy Keras can break model loading when tf_keras is not available.
            import tensorflow as tensorflow_module
            tf = tensorflow_module
            tf_version = str(getattr(tensorflow_module, '__version__', 'unknown'))
            if not tf_version.startswith('2.15'):
                logging.getLogger(__name__).warning(
                    'Detected TensorFlow %s. This project expects TensorFlow 2.15.x for model compatibility. '
                    'Use the project virtual environment at final_dti/.venv.',
                    tf_version,
                )
            try:
                import keras as keras_module
                keras_version = str(getattr(keras_module, '__version__', 'unknown'))
                if not keras_version.startswith('2.15'):
                    logging.getLogger(__name__).warning(
                        'Detected Keras %s. This project expects Keras 2.15.x for model compatibility. '
                        'Use the project virtual environment at final_dti/.venv.',
                        keras_version,
                    )
            except Exception:
                logging.getLogger(__name__).warning(
                    'Keras import check failed. Model compatibility may be limited outside final_dti/.venv.'
                )
        except Exception:
            tf = False
            logging.getLogger(__name__).exception('TensorFlow import failed')
    return None if tf is False else tf


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
        tensorflow_module = get_tf()
        if tensorflow_module is None:
            warning = 'AI image model is unavailable right now, so NutriDetect is using BMI-based analysis.'
            existing_warning = current_app.config.get('MODEL_LOAD_WARNING')
            current_app.config['MODEL_LOAD_WARNING'] = (
                f'{existing_warning} {warning}'.strip() if existing_warning else warning
            )
            current_app.logger.warning('Malnutrition model load skipped because TensorFlow is unavailable.')
            model = False
            return model

        load_errors = []
        requested_model_path = current_app.config['MALNUTRITION_MODEL_PATH']
        for model_path in current_app.config.get(
            'MALNUTRITION_MODEL_CANDIDATE_PATHS',
            [current_app.config['MALNUTRITION_MODEL_PATH']],
        ):
            if not os.path.exists(model_path):
                continue
            try:
                model = tensorflow_module.keras.models.load_model(model_path, compile=False)
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
                load_errors.append(f'{model_path}: {type(exc).__name__}: {str(exc)[:240]}')
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
        tensorflow_module = get_tf()
        if tensorflow_module is None:
            warning = 'Human image verification is limited right now, so NutriDetect is using face-detection fallback.'
            existing_warning = current_app.config.get('MODEL_LOAD_WARNING')
            current_app.config['MODEL_LOAD_WARNING'] = (
                f'{existing_warning} {warning}'.strip() if existing_warning else warning
            )
            current_app.logger.warning('Human model load skipped because TensorFlow is unavailable.')
            human_model = False
            return human_model

        load_errors = []
        requested_model_path = current_app.config['HUMAN_MODEL_PATH']
        for model_path in current_app.config.get(
            'HUMAN_MODEL_CANDIDATE_PATHS',
            [current_app.config['HUMAN_MODEL_PATH']],
        ):
            if not os.path.exists(model_path):
                continue
            try:
                human_model = tensorflow_module.keras.models.load_model(model_path, compile=False)
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
                load_errors.append(f'{model_path}: {type(exc).__name__}: {str(exc)[:240]}')
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

    tensorflow_module = get_tf()
    if tensorflow_module is None:
        raise RuntimeError('TensorFlow is unavailable for serving_default inference.')
    output_map = serving_fn(tensorflow_module.constant(img_array))
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


def _xlsx_col_to_index(reference):
    letters = ''.join(ch for ch in reference if ch.isalpha()).upper()
    value = 0
    for ch in letters:
        value = (value * 26) + (ord(ch) - 64)
    return value - 1


def _build_excel_candidate_paths(filename):
    roots = [
        current_app.root_path,
        os.path.dirname(current_app.root_path),
        os.path.dirname(os.path.dirname(current_app.root_path)),
        os.getcwd(),
    ]
    seen = set()
    candidates = []
    for root in roots:
        if not root:
            continue
        normalized = os.path.abspath(root)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(os.path.join(normalized, filename))
    return candidates


def _parse_wfh_sheet(path):
    namespaces = {
        'm': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'p': 'http://schemas.openxmlformats.org/package/2006/relationships',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }
    with zipfile.ZipFile(path) as xlsx:
        shared = []
        if 'xl/sharedStrings.xml' in xlsx.namelist():
            shared_root = ET.fromstring(xlsx.read('xl/sharedStrings.xml'))
            for node in shared_root.findall('.//m:si', namespaces):
                chunks = [item.text or '' for item in node.findall('.//m:t', namespaces)]
                shared.append(''.join(chunks))

        workbook = ET.fromstring(xlsx.read('xl/workbook.xml'))
        relationships = ET.fromstring(xlsx.read('xl/_rels/workbook.xml.rels'))
        rel_map = {
            rel.attrib['Id']: rel.attrib['Target']
            for rel in relationships.findall('.//p:Relationship', namespaces)
        }

        first_sheet = workbook.find('.//m:sheets/m:sheet', namespaces)
        if first_sheet is None:
            return []
        relation_id = first_sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        worksheet_path = f"xl/{rel_map[relation_id].lstrip('/')}"

        worksheet = ET.fromstring(xlsx.read(worksheet_path))
        rows = []
        headers = None
        for row in worksheet.findall('.//m:sheetData/m:row', namespaces):
            values = {}
            for cell in row.findall('m:c', namespaces):
                ref = cell.attrib.get('r', 'A1')
                index = _xlsx_col_to_index(ref)
                value_node = cell.find('m:v', namespaces)
                raw_value = value_node.text if value_node is not None and value_node.text is not None else ''
                if cell.attrib.get('t') == 's' and raw_value.isdigit():
                    shared_index = int(raw_value)
                    value = shared[shared_index] if shared_index < len(shared) else raw_value
                else:
                    value = raw_value
                values[index] = value

            if not values:
                continue

            max_index = max(values)
            materialized = [values.get(i, '') for i in range(max_index + 1)]
            if headers is None:
                headers = [str(item).strip() for item in materialized]
                continue

            record = {}
            for idx, header in enumerate(headers):
                if not header:
                    continue
                entry = materialized[idx] if idx < len(materialized) else ''
                if header == 'Height':
                    record[header] = float(entry)
                elif header.startswith('SD'):
                    record[header] = float(entry)
            if {'Height', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3'}.issubset(record):
                rows.append(record)

        return rows


def get_wfh_tables():
    global wfh_tables
    if wfh_tables is not None:
        return wfh_tables

    table_files = {
        'male': 'wfh_boys_2-to-5-years_zscores.xlsx',
        'female': 'wfh_girls_2-to-5-years_zscores.xlsx',
    }
    loaded_tables = {}
    for gender, filename in table_files.items():
        table_rows = []
        for candidate in _build_excel_candidate_paths(filename):
            if not os.path.exists(candidate):
                continue
            try:
                table_rows = _parse_wfh_sheet(candidate)
                if table_rows:
                    break
            except Exception:
                current_app.logger.exception('Unable to parse WFH table from %s', candidate)
        loaded_tables[gender] = table_rows
    wfh_tables = loaded_tables
    return wfh_tables


def derive_score_from_wfh_table(gender, height, weight, age):
    if gender not in {'male', 'female'} or height is None or weight is None:
        return None
    if age is not None and (age < 2 or age > 5):
        return {
            'score': None,
            'status': 'WHO WFH table supports only ages 2 to 5 years',
            'severity': None,
            'reference': None,
        }

    rows = get_wfh_tables().get(gender, [])
    if not rows:
        return {
            'score': None,
            'status': 'WHO WFH table file not found',
            'severity': None,
            'reference': None,
        }

    nearest = min(rows, key=lambda row: abs(float(row['Height']) - float(height)))
    reference = (
        f"Nearest height {nearest['Height']} cm; "
        f"SD3neg {nearest['SD3neg']} kg, SD2neg {nearest['SD2neg']} kg, SD1neg {nearest['SD1neg']} kg"
    )
    if weight < nearest['SD3neg']:
        return {
            'score': 0.97,
            'status': 'Severe acute malnutrition (< -3 SD)',
            'severity': 'severe',
            'reference': reference,
        }
    if weight < nearest['SD2neg']:
        return {
            'score': 0.86,
            'status': 'Moderate malnutrition (-3 SD to -2 SD)',
            'severity': 'moderate',
            'reference': reference,
        }
    if weight < nearest['SD1neg']:
        return {
            'score': 0.62,
            'status': 'Mild malnutrition risk (-2 SD to -1 SD)',
            'severity': 'mild',
            'reference': reference,
        }
    return {
        'score': 0.12,
        'status': 'Within normal WFH range (>= -1 SD)',
        'severity': 'normal',
        'reference': reference,
    }


def combine_prediction_scores(image_score, bmi, table_score=None):
    bmi_score = derive_score_from_bmi(bmi)
    weighted_components = []

    if image_score is not None:
        weighted_components.append(('image', image_score, 0.55))
    if bmi_score is not None:
        weighted_components.append(('bmi', bmi_score, 0.25))
    if table_score is not None:
        weighted_components.append(('wfh', table_score, 0.20))

    if not weighted_components:
        return 0.5, 'Insufficient data for nutritional analysis'

    total_weight = sum(component[2] for component in weighted_components)
    combined_score = sum(component[1] * component[2] for component in weighted_components) / total_weight

    parts = [component[0] for component in weighted_components]
    if parts == ['image']:
        basis = 'Uploaded photo screening model'
    elif parts == ['bmi']:
        basis = 'BMI-based analysis'
    elif parts == ['wfh']:
        basis = 'WHO weight-for-height (WFH) table analysis'
    else:
        basis = 'Combined ' + ', '.join(parts).replace('wfh', 'WHO WFH table') + ' analysis'
    return combined_score, basis


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


def build_recommendations_summary(protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice):
    summary = []
    blocked = set(precautions or [])
    for item in [
        protein_recommendation,
        calorie_recommendation,
        foods_to_eat[0] if foods_to_eat else None,
        doctor_advice[0] if doctor_advice else None,
    ]:
        if item and item not in summary and item not in blocked:
            summary.append(item)
    return summary


def _dedupe_keep_order(items):
    unique = []
    for item in items:
        if item and item not in unique:
            unique.append(item)
    return unique


def personalize_guidance_by_risk(
    is_malnourished,
    risk_level,
    protein_recommendation,
    calorie_recommendation,
    precautions,
    foods_to_eat,
    doctor_advice,
):
    base_precaution = precautions[0] if precautions else None
    base_food = foods_to_eat[0] if foods_to_eat else None
    base_doctor = doctor_advice[0] if doctor_advice else None

    if is_malnourished and risk_level == 'High':
        protein_recommendation = 'Increase protein to around 80-95 g/day using eggs, milk, paneer, dal, soy, fish, or chicken in frequent meals.'
        calorie_recommendation = 'Increase calories to around 2400-2800 kcal/day with 3 meals and 2-3 energy-dense snacks.'
        precautions = _dedupe_keep_order([
            base_precaution or 'High malnutrition risk detected in current analysis.',
            'Do not skip meals and avoid fasting or long meal gaps.',
            'Monitor weakness, dizziness, swelling, repeated illness, and rapid weight loss daily.',
        ])
        foods_to_eat = _dedupe_keep_order([
            base_food or 'Use calorie-dense choices like banana shake, peanut chikki, curd rice, khichdi, nuts, and healthy oils.',
            'Add 2-3 high-protein snacks daily such as sprouts, boiled eggs, roasted chana, curd, or paneer.',
        ])
        doctor_advice = _dedupe_keep_order([
            base_doctor or 'Consult a doctor or dietitian urgently (within 24-48 hours).',
            'Go to urgent care immediately for fainting, severe weakness, dehydration, or inability to eat.',
        ])
        return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice

    if is_malnourished and risk_level == 'Moderate':
        protein_recommendation = 'Increase protein to around 70-85 g/day with one protein source in every meal and at least one protein snack.'
        calorie_recommendation = 'Aim for 2200-2500 kcal/day using balanced meals with extra nutritious snacks.'
        precautions = _dedupe_keep_order([
            base_precaution or 'Moderate malnutrition risk is present in current analysis.',
            'Avoid missed meals and maintain consistent meal timing every day.',
            'Track appetite, energy, and body weight weekly to ensure improvement.',
        ])
        foods_to_eat = _dedupe_keep_order([
            base_food or 'Use meals with rice/roti, dal, vegetables, curd, nuts, fruit, and one additional protein source.',
            'Prefer home-cooked nutrient-dense foods over packaged snacks and sugary drinks.',
        ])
        doctor_advice = _dedupe_keep_order([
            base_doctor or 'Consult a doctor or registered dietitian within a few days for a structured plan.',
            'Seek earlier consultation if weakness, appetite loss, or weight loss continues.',
        ])
        return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice

    # Default non-malnourished / low-risk prevention guidance.
    protein_recommendation = 'Maintain protein around 50-65 g/day with balanced meals and regular hydration.'
    calorie_recommendation = 'Maintain calories around 1800-2200 kcal/day, adjusted to activity level and age.'
    precautions = _dedupe_keep_order([
        base_precaution or 'Current analysis suggests low immediate malnutrition risk.',
        'Maintain routine sleep, hydration, and physical activity.',
        'Recheck if there is recent weight loss, poor appetite, or ongoing fatigue.',
    ])
    foods_to_eat = _dedupe_keep_order([
        base_food or 'Continue balanced meals with cereals, pulses, vegetables, fruits, milk, and healthy fats.',
        'Include at least one protein source in each main meal and fruit daily.',
    ])
    doctor_advice = _dedupe_keep_order([
        base_doctor or 'Consult a doctor if symptoms persist despite normal BMI and low risk score.',
        'Plan routine health review if there is thyroid, diabetes, or chronic illness history.',
    ])
    return protein_recommendation, calorie_recommendation, precautions, foods_to_eat, doctor_advice


def build_prediction_details(
    score,
    bmi=None,
    analysis_status=None,
    assessment_basis=None,
    table_status=None,
    table_reference=None,
    age=None,
    table_severity=None,
):
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
        recommendations = build_recommendations_summary(
            protein_recommendation,
            calorie_recommendation,
            precautions,
            foods_to_eat,
            doctor_advice,
        )
        return {
            'result': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'nutritional_status': nutritional_status,
            'bmi_category': bmi_category,
            'bmi_value': bmi_value,
            'assessment_basis': assessment_basis or 'Human-detection screening',
            'table_status': table_status,
            'table_reference': table_reference,
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
        recommendations = build_recommendations_summary(
            protein_recommendation,
            calorie_recommendation,
            precautions,
            foods_to_eat,
            doctor_advice,
        )
        return {
            'result': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'nutritional_status': nutritional_status,
            'bmi_category': bmi_category,
            'bmi_value': bmi_value,
            'assessment_basis': assessment_basis or 'Image screening model',
            'table_status': table_status,
            'table_reference': table_reference,
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

    # Child guidance (2-5 years) prioritizes WHO WFH table interpretation.
    if age is not None and 2 <= age <= 5:
        if age <= 3:
            base_protein = '13 g/day'
            calcium_target = '700 mg/day'
        else:
            base_protein = '19 g/day'
            calcium_target = '1000 mg/day'
        iron_target = '7 mg/day' if age <= 3 else '10 mg/day'
        zinc_target = '3 mg/day' if age <= 3 else '5 mg/day'
        vitd_target = '600 IU/day'

        if table_severity == 'severe':
            protein_recommendation = f'Provide about 18-25 g/day protein (minimum baseline {base_protein}) in small, frequent feeds.'
            calorie_recommendation = 'Aim around 1200-1500 kcal/day with calorie-dense supervised meals; urgent clinical follow-up is recommended.'
        elif table_severity == 'moderate':
            protein_recommendation = f'Provide about 16-22 g/day protein (minimum baseline {base_protein}) with daily eggs/dal/milk/curd.'
            calorie_recommendation = 'Aim around 1100-1400 kcal/day with 3 meals + 2 nutritious snacks.'
        elif table_severity == 'mild':
            protein_recommendation = f'Provide around 14-20 g/day protein (minimum baseline {base_protein}) with balanced daily meals.'
            calorie_recommendation = 'Aim around 1000-1300 kcal/day with regular meal timing.'
        else:
            protein_recommendation = f'Maintain age-appropriate protein intake around {base_protein}.'
            calorie_recommendation = 'Maintain balanced age-appropriate calories with fruits, vegetables, grains, milk, and protein foods.'

        precautions = [
            'Result is interpreted using image + BMI + WHO WFH Excel table when available.',
            'Monitor weight and appetite weekly and recheck growth trend.',
            'Seek pediatric review immediately if child has edema, persistent fever, vomiting, or poor feeding.',
        ]
        foods_to_eat = [
            f'Include iron-rich foods and vitamin C daily (target iron {iron_target}).',
            f'Include calcium-rich foods (target calcium {calcium_target}) and vitamin D ({vitd_target}).',
            f'Include zinc-rich foods daily (target zinc {zinc_target}).',
        ]
        doctor_advice = [
            'Consult a pediatrician or pediatric dietitian for individualized growth plan.',
        ]
        recommendations = build_recommendations_summary(
            protein_recommendation,
            calorie_recommendation,
            precautions,
            foods_to_eat,
            doctor_advice,
        )
        return {
            'result': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'nutritional_status': nutritional_status,
            'bmi_category': bmi_category,
            'bmi_value': bmi_value,
            'assessment_basis': assessment_basis,
            'table_status': table_status,
            'table_reference': table_reference,
            'protein_recommendation': protein_recommendation,
            'calorie_recommendation': calorie_recommendation,
            'precautions': precautions,
            'foods_to_eat': foods_to_eat,
            'doctor_advice': doctor_advice,
            'recommendations': recommendations,
        }

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
        if bmi < 21:
            protein_recommendation = 'Maintain protein intake around 55-65 g/day with balanced meals and one extra protein snack.'
            calorie_recommendation = 'Maintain calorie intake around 2000-2300 kcal/day to avoid further weight drop.'
            bmi_note = 'Your BMI is normal but near the lower side of the healthy range.'
        elif bmi < 23:
            protein_recommendation = 'Maintain protein intake around 50-60 g/day with balanced meals.'
            calorie_recommendation = 'Maintain calorie intake around 1800-2200 kcal/day depending on age and activity.'
            bmi_note = 'Your BMI is in the healthy mid-normal range.'
        else:
            protein_recommendation = 'Keep protein around 50-60 g/day and focus on lean protein sources.'
            calorie_recommendation = 'Maintain calories near 1700-2100 kcal/day with regular activity and portion balance.'
            bmi_note = 'Your BMI is normal but close to the upper side of the healthy range.'
        precautions = [
            bmi_note,
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

    (
        protein_recommendation,
        calorie_recommendation,
        precautions,
        foods_to_eat,
        doctor_advice,
    ) = personalize_guidance_by_risk(
        is_malnourished,
        risk_level,
        protein_recommendation,
        calorie_recommendation,
        precautions,
        foods_to_eat,
        doctor_advice,
    )

    recommendations = build_recommendations_summary(
        protein_recommendation,
        calorie_recommendation,
        precautions,
        foods_to_eat,
        doctor_advice,
    )
    return {
        'result': result,
        'confidence': confidence,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'nutritional_status': nutritional_status,
        'bmi_category': bmi_category,
        'bmi_value': bmi_value,
        'assessment_basis': assessment_basis,
        'table_status': table_status,
        'table_reference': table_reference,
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
        'can_view_nutriplan': can_view_nutriplan(analysis, details.get('result')),
    })
    return details


def can_view_nutriplan(analysis, result_label=None):
    if analysis.ai_status in {'Image does not appear to contain a person', 'Uncertain nutrition result'}:
        return False
    label = str(result_label if result_label is not None else analysis.ai_status or '').strip().lower()
    return label == 'malnourished'


def get_nutriplan_level(analysis, details):
    if analysis.ai_status in {'Image does not appear to contain a person', 'Uncertain nutrition result'}:
        return 'general'

    bmi = analysis.bmi
    risk_level = (details.get('risk_level') or '').lower()
    is_malnourished = details.get('result') == 'Malnourished'

    if is_malnourished and (risk_level == 'high' or (bmi is not None and bmi < 16)):
        return 'severe'
    if is_malnourished and (risk_level in {'moderate', 'high'} or (bmi is not None and bmi < 18.5)):
        return 'moderate'
    if is_malnourished:
        return 'mild'
    return 'recovery'


def get_nutriplan_metadata(level):
    plan_titles = {
        'severe': 'Severe Support Plan',
        'moderate': 'Moderate Recovery Plan',
        'mild': 'Mild Recovery Plan',
        'recovery': 'Balanced Recovery Plan',
        'general': 'General Nutrition Plan',
    }
    plan_notes = {
        'severe': 'Higher calories and protein with frequent soft meals every 3-4 hours.',
        'moderate': 'High protein and energy-dense meals with careful hydration and meal timing.',
        'mild': 'Balanced high-quality meals with steady protein support and regular snacks.',
        'recovery': 'Maintenance-focused plan with balanced portions and micronutrient-rich foods.',
        'general': 'Use this as a practical baseline and personalize with a doctor or dietitian.',
    }
    return plan_titles.get(level, 'General Nutrition Plan'), plan_notes.get(
        level,
        'Use this as a practical baseline and personalize with a doctor or dietitian.',
    )


def apply_level_adjustments(day_plan, level):
    adjustments = {
        'severe': {
            'morning': ' Add: one extra protein item (egg or paneer) and a small banana shake.',
            'evening': ' Add: roasted chana or peanuts plus fruit.',
            'night': ' Add: soft dal soup and one cup curd before bed.',
        },
        'moderate': {
            'morning': ' Add: milk and nuts daily.',
            'evening': ' Add: fruit with protein snack.',
            'night': ' Add: curd or buttermilk before bed.',
        },
        'mild': {
            'morning': ' Keep portions steady and include one protein source.',
            'evening': ' Prefer fruit + milk over packaged snacks.',
            'night': ' Keep dinner soft, warm, and easy to digest.',
        },
        'recovery': {
            'morning': ' Keep balanced breakfast with milk or curd.',
            'evening': ' Use light fruit-based snacks and hydrate well.',
            'night': ' Maintain moderate portions and avoid very oily foods.',
        },
        'general': {
            'morning': ' Keep meals soft and easy to digest.',
            'evening': ' Hydrate well and avoid skipping snacks.',
            'night': ' Eat early and include warm milk before sleep if tolerated.',
        },
    }
    note = adjustments.get(level, adjustments['general'])
    return {
        'day': day_plan['day'],
        'morning': day_plan['morning'] + note['morning'],
        'evening': day_plan['evening'] + note['evening'],
        'night': day_plan['night'] + note['night'],
    }


def parse_nutriplan_preferences(args):
    diet_type = str(args.get('diet_type', 'mixed')).strip().lower()
    budget = str(args.get('budget', 'medium')).strip().lower()
    cuisine = str(args.get('cuisine', 'south_indian')).strip().lower()

    if diet_type not in {'veg', 'non_veg', 'mixed'}:
        diet_type = 'mixed'
    if budget not in {'low', 'medium', 'high'}:
        budget = 'medium'
    if cuisine not in {'south_indian', 'north_indian', 'odisha'}:
        cuisine = 'south_indian'

    return {
        'diet_type': diet_type,
        'budget': budget,
        'cuisine': cuisine,
    }


def build_nutriplan_profile(analysis, details, level, preferences):
    age = analysis.age if analysis.age is not None else details.get('age')
    bmi = analysis.bmi
    risk_level = str(details.get('risk_level') or '').lower()
    confidence = analysis.confidence or 0
    diet_type = preferences.get('diet_type', 'mixed')
    budget = preferences.get('budget', 'medium')
    cuisine = preferences.get('cuisine', 'south_indian')
    seed = (
        (analysis.id or 0) * 131
        + int(confidence * 10)
        + (int(age) if age is not None else 0) * 17
        + int((bmi if bmi is not None else 20) * 10)
        + sum(ord(ch) for ch in risk_level)
        + len(level) * 23
        + sum(ord(ch) for ch in diet_type)
        + sum(ord(ch) for ch in budget)
        + sum(ord(ch) for ch in cuisine)
    )
    rng = random.Random(seed)

    if age is not None and age <= 5:
        age_group = 'child'
    elif age is not None and age <= 17:
        age_group = 'teen'
    else:
        age_group = 'adult'

    protein_focus_bonus = 0
    if level in {'severe', 'moderate'}:
        protein_focus_bonus += 1
    if risk_level == 'high':
        protein_focus_bonus += 1
    if bmi is not None and bmi < 16:
        protein_focus_bonus += 1

    if budget == 'high':
        value_bonus = rng.randint(8, 14)
    elif budget == 'low':
        value_bonus = rng.randint(1, 6)
    else:
        value_bonus = rng.randint(4, 10)

    if diet_type == 'veg':
        protein_track = 'veg-forward'
    elif diet_type == 'non_veg':
        protein_track = 'non-veg-focused'
    else:
        protein_track = 'mixed-protein'

    return {
        'seed': seed,
        'age_group': age_group,
        'rotation_offset': rng.randint(0, 3),
        'protein_track': protein_track,
        'value_bonus': value_bonus + protein_focus_bonus,
        'budget': budget,
        'cuisine': cuisine,
        'diet_type': diet_type,
    }


def normalize_meal_by_profile(text, profile):
    normalized = text
    if profile['protein_track'] == 'veg-forward':
        normalized = normalized.replace('egg curry or paneer option', 'paneer option')
        normalized = normalized.replace('egg curry or paneer', 'paneer')
        normalized = normalized.replace('Boiled eggs with soft roti and milk', 'Paneer cubes with soft roti and milk')
        normalized = normalized.replace('Egg curry with soft rice', 'Paneer curry with soft rice')
    elif profile['protein_track'] == 'non-veg-focused':
        normalized = normalized.replace('paneer option', 'egg option')
        normalized = normalized.replace('paneer', 'egg/chicken')
        normalized = normalized.replace('soy or paneer bhurji', 'egg/chicken bhurji')

    cuisine = profile.get('cuisine', 'south_indian')
    if cuisine == 'north_indian':
        normalized = normalized.replace('idli', 'soft roti')
        normalized = normalized.replace('dosa', 'vegetable cheela')
        normalized = normalized.replace('upma', 'dalia')
        normalized = normalized.replace('poha', 'dalia')
        normalized = normalized.replace('khichdi', 'dal khichdi')
        normalized = normalized.replace('soft rice', 'jeera rice')
        normalized = normalized.replace('buttermilk', 'lassi')
    elif cuisine == 'odisha':
        normalized = normalized.replace('idli', 'pakhala rice (light)')
        normalized = normalized.replace('dosa', 'chakuli pitha')
        normalized = normalized.replace('khichdi', 'dalma rice')
        normalized = normalized.replace('chapati', 'rice roti')
        normalized = normalized.replace('paneer', 'chhena')

    budget = profile.get('budget', 'medium')
    if budget == 'low':
        normalized = normalized.replace('paneer', 'soya chunks/chana')
        normalized = normalized.replace('dry fruits', 'groundnuts')
        normalized = normalized.replace('nuts', 'groundnuts')
        normalized = normalized.replace('fruit juice', 'seasonal fruit')
        normalized = normalized.replace('banana shake', 'banana + milk')
    elif budget == 'high':
        normalized = normalized.replace('milk', 'fortified milk')
        normalized = normalized.replace('curd', 'probiotic curd')
        normalized = normalized.replace('fruit', 'premium seasonal fruit')
    return normalized


def build_weekly_shopping_list(profile):
    base_items = {
        'Week 1': ['Rice', 'Moong dal', 'Milk', 'Curd', 'Banana', 'Peanuts'],
        'Week 2': ['Oats', 'Poha', 'Seasonal fruits', 'Roasted chana', 'Paneer', 'Buttermilk'],
        'Week 3': ['Sprouts', 'Dates', 'Nuts', 'Dal', 'Whole wheat flour', 'Vegetables'],
        'Week 4': ['Ragi or oats', 'Curd', 'Light vegetables', 'Rice', 'Lentils', 'Fruits'],
    }

    if profile['diet_type'] == 'veg':
        extras = ['Paneer', 'Soya chunks', 'Groundnuts', 'Chickpeas']
    elif profile['diet_type'] == 'non_veg':
        extras = ['Eggs', 'Chicken (lean)', 'Fish (optional)', 'Curd']
    else:
        extras = ['Eggs', 'Paneer', 'Soya chunks', 'Curd']

    if profile['budget'] == 'low':
        budget_tip = 'Budget mode: Prefer local seasonal produce, dal, chana, peanuts, and eggs/soya for protein.'
    elif profile['budget'] == 'high':
        budget_tip = 'Premium mode: Include dry fruits, quality protein options, and diverse fruits through the week.'
    else:
        budget_tip = 'Balanced budget: Mix affordable staples with one premium protein/fruit item daily.'

    cuisine_addons = {
        'south_indian': ['Idli/Dosa batter', 'Ragi flour', 'Coconut (optional)', 'Sambar ingredients'],
        'north_indian': ['Atta', 'Besan', 'Jeera/Ajwain', 'Lassi or curd'],
        'odisha': ['Rice', 'Dalma ingredients', 'Chhena/paneer', 'Seasonal greens'],
    }

    weekly = {}
    for week, items in base_items.items():
        merged = items + extras + cuisine_addons.get(profile['cuisine'], [])
        deduped = []
        seen = set()
        for item in merged:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        weekly[week] = deduped

    return weekly, budget_tip


def personalize_day_plan(day_plan, profile, level):
    day = day_plan['day']
    rng = random.Random(profile['seed'] + day * 53)

    child_morning_boosters = [
        ' Add: mashed banana with milk.',
        ' Add: soft fruit puree and curd.',
        ' Add: extra spoon of ghee in breakfast.',
    ]
    adult_morning_boosters = [
        ' Add: one protein side (paneer/egg) and warm milk.',
        ' Add: soaked nuts and banana shake.',
        ' Add: curd with roasted seeds.',
    ]
    evening_boosters = [
        ' Add: seasonal fruit and 1 protein snack.',
        ' Add: buttermilk and roasted chana.',
        ' Add: peanut chikki or sprout salad.',
    ]
    night_boosters = [
        ' Add: warm turmeric milk before sleep.',
        ' Add: thin dal soup with soft dinner.',
        ' Add: curd or buttermilk if tolerated.',
    ]
    cuisine_boosters = {
        'south_indian': ' Include one South Indian home-style option (idli/dosa/upma/ragi).',
        'north_indian': ' Include one North Indian home-style option (roti/dal/khichdi).',
        'odisha': ' Include one Odisha home-style option (dalma/rice/greens).',
    }
    budget_boosters = {
        'low': ' Budget tip: prioritize seasonal local foods and simple protein sources.',
        'medium': ' Keep balanced portions with affordable protein + fruit daily.',
        'high': ' Add a premium protein or dry-fruit serving for recovery support.',
    }

    morning_boosters = child_morning_boosters if profile['age_group'] == 'child' else adult_morning_boosters

    morning = normalize_meal_by_profile(day_plan['morning'], profile) + rng.choice(morning_boosters)
    evening = normalize_meal_by_profile(day_plan['evening'], profile) + rng.choice(evening_boosters)
    night = normalize_meal_by_profile(day_plan['night'], profile) + rng.choice(night_boosters)

    if profile.get('diet_type') == 'veg':
        morning = morning.replace('egg', 'paneer/soya')
        evening = evening.replace('egg', 'sprouts')
        night = night.replace('egg', 'paneer/soya')
    elif profile.get('diet_type') == 'non_veg':
        morning += ' Include egg/chicken/fish option for protein.'
        evening += ' Add boiled egg or chicken soup if possible.'
        night += ' Prefer one non-veg protein serving (egg/chicken/fish).'

    style_note = cuisine_boosters.get(profile['cuisine'], '')
    budget_note = budget_boosters.get(profile['budget'], '')

    personalized = {
        'day': day,
        'morning': morning + style_note,
        'evening': evening + budget_note,
        'night': night,
    }
    return apply_level_adjustments(personalized, level)


def get_daily_nutrition_value(level, day, profile_bonus=0):
    base_by_level = {
        'severe': 125,
        'moderate': 115,
        'mild': 105,
        'recovery': 95,
        'general': 90,
    }
    week_boost = 0
    if 8 <= day <= 14:
        week_boost = 4
    elif 15 <= day <= 21:
        week_boost = 8
    elif 22 <= day <= 30:
        week_boost = 5
    day_variation = (day % 3)
    return base_by_level.get(level, 90) + week_boost + profile_bonus + day_variation


def build_nutriplan_30_days(level, analysis, details, preferences):
    profile = build_nutriplan_profile(analysis, details, level, preferences)
    rotation_offset = profile['rotation_offset']

    first_week = [
        {
            'day': 1,
            'morning': 'Early morning milk with soaked almonds; breakfast idli with ghee',
            'evening': 'Banana or fruit mash',
            'night': 'Khichdi with curd; warm milk before bed',
        },
        {
            'day': 2,
            'morning': 'Warm milk; upma with peanuts',
            'evening': 'Apple juice with a small handful of roasted gram',
            'night': 'Chapati with paneer gravy; milk before bed',
        },
        {
            'day': 3,
            'morning': 'Milk; oats porridge with dry fruits',
            'evening': 'Banana and curd smoothie',
            'night': 'Vegetable khichdi with curd',
        },
        {
            'day': 4,
            'morning': 'Warm water; dosa with chutney',
            'evening': 'Coconut water and soft fruit',
            'night': 'Chapati with egg curry or paneer option',
        },
        {
            'day': 5,
            'morning': 'Milk; poha with peanuts',
            'evening': 'Apple slices or fresh juice',
            'night': 'Soft khichdi with curd',
        },
        {
            'day': 6,
            'morning': 'Milk; oats with dry fruits',
            'evening': 'Banana shake',
            'night': 'Chapati with paneer and light dal',
        },
        {
            'day': 7,
            'morning': 'Early milk; idli with ghee',
            'evening': 'Seasonal fruit and buttermilk',
            'night': 'Khichdi and curd',
        },
    ]

    week2_morning = [
        'Milk and idli with chutney',
        'Milk and oats porridge',
        'Warm water and poha with peanuts',
        'Milk and dosa with chutney',
    ]
    week2_evening = [
        'Banana and milkshake',
        'Fresh fruit juice and soaked nuts',
        'Fruit bowl with curd',
        'Banana with roasted chana',
    ]
    week2_night = [
        'Khichdi with curd',
        'Chapati with paneer gravy',
        'Chapati with egg curry or paneer',
        'Vegetable dal soup with soft rice',
    ]

    week3_morning = [
        'Milk and oats with nuts',
        'Boiled eggs with soft roti and milk',
        'Paneer sandwich and fruit milk',
        'Moong chilla with curd',
    ]
    week3_evening = [
        'Nuts and banana shake',
        'Sprouts chaat and buttermilk',
        'Peanut laddu and fruit',
        'Curd smoothie with dates',
    ]
    week3_night = [
        'Chapati with paneer curry',
        'Egg curry with soft rice',
        'Dal khichdi with curd',
        'Chapati with soy or paneer bhurji',
    ]

    week4_morning = [
        'Soft idli and warm milk',
        'Oats porridge with banana',
        'Rice porridge with milk',
        'Poha with curd',
    ]
    week4_evening = [
        'Fruit and milk',
        'Fresh juice with nuts',
        'Banana and curd',
        'Papaya and buttermilk',
    ]
    week4_night = [
        'Khichdi and curd',
        'Vegetable soup with chapati',
        'Soft dal rice and curd',
        'Light paneer curry with chapati',
    ]

    plan_days = []
    for day in range(1, 8):
        idx = (day - 1 + rotation_offset) % len(first_week)
        base_plan = dict(first_week[idx])
        base_plan['day'] = day
        plan_days.append(base_plan)

    for day in range(8, 15):
        idx = (day - 8 + rotation_offset) % 4
        plan_days.append({
            'day': day,
            'morning': week2_morning[idx],
            'evening': week2_evening[idx],
            'night': week2_night[idx],
        })

    for day in range(15, 22):
        idx = (day - 15 + rotation_offset) % 4
        plan_days.append({
            'day': day,
            'morning': week3_morning[idx],
            'evening': week3_evening[idx],
            'night': week3_night[idx],
        })

    for day in range(22, 31):
        idx = (day - 22 + rotation_offset) % 4
        plan_days.append({
            'day': day,
            'morning': week4_morning[idx],
            'evening': week4_evening[idx],
            'night': week4_night[idx],
        })

    adjusted = [personalize_day_plan(day_plan, profile, level) for day_plan in plan_days]
    enriched = []
    for day_plan in adjusted:
        day = day_plan['day']
        day_plan['nutrition_value'] = get_daily_nutrition_value(level, day, profile_bonus=profile['value_bonus'])
        enriched.append(day_plan)
    return enriched, profile


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


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_utc_iso(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_pending_otp_expired(pending_registration):
    created_at = _parse_utc_iso((pending_registration or {}).get('otp_created_at'))
    if created_at is None:
        return True
    elapsed = (_utcnow() - created_at).total_seconds()
    return elapsed > OTP_EXPIRY_SECONDS


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


def warmup_analysis_assets(app):
    with app.app_context():
        try:
            get_threshold_config()
            get_wfh_tables()
            get_human_model()
            get_model()
            app.logger.info('Analysis assets warmed up successfully.')
        except Exception:
            app.logger.exception('Warmup of analysis assets failed.')


def start_background_warmup():
    global warmup_started
    if warmup_started:
        return
    if not current_app.config.get('WARMUP_ANALYSIS_ASSETS', True):
        return

    with warmup_lock:
        if warmup_started:
            return
        app_obj = current_app._get_current_object()
        worker = threading.Thread(
            target=warmup_analysis_assets,
            args=(app_obj,),
            daemon=True,
            name='nutridetect-warmup',
        )
        worker.start()
        warmup_started = True


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
    start_background_warmup()


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
        password = request.form.get('password', '')
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
        if user is not None:
            sent, error_message = send_password_reset_email(mail, user, request.host_url.rstrip('/'))
            if not sent:
                current_app.logger.warning(
                    'Password reset email failed for %s: %s',
                    user.email,
                    format_mail_error(error_message),
                )

        # Avoid account enumeration by returning the same message either way.
        flash('If an account exists for that email, a password reset link has been sent.', 'success')
        return redirect(url_for('main.login_page'))

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

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
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
    pending_registration = session.get('pending_registration')

    if request.method == 'POST':
        action = request.form.get('action', '').strip().lower()
        entered_otp = request.form.get('otp', '').strip()

        if action == 'reset_registration':
            session.pop('pending_registration', None)
            flash('Registration reset. Please fill the form again.', 'success')
            return redirect(url_for('main.register_page'))

        if action == 'resend_otp':
            if not pending_registration:
                flash('Registration session expired. Please fill the form again.', 'danger')
                return redirect(url_for('main.register_page'))

            otp = f"{random.randint(100000, 999999)}"
            pending_registration['otp'] = otp
            pending_registration['otp_created_at'] = _utcnow().isoformat()
            pending_registration['otp_attempts'] = 0
            session['pending_registration'] = pending_registration

            email = pending_registration.get('email')
            full_name = pending_registration.get('full_name')

            if is_mail_configured() and mail is not None:
                message = Message(
                    subject='NutriDetect Account Verification OTP (Resent)',
                    recipients=[email],
                    body="\n".join([
                        f"Hello {full_name},",
                        "",
                        f"Your new NutriDetect OTP is: {otp}",
                        "Use this OTP to complete registration.",
                    ]),
                )
                try:
                    mail.send(message)
                    flash(f'New OTP sent to {email}.', 'success')
                except Exception as exc:
                    current_app.logger.exception('Failed to resend registration OTP')
                    flash(
                        f'Unable to resend OTP email right now: {format_mail_error(str(exc))}. '
                        f'Use this OTP for now: {otp}',
                        'danger',
                    )
            else:
                flash(f'Mail is not configured. Your new OTP is: {otp}', 'danger')

            return render_template(
                'register.html',
                otp_required=True,
                pending_email=email,
                pending_full_name=full_name,
            )

        if pending_registration and not entered_otp:
            full_name_input = request.form.get('full_name', '').strip()
            email_input = request.form.get('email', '').strip()
            password_input = request.form.get('password', '').strip()
            confirm_password_input = request.form.get('confirm_password', '').strip()

            # OTP step form does not send profile fields, so keep user on OTP step with clear instruction.
            if not any([full_name_input, email_input, password_input, confirm_password_input]):
                flash('Please enter OTP to continue, or click Start Over.', 'danger')
                return render_template(
                    'register.html',
                    otp_required=True,
                    pending_email=pending_registration.get('email'),
                    pending_full_name=pending_registration.get('full_name'),
                )

        if entered_otp:
            if not pending_registration:
                flash('Registration session expired. Please fill the form again.', 'danger')
                return redirect(url_for('main.register_page'))

            if _is_pending_otp_expired(pending_registration):
                session.pop('pending_registration', None)
                flash('OTP expired. Please register again to get a new OTP.', 'danger')
                return redirect(url_for('main.register_page'))

            if pending_registration.get('otp') != entered_otp:
                attempts = int(pending_registration.get('otp_attempts', 0)) + 1
                pending_registration['otp_attempts'] = attempts
                session['pending_registration'] = pending_registration
                if attempts >= OTP_MAX_ATTEMPTS:
                    session.pop('pending_registration', None)
                    flash('Too many invalid OTP attempts. Please register again.', 'danger')
                    return redirect(url_for('main.register_page'))
                flash('Invalid OTP. Please try again.', 'danger')
                return render_template(
                    'register.html',
                    otp_required=True,
                    pending_email=pending_registration.get('email'),
                    pending_full_name=pending_registration.get('full_name'),
                )

            email = pending_registration.get('email')
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                session.pop('pending_registration', None)
                flash('An account with this email already exists', 'danger')
                return redirect(url_for('main.register_page'))

            new_user = User(
                full_name=pending_registration.get('full_name'),
                email=email,
                password_hash=pending_registration.get('password_hash'),
            )
            db.session.add(new_user)
            db.session.commit()
            session.pop('pending_registration', None)
            flash('Account created successfully', 'success')
            return redirect(url_for('main.login_page'))

        full_name = request.form.get('full_name')
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not full_name or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('main.register_page'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('main.register_page'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('main.register_page'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('An account with this email already exists', 'danger')
            return redirect(url_for('main.register_page'))

        otp = f"{random.randint(100000, 999999)}"
        session['pending_registration'] = {
            'full_name': full_name,
            'email': email,
            'password_hash': generate_password_hash(password),
            'otp': otp,
            'otp_created_at': _utcnow().isoformat(),
            'otp_attempts': 0,
        }

        if is_mail_configured() and mail is not None:
            message = Message(
                subject='NutriDetect Account Verification OTP',
                recipients=[email],
                body="\n".join([
                    f"Hello {full_name},",
                    "",
                    f"Your NutriDetect OTP is: {otp}",
                    "It will be required to complete registration.",
                ]),
            )
            try:
                mail.send(message)
                flash(f'OTP sent to {email}. Enter it below to complete registration.', 'success')
            except Exception as exc:
                current_app.logger.exception('Failed to send registration OTP')
                flash(
                    f'Unable to send OTP email right now: {format_mail_error(str(exc))}. '
                    f'Use this OTP for now: {otp}',
                    'danger',
                )
        else:
            flash(f'Mail is not configured. Use this OTP to continue: {otp}', 'danger')

        return render_template(
            'register.html',
            otp_required=True,
            pending_email=email,
            pending_full_name=full_name,
        )

    if pending_registration:
        return render_template(
            'register.html',
            otp_required=True,
            pending_email=pending_registration.get('email'),
            pending_full_name=pending_registration.get('full_name'),
        )

    return render_template('register.html', otp_required=False)


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

        try:
            dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Please enter a valid date of birth.', 'danger')
            return redirect(url_for('main.contact'))
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
    gender = request.form.get('gender', '').strip().lower()
    age = request.form.get('age', type=int)
    height = request.form.get('height', type=float)
    weight = request.form.get('weight', type=float)
    if gender not in {'male', 'female'}:
        gender = None
    bmi = None
    if height and weight and height > 0:
        bmi = weight / ((height / 100) ** 2)
    table_lookup = derive_score_from_wfh_table(gender, height, weight, age)
    table_score = table_lookup['score'] if table_lookup else None
    table_status = table_lookup['status'] if table_lookup else None
    table_severity = table_lookup['severity'] if table_lookup else None
    table_reference = table_lookup['reference'] if table_lookup else None

    try:
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
    except Exception:
        current_app.logger.exception('Image preprocessing or human-verification failed.')
        flash('Unable to process this image. Please upload a clearer valid image and try again.', 'danger')
        try:
            os.remove(filepath)
        except OSError:
            pass
        return redirect(url_for('main.dashboard'))

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
            table_status=table_status,
            table_reference=table_reference,
            age=age,
            table_severity=table_severity,
        )
        details['confidence'] = confidence
    else:
        loaded_model = get_model()
        if loaded_model is False:
            if bmi is None and table_score is None:
                prediction = 0.5
                analysis_status = 'uncertain'
                assessment_basis = 'Model unavailable and BMI/WFH table data not provided'
            else:
                prediction, assessment_basis = combine_prediction_scores(None, bmi, table_score)
        else:
            raw_prediction = safe_probability(loaded_model.predict(img_array, verbose=0)[0][0])
            image_prediction = get_malnutrition_probability(raw_prediction)
            prediction, assessment_basis = combine_prediction_scores(image_prediction, bmi, table_score)
            delta = abs(prediction - float(thresholds['malnutrition_threshold']))
            if delta < float(thresholds['malnutrition_uncertain_margin']):
                analysis_status = 'uncertain'
        details = build_prediction_details(
            prediction,
            bmi,
            analysis_status=analysis_status,
            assessment_basis=assessment_basis,
            table_status=table_status,
            table_reference=table_reference,
            age=age,
            table_severity=table_severity,
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
        app_obj = current_app._get_current_object()
        threading.Thread(
            target=send_analysis_report_email_async,
            args=(app_obj, mail, user.id, analysis.id),
            daemon=True,
            name=f'analysis-email-{analysis.id}',
        ).start()
        flash(f'Analysis completed successfully. Report will be emailed to {user.email}.', 'success')
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


@bp.route('/report/<int:analysis_id>/nutriplan-30days')
@bp.route('/result/<int:analysis_id>/nutriplan-30days')
@login_required
def nutriplan_30days(analysis_id):
    user = get_authenticated_user()
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if analysis is None:
        abort(404)

    details = serialize_analysis(analysis)
    if not details.get('can_view_nutriplan'):
        flash('Diet plan is available only for malnourished human-result reports.', 'danger')
        return redirect(url_for('main.analysis_report', analysis_id=analysis.id))
    preferences = parse_nutriplan_preferences(request.args)
    plan_requested = request.args.get('generate') == '1'
    level = get_nutriplan_level(analysis, details)
    level_title, level_note = get_nutriplan_metadata(level)
    plan_days = []
    shopping_by_week = {}
    budget_tip = ''
    if plan_requested:
        plan_days, profile = build_nutriplan_30_days(level, analysis, details, preferences)
        shopping_by_week, budget_tip = build_weekly_shopping_list(profile)
    return render_template(
        'nutriplan_30days.html',
        analysis_id=analysis.id,
        result=details['result'],
        risk_level=details['risk_level'],
        nutritional_status=details['nutritional_status'],
        level=level,
        level_title=level_title,
        level_note=level_note,
        plan_requested=plan_requested,
        plan_days=plan_days,
        preferences=preferences,
        shopping_by_week=shopping_by_week,
        budget_tip=budget_tip,
    )


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
