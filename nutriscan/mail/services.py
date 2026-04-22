from flask import current_app
from flask_mail import Message


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


def is_mail_configured():
    return bool(current_app.config.get('MAIL_USERNAME') and current_app.config.get('MAIL_PASSWORD'))


def send_password_reset_email(mail, recipient_name, recipient_email, reset_url):
    if not is_mail_configured():
        return False, 'Email is not configured yet.'

    if mail is None:
        return False, 'Mail service is not initialized.'

    message = Message(
        subject='NutriDetect Password Reset',
        recipients=[recipient_email],
        body="\n".join([
            f'Hello {recipient_name},',
            '',
            'We received a request to reset your NutriDetect password.',
            f'Open this link to choose a new password: {reset_url}',
            '',
            'This link will expire in 1 hour.',
            'If you did not request this, you can ignore this email.',
        ]),
    )
    try:
        mail.send(message)
        return True, None
    except Exception as exc:
        current_app.logger.exception('Failed to send password reset email')
        return False, str(exc)


def send_registration_otp_email(mail, recipient_email, otp_code):
    if not is_mail_configured():
        return False, 'Email is not configured yet.'

    if mail is None:
        return False, 'Mail service is not initialized.'

    message = Message(
        subject='NutriDetect Registration OTP',
        recipients=[recipient_email],
        body="\n".join([
            'Welcome to NutriDetect!',
            '',
            f'Your OTP code is: {otp_code}',
            '',
            'Enter this code on the verification page to complete your account registration.',
        ]),
    )
    try:
        mail.send(message)
        return True, None
    except Exception as exc:
        current_app.logger.exception('Failed to send registration OTP email')
        return False, str(exc)


def send_analysis_report_email(mail, recipient_name, recipient_email, analysis_id, details):
    if not is_mail_configured():
        return False, 'Email is not configured yet.'

    if mail is None:
        return False, 'Mail service is not initialized.'

    body = "\n".join([
        'NutriDetect Analysis Report',
        '',
        f'Hello {recipient_name},',
        '',
        'Your nutrition analysis is complete.',
        f'Report ID: {analysis_id}',
        f"Generated: {details.get('created_at', 'N/A')}",
        f"AI Result: {details.get('result', 'N/A')}",
        f"Nutritional Status: {details.get('nutritional_status', 'N/A')}",
        f"Risk Level: {details.get('risk_level', 'N/A')}",
        f"Confidence: {details.get('confidence', 'N/A')}%",
        f"BMI: {details.get('bmi_value', 'N/A')}",
        f"BMI Category: {details.get('bmi_category', 'N/A')}",
        f"Protein Guidance: {details.get('protein_recommendation', 'N/A')}",
        f"Calorie Guidance: {details.get('calorie_recommendation', 'N/A')}",
        '',
        'Recommendations:',
        *[f"- {item}" for item in details.get('recommendations', [])],
        '',
        'Thanks for using NutriDetect.',
    ])

    message = Message(
        subject=f'NutriDetect Report #{analysis_id}',
        recipients=[recipient_email],
        body=body,
    )
    try:
        mail.send(message)
        return True, None
    except Exception as exc:
        current_app.logger.exception('Failed to send analysis report email')
        return False, str(exc)
