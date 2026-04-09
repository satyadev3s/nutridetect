import secrets
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
import os
from flask import Flask, redirect, render_template, request, session, url_for


app = Flask(__name__, template_folder=".")
app.secret_key = "change-this-secret-key"
users = {}


@app.after_request
def add_no_store_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def load_env_file(env_path: str = "mail.env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


load_env_file(".env")
load_env_file("mail.env")
app.secret_key = os.getenv("FLASK_SECRET_KEY", app.secret_key)


def read_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


def send_otp_email(recipient_email: str, otp: str) -> tuple[bool, str]:
    load_env_file(".env")
    load_env_file("mail.env")

    smtp_host = read_env("SMTP_HOST", "MAIL_SERVER")
    smtp_port = read_env("SMTP_PORT", "MAIL_PORT")
    smtp_user = read_env("SMTP_USER", "MAIL_USERNAME")
    smtp_pass = read_env("SMTP_PASS", "MAIL_PASSWORD")
    sender_email = read_env("SENDER_EMAIL", "MAIL_DEFAULT_SENDER", default=smtp_user)

    values = [smtp_host, smtp_port, smtp_user, smtp_pass, sender_email]
    placeholders = ["your-email", "your-16-char-app-password", "replace-with"]
    has_placeholder = any(any(token in val for token in placeholders) for val in values)

    if not all(values):
        return False, "Email config missing. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SENDER_EMAIL."
    if has_placeholder:
        return False, "Replace placeholder values in mail.env/.env with real email credentials."

    msg = EmailMessage()
    msg["Subject"] = "Your OTP Code"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content(f"Your OTP is: {otp}\nIt expires after you verify your account.")

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=20) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as exc:
        return False, str(exc)


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    message = request.args.get("message", "").strip()
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = users.get(email)
        if not email or not password:
            message = "Please enter email and password."
        elif not user or user.get("password") != password:
            message = "Invalid email or password."
        else:
            message = f"Welcome, {user.get('name', 'User')}."
    return render_template("login.html", message=message)


@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if name and email and password:
            otp = f"{secrets.randbelow(1000000):06d}"
            sent, error = send_otp_email(email, otp)
            if not sent:
                message = f"Could not send OTP email: {error}"
                return render_template("register.html", message=message)

            session["pending_registration"] = {
                "name": name,
                "email": email,
                "password": password,
                "otp": otp,
            }
            return redirect(url_for("verify"))
        else:
            message = "Please fill in all fields."
    return render_template("register.html", message=message)


@app.route("/verify", methods=["GET", "POST"])
def verify():
    message = ""
    pending = session.get("pending_registration")
    if not pending:
        return redirect(url_for("register"))

    email = pending.get("email", "")
    if request.method == "POST":
        code = request.form.get("code", "").strip()
        if not code:
            message = "Please enter the verification code."
        elif code == pending.get("otp"):
            users[pending["email"]] = {
                "name": pending["name"],
                "password": pending["password"],
            }
            session.pop("pending_registration", None)
            return redirect(
                url_for("login", message="Registration complete. Please login.")
            )
        else:
            message = "Incorrect OTP. Please try again."
    return render_template("verify.html", message=message, email=email)


if __name__ == "__main__":
    app.run(debug=True)
