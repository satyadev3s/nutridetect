# Deploy NutriDetect On Render (Free)

## 1) Push latest code

```powershell
cd C:\Users\durga\Desktop\dataset\templates\final_dti
git push origin main
```

## 2) Create service on Render

1. Open Render dashboard: https://dashboard.render.com/
2. Click `New +` -> `Blueprint`.
3. Select repo: `satyadev3s/nutridetect`.
4. Render auto-detects `render.yaml`.
5. Click `Apply`.

## 3) Set required environment variables

In Render service -> `Environment`, add:

- `MAIL_SERVER` = `smtp.gmail.com`
- `MAIL_PORT` = `587`
- `MAIL_USE_TLS` = `True`
- `MAIL_USERNAME` = your sender Gmail
- `MAIL_PASSWORD` = your Gmail app password
- `MAIL_DEFAULT_SENDER` = same sender Gmail
- `APP_TIMEZONE` = `Asia/Kolkata` (optional)

`SECRET_KEY` and `FLASK_DEBUG` are already configured by `render.yaml`.

## 4) Redeploy

After env vars are saved, click `Manual Deploy` -> `Deploy latest commit`.

## 5) Verify

- Open your Render URL.
- Check health endpoint:
  - `https://<your-service>.onrender.com/healthz`
  - Expected response: `ok`

## Notes

- This project currently uses SQLite (`nutriscan/instance/nutridetect.db`).
- On free hosting, local filesystem can be ephemeral, so DB/uploads may reset after restarts.
- For production reliability, use managed Postgres + object storage later.
