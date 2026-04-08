# Disease-Aware Quantum Protein Folding Analyzer
### Deployment Guide — Render.com

---

## Step 1 — Push to GitHub

1. Go to github.com → New Repository
2. Name it: `quantum-protein-analyzer`
3. Set to **Public**
4. Click **Create Repository**

Then in your project folder, open terminal and run:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/quantum-protein-analyzer.git
git push -u origin main
```

⚠️ Your `firebase_service_account.json` will NOT be uploaded (protected by .gitignore)

---

## Step 2 — Deploy on Render

1. Go to **render.com** and sign up (free)
2. Click **New** → **Web Service**
3. Connect your GitHub account
4. Select your `quantum-protein-analyzer` repository
5. Fill in these settings:
   - **Name:** quantum-protein-analyzer
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 app:app`
6. Click **Advanced** → **Add Environment Variable**

---

## Step 3 — Add Firebase Credentials as Environment Variable

This is the most important step — your credentials stay secret.

1. Open your `firebase_service_account.json` file in Notepad
2. Select ALL the text (Ctrl+A) and copy it (Ctrl+C)
3. In Render → Environment Variables, add:
   - **Key:** `FIREBASE_CREDENTIALS`
   - **Value:** paste the entire JSON content
4. Click **Save**

---

## Step 4 — Deploy

1. Click **Create Web Service**
2. Wait 5-10 minutes for build to complete
3. Your URL will be: `https://quantum-protein-analyzer.onrender.com`

---

## Step 5 — Keep It Always Awake (Free)

Render free tier sleeps after 15 mins of inactivity.
Fix this with UptimeRobot:

1. Go to **uptimerobot.com** → sign up free
2. Click **Add New Monitor**
3. Set:
   - Monitor Type: HTTP(s)
   - Friendly Name: Quantum Protein
   - URL: your Render URL
   - Monitoring Interval: 5 minutes
4. Click **Create Monitor**

Your site will now stay awake 24/7 for free.

---

## Local Development (unchanged)

```bash
python app.py
# Opens at http://localhost:5000
```

Firebase credentials still load from `firebase_service_account.json` locally.
