# Adversarial Sales Forecaster – Module 1
### Data Preprocessing & Anomaly Detection

A Flask web app that runs the full preprocessing pipeline in your browser.
Deployed on [Railway](https://railway.com).

---

## 🗂 Project Structure

```
railway_project/
├── app.py              ← Flask app + full pipeline
├── requirements.txt    ← Python dependencies
├── Procfile            ← Tells Railway how to start the server
└── README.md
```

---

## 🚀 Deploy to Railway (Step-by-Step)

### Step 1 – Push to GitHub

1. Create a new GitHub repository (e.g. `sales-forecaster`)
2. Upload these 4 files into it (drag & drop on GitHub or use Git CLI):
   - `app.py`
   - `requirements.txt`
   - `Procfile`
   - `README.md`

```bash
# Or via terminal:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/sales-forecaster.git
git push -u origin main
```

### Step 2 – Deploy on Railway

1. Go to [railway.com](https://railway.com) and sign in (GitHub login recommended)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `sales-forecaster` repository
5. Railway will auto-detect it as a Python app ✅
6. Click **"Deploy"** — Railway installs packages and starts the server

### Step 3 – Get Your Public URL

1. Once deployed, go to your project's **Settings** tab
2. Under **Networking**, click **"Generate Domain"**
3. You'll get a URL like: `https://sales-forecaster.up.railway.app`
4. Open it in your browser and click **▶ Run Pipeline**!

---

## ⚙️ How It Works

| Step | What happens |
|------|-------------|
| 1 | Generates 730 days of realistic dummy sales data |
| 2 | Injects 18 anomalies + ~3% missing values |
| 3 | KNN Imputation fills missing values |
| 4 | IQR flags statistical outliers |
| 5 | Isolation Forest flags ML-detected anomalies |
| 6 | Ensemble: only flag if BOTH agree (reduces false positives) |
| 7 | Rolling median corrects detected anomalies |
| 8 | 4-panel chart rendered and displayed in browser |

---

## 🔧 Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```
