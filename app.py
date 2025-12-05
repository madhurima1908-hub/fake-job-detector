from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Path to joblib (ensure file is at repo root)
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "fake_job_pipeline.joblib")
bundle = joblib.load(BUNDLE_PATH)

tfidf = bundle.get("tfidf")
ohe = bundle.get("ohe")
num_cols = bundle.get("num_cols", [])
cat_cols = bundle.get("cat_cols", [])
model = bundle.get("model")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        title = request.form.get("title", "")
        location = request.form.get("location", "")
        description = request.form.get("description", "")

        # Build a 1-row dataframe matching training-time columns
        df = pd.DataFrame([{
            "title": title,
            "location": location,
            "company_profile": "",
            "description": description,
            "industry": "",
            "function": "",
            "employment_type": "",
            "required_experience": "",
            "required_education": "",
            "telecommuting": 0,
            "has_company_logo": 0,
            "has_questions": 0
        }])

        # Combine text exactly as training
        df["all_text"] = df["title"].astype(str) + " " + df["company_profile"].astype(str) + " " + df["description"].astype(str)

        # Transform text with tfidf
        X_text = tfidf.transform(df["all_text"]).toarray()

        # Transform categorical if present
        if ohe is not None and len(cat_cols) > 0:
            X_cat = ohe.transform(df[cat_cols])
        else:
            X_cat = np.zeros((1, 0))

        # Numeric
        if len(num_cols) > 0:
            X_num = df[num_cols].values.astype(float)
        else:
            X_num = np.zeros((1, 0))

        # Combine to final feature vector
        X = np.hstack([X_text, X_cat, X_num])

        # Predict
        result = model.predict(X)[0]
        prediction = "Fake Job Posting ❌" if int(result) == 1 else "Real Job Posting ✔"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
