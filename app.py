from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pipeline bundle
bundle = joblib.load("fake_job_pipeline.joblib")
model = bundle["model"]
tfidf = bundle["tfidf"]
ohe = bundle.get("ohe")
num_cols = bundle.get("num_cols", [])
cat_cols = bundle.get("cat_cols", [])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        # Get input from form
        title = request.form.get("title", "")
        location = request.form.get("location", "")
        description = request.form.get("description", "")

        # Build a 1-row dataframe with same structure as training
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

        # Combine text for TF-IDF
        df["all_text"] = df["title"] + " " + df["company_profile"] + " " + df["description"]
        X_text = tfidf.transform(df["all_text"]).toarray()

        # Transform categorical features
        if ohe is not None and len(cat_cols) > 0:
            X_cat = ohe.transform(df[cat_cols])
        else:
            X_cat = np.zeros((1, 0))

        # Numeric features
        if len(num_cols) > 0:
            X_num = df[num_cols].values.astype(float)
        else:
            X_num = np.zeros((1, 0))

        # Combine all features
        X = np.hstack([X_text, X_cat, X_num])

        # Predict
        result = model.predict(X)[0]
        prediction = "Fake Job Posting ❌" if int(result) == 1 else "Real Job Posting ✔"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
