from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load("fake_job_pipeline.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        title = request.form.get("title")
        location = request.form.get("location")
        description = request.form.get("description")

        data = np.array([[title, location, description]])
        result = model.predict(data)[0]

        prediction = "Fake Job Posting ❌" if result == 1 else "Real Job Posting ✔"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
