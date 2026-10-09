from flask import Flask, request, render_template
import sklearn
import joblib
import numpy as np

obj = joblib.load(r'california.joblib')

model = obj['model']
columns = obj['columns']

print(columns)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict')
def predict():

    input_data = []

    for i in columns:
        val = request.args.get(i)
        input_data.append(float(val))

    print(input_data)

    out = model.predict([input_data])

    prediction = out[0]

    return render_template(
        'index.html',
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)