from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and tfidf
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]

        if message.strip() != "":
            vector = tfidf.transform([message])
            result = model.predict(vector)[0]

            # Direct label mapping
            if result == 1:
                prediction = "Spam"
            else:
                prediction = "Not Spam"
        else:
            prediction = "Please enter a message"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)