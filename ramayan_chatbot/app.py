from flask import Flask, render_template, request
import ollama

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response["message"]["content"]

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)