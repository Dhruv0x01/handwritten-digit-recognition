from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():          
    return render_template("index.html")  

# Create the route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received data:", data)

    return jsonify({"prediction": 5})

if __name__ == "__main__":
    app.run(debug=True)            







