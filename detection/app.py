from flask import Flask, render_template, request
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Load the web page

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]  # Get user input
        transformed_text = vectorizer.transform([news_text]) 
         # Convert to TF-IDF
        if transformed_text.nnz == 0:  # nnz = number of non-zero elements
               return "తెలియని / అనిశ్చితమైన"
               
        prediction = model.predict(transformed_text)[0]  # Make prediction
        
        result = "✅ నిజమైన వార్తలు" if prediction == 1 else "❌ తప్పుడు వార్తలు!"
        
        return render_template("index.html", prediction=result)  # Show result on the page

if __name__ == "__main__":
    app.run(debug=True)
