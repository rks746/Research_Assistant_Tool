from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from summarizer import generate_summary
from searcher import Searcher
import pdfplumber

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize searcher
searcher = Searcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Extract text from PDF
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    summary = generate_summary(text)
    searcher.add_document(file.filename, text)  # Index the document

    print ("Extracted text length: ", len(text))
    print ("Extracted text length: ", text[:500])
    
    return jsonify({"summary": summary})

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is missing"}), 400
    
    results = searcher.search(query)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
