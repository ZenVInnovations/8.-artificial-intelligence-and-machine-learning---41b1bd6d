from flask import Flask, request, render_template, jsonify, send_file
from PIL import Image
import pytesseract
import os
import fitz  # for PDF support (PyMuPDF)

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

authenticator = IAMAuthenticator("U0eUK2PyFxFAMoeH7kouJsbaSJZ-D02wYa5jTSZ9tRGI")
nlu = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
nlu.set_service_url("https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/ef9739eb-ebab-4cef-a42c-b13561a34afc")


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Watson NLU Credentials
api_key = "U0eUK2PyFxFAMoeH7kouJsbaSJZ-D02wYa5jTSZ9tRGI"
service_url = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/ef9739eb-ebab-4cef-a42c-b13561a34afc"

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(service_url)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    filename = file.filename
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    extracted_text = ""

    # ===== File Type Handling =====
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = pytesseract.image_to_string(Image.open(file_path))

    elif filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            extracted_text = f.read()

    elif filename.endswith('.pdf'):
        doc = fitz.open(file_path)
        for page in doc:
            extracted_text += page.get_text()

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # ===== Watson NLU Processing =====
    response = nlu.analyze(
        text=extracted_text,
        features=Features(
            keywords=KeywordsOptions(limit=5),
            entities=EntitiesOptions(limit=5)
        )
    ).get_result()
    # Save analysis results to a file
    output_file_path = os.path.join(OUTPUT_FOLDER, 'analysis_result.txt')
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        out_file.write("Extracted Text:\n")
        out_file.write(extracted_text + "\n\n")

        out_file.write("Keywords:\n")
        for k in response.get('keywords', []):
            out_file.write(f"- {k['text']} (Relevance: {k['relevance']})\n")

        out_file.write("\nEntities:\n")
        for e in response.get('entities', []):
            out_file.write(f"- {e['text']} ({e['type']}) (Relevance: {e['relevance']})\n")

    # Save result to a text file
    result_path = os.path.join("static", "result.txt")
    with open(result_path, "w", encoding='utf-8') as f:
        f.write("Extracted Text:\n")
        f.write(extracted_text + "\n\n")
        f.write("Keywords:\n")
        for kw in response['keywords']:
            f.write(f"{kw['text']} (Relevance: {kw['relevance']})\n")
        f.write("\nEntities:\n")
        for ent in response['entities']:
            f.write(f"{ent['text']} - {ent['type']} (Relevance: {ent['relevance']})\n")



    return jsonify({
    "text": extracted_text,
    "keywords": response['keywords'],
    "entities": response['entities'],
    "download_url": "/download-result"
})

@app.route('/download-result')
def download_result():
    result_path = "static/result.txt"
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    else:
        return "No result available for download.", 404


if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
