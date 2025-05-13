from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import fitz
import pytesseract
from PIL import Image
import io
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

load_dotenv()
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
CORS(app)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


def sample_text_segments_by_tokens(text, max_tokens=1024):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) <= max_tokens:
        return text  # No need to sample

    # Divide token list into beginning, middle, and end
    part_size = max_tokens // 3
    begin = tokens[:part_size]
    middle_start = len(tokens) // 2 - part_size // 2
    middle = tokens[middle_start:middle_start + part_size]
    end = tokens[-part_size:]

    sampled_tokens = begin + middle + end
    sampled_text = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
    return sampled_text

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text")
    length = data.get("size")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        sampled_text = sample_text_segments_by_tokens(text)
        summary = summarizer(sampled_text, max_length=130, min_length=30, do_sample=False)
        return jsonify({"summary": summary[0]['summary_text']})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        if doc.page_count == 0:
            return jsonify({"error": "PDF contains no pages"}), 400

        full_text = ""

        for i in range(doc.page_count):
            try:
                page = doc.load_page(i)
            except Exception as e:
                print(f"Skipping page {i} (load error): {e}")
                continue

            try:
                text = page.get_text("text")
                if text.strip():
                    full_text += text + "\n"
                    continue  # skip OCR if we already got usable text
            except Exception as e:
                print(f"Text extraction failed on page {i}: {e}")

            # Fallback to OCR
            try:
                pix = page.get_pixmap(dpi=300, alpha=False)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale improves OCR
                ocr_text = pytesseract.image_to_string(image, lang="eng")
                full_text += ocr_text + "\n"
            except Exception as ocr_e:
                print(f"OCR failed on page {i}: {ocr_e}")
                continue

        if not full_text.strip():
            return jsonify({"error": "No readable text found in this PDF."}), 400

        try:
            sampled_text = sample_text_segments_by_tokens(full_text)
            summary = summarizer(sampled_text, max_length=500, min_length=130, do_sample=False)
            return jsonify({"summary": summary[0]['summary_text']})
        except Exception as summarizer_error:
            print("Summarization failed:", summarizer_error)
            return jsonify({"error": "Text extracted but summarization failed."}), 500

    except Exception as e:
        print("PDF OCR Error:", e)
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

@app.route('/api/verify-google-token', methods=['POST'])
def verify_google_token():
    token = request.json.get('token')
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        return jsonify({
            'status': 'success',
            'user': {
                'email': idinfo['email'],
                'name': idinfo.get('name'),
                'picture': idinfo.get('picture')
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 401


if __name__ == "__main__":
    app.run(debug=True)
