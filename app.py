from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_cors import CORS
import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import base64
import requests
import json
import re
import tempfile
import uuid
import subprocess
import sys
import urllib.parse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGTEXT

# --- CONFIGURATION (MUST BE ADJUSTED FOR YOUR SYSTEM) ---
POPPLER_PATH = r"PUT YOUR POPPLER PATH" # <--- VERIFY/UPDATE THIS PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- VERIFY/UPDATE THIS PATH

app = Flask(__name__)
CORS(app)
app.secret_key = 'super_secret_key_for_flash_messages' 

# --- ACCESS CONTROL & API KEYS ---
OPERATOR_SECRET_KEY = "PUT YOUR KEY" 
OPENROUTER_API_KEY_HARDCODED = "PUT YOUR KEY"
# ---------------------------------

# --- DATABASE CONFIGURATION ---
mysql_password = urllib.parse.quote_plus("PUT YOUR PASSWORD") 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:' + mysql_password + '@localhost:3306/medical_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DATABASE MODEL ---
class DocumentAnalysis(db.Model):
    __tablename__ = 'document_analysis'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    gender = db.Column(db.String(20))
    tag = db.Column(db.String(50))
    imaging_type = db.Column(db.String(50))
    body_part_imaged = db.Column(db.String(100))
    medical_tests = db.Column(db.Text)
    medicines = db.Column(db.Text)
    doctor_name = db.Column(db.String(100))
    total_bill = db.Column(db.String(50))
    extracted_text = db.Column(LONGTEXT)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'gender': self.gender,
            'tag': self.tag,
            'imaging_type': self.imaging_type,
            'body_part_imaged': self.body_part_imaged,
            'medical_tests': json.loads(self.medical_tests) if self.medical_tests else [],
            'medicines': json.loads(self.medicines) if self.medicines else [],
            'doctor_name': self.doctor_name,
            'total_bill': self.total_bill,
            'extracted_text': self.extracted_text,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# --- HELPER FUNCTIONS (OCR & LLM) ---

def check_executables():
    poppler_exists = os.path.exists(POPPLER_PATH) and os.path.isdir(POPPLER_PATH)
    tesseract_exists = os.path.exists(pytesseract.pytesseract.tesseract_cmd) and os.path.isfile(pytesseract.pytesseract.tesseract_cmd)
    # ... (rest of check_executables remains the same)

def pdf_to_images(pdf_path):
    """Converts a PDF file into a list of PIL Image objects."""
    try:
        return convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)
    except Exception as e:
        raise Exception(f"Error during PDF to image conversion (pdf2image/Poppler). Check POPPLER_PATH: {POPPLER_PATH}. Error: {e}")

def image_to_base64(image):
    """Converts a PIL Image object to a base64 encoded string."""
    from io import BytesIO
    buffer = BytesIO()
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

def send_to_llava_for_tagging_and_details(base64_image, full_document_text):
    """
    Sends the document's first page image and extracted text to LLaVA for analysis.
    """
    openrouter_api_key = OPENROUTER_API_KEY_HARDCODED
    if not openrouter_api_key:
        return {"tag": "Other", "error": "API_KEY_MISSING"}
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    model_id = "google/gemma-3n-e4b-it:free"

    # Minimal prompt to stay within token limits and focus on JSON output
    prompt_text = f"""
    Analyze this document based on its image and text. Identify the document's category (tag) and extract relevant details.
    
    *Document Categories (Choose One):* X-ray, MRI, CT Scan, Ultrasound, Prescription, Medical Bill, Report (if medical), Receipt, Other.
    
    *Required Extraction:*
    - **Tag** (String, required)
    - If Tag is an **Imaging** type: **imaging_type** (e.g., 'X-ray') and **body_part_imaged** (e.g., 'Chest').
    - If Tag is **Report**: **medical_tests** (Array of Strings). Set **is_medical_report** to true if medical.
    - If Tag is **Prescription/Medical Bill**: **medicines** (Array of Strings).
    
    *Your response MUST be a clean JSON object.*
    
    Full Document Text:
    {full_document_text}
    """

    data = {
        "model": model_id,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_content = response.json()["choices"][0]["message"]["content"]

        match = re.search(r"\{[\s\S]*?\}", response_content)
        if match:
            json_string = match.group(0)
            return json.loads(json_string)
        else:
            # If JSON parsing fails, include the raw response for post-processing check
            return {"tag": "Other", "error": "NO_JSON_FOUND", "raw_response": response_content}

    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"tag": "Other", "error": "API_CALL_FAILED", "api_response": str(e)}

def post_process_tag(llava_response, full_document_text):
    """
    Performs secondary classification and detail extraction based on strong keywords,
    with improved medicine extraction logic and visual-only fallback.
    """
    processed_response = llava_response.copy()
    current_tag = processed_response.get("tag", "Other").lower()
    text_lower = full_document_text.lower()
    
    # --- NEW CHECK: Visual-only document fallback ---
    if not full_document_text.strip():
        # If there is no text, we assume it's a visual scan like an X-ray.
        # We rely on the raw LLM response (which is more descriptive than the final JSON)
        raw_llm_output = str(llava_response.get('raw_response', '')).lower()
        if 'chest x-ray' in raw_llm_output or 'radiograph of the chest' in raw_llm_output or 'thorax' in raw_llm_output:
            processed_response["tag"] = "X-ray"
            processed_response["imaging_type"] = "X-ray"
            processed_response["body_part_imaged"] = "Chest"
            return processed_response # Exit early if a clear visual tag is found
        elif 'mri' in raw_llm_output or 'ct scan' in raw_llm_output or 'ultrasound' in raw_llm_output:
             # Basic fallback if a different imaging type is mentioned visually
            if 'mri' in raw_llm_output:
                processed_response["tag"] = "MRI"
                processed_response["imaging_type"] = "MRI"
            elif 'ct scan' in raw_llm_output:
                processed_response["tag"] = "CT Scan"
                processed_response["imaging_type"] = "CT Scan"
            elif 'ultrasound' in raw_llm_output:
                processed_response["tag"] = "Ultrasound"
                processed_response["imaging_type"] = "Ultrasound"
            # Attempt to set body part
            body_parts = ["brain", "knee", "spine", "abdomen", "pelvis", "hand", "foot", "shoulder", "hip"]
            for bp in body_parts:
                 if bp in raw_llm_output:
                     processed_response["body_part_imaged"] = bp.capitalize()
                     break
            return processed_response
    # --- END NEW CHECK ---
    
    # Define keywords (for robust fallback)
    xray_keywords = ["x-ray", "radiograph"]
    mri_keywords = ["mri", "magnetic resonance imaging"]
    ct_keywords = ["ct scan", "computed tomography"]
    ultrasound_keywords = ["ultrasound", "sonography"]
    
    # --- CRITICAL FIX FOR PRESCRIPTION/BILL (TEXT-BASED) ---
    if current_tag == "other":
        prescription_keywords = ["sig.", "dispense", "refills", "qty", "tablet", "capsule", "mg", "ml", "dr.", "m.d.", "d.o.", "po q"]
        bill_keywords = ["invoice", "total due", "amount paid", "charges", "billing statement", "cost of services"]
        
        if any(keyword in text_lower for keyword in prescription_keywords):
            processed_response["tag"] = "Prescription"
            processed_response["imaging_type"] = None
            processed_response["body_part_imaged"] = None
            current_tag = "prescription" 
            
        elif any(keyword in text_lower for keyword in bill_keywords):
            processed_response["tag"] = "Medical Bill"
            processed_response["imaging_type"] = None
            processed_response["body_part_imaged"] = None
            current_tag = "medical bill" 
    # --- END CRITICAL FIX ---

    # Post-processing for Imaging Tags (Text-based confirmation)
    if any(keyword in text_lower for keyword in xray_keywords):
        processed_response["tag"] = "X-ray"
        processed_response["imaging_type"] = "X-ray"
    elif any(keyword in text_lower for keyword in mri_keywords):
        processed_response["tag"] = "MRI"
        processed_response["imaging_type"] = "MRI"
    elif any(keyword in text_lower for keyword in ct_keywords):
        processed_response["tag"] = "CT Scan"
        processed_response["imaging_type"] = "CT Scan"
    elif any(keyword in text_lower for keyword in ultrasound_keywords):
        processed_response["tag"] = "Ultrasound"
        processed_response["imaging_type"] = "Ultrasound"
        
    if processed_response.get("imaging_type"):
        body_parts = ["chest", "brain", "knee", "spine", "abdomen", "pelvis", "hand", "foot", "shoulder", "hip"]
        found_body_part = "Unknown"
        for bp in body_parts:
            if bp in text_lower:
                found_body_part = bp.capitalize()
                break
        processed_response["body_part_imaged"] = found_body_part

    # Post-processing for Report Tags (Medical Tests) - Unchanged
    if current_tag == "report" and processed_response.get("is_medical_report", False) and not processed_response.get("medical_tests"):
        medical_test_keywords = ["complete blood count", "lipid profile", "urinalysis", "lft", "kft"]
        found_tests = []
        for test_kw in medical_test_keywords:
            if re.search(r'\b' + re.escape(test_kw) + r'\b', text_lower):
                found_tests.append(test_kw.title())
        if found_tests:
            processed_response["medical_tests"] = list(set(found_tests))
    
    # --- Medicine Extraction Logic ---
    if current_tag in ["medical bill", "prescription"] and not processed_response.get("medicines"):
        
        pattern_units = r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}\s*\d{1,4}(?:\.\d+)?\s*(?:\(?\w+\)?\s*)?(?:mg|g|mcg|ml|iu|units|capsules|tabs|tablets|po|prn|q\d+h|bid|tid|qid|tab|cap|caps))'
        pattern_sig = r'(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3})\s+\d{1,4}(?:\s+\(sig|sig\.)'

        found_medicines = re.findall(pattern_units, text_lower, re.IGNORECASE)
        found_medicines.extend(re.findall(pattern_sig, text_lower, re.IGNORECASE))
        
        strict_exclusion_list = ["john", "anna", "smith", "johnson", "date", "name", "age", "phone", "address", 
                                 "gender", "email", "male", "female", "doctor", "street", "city", "state", 
                                 "sig", "po", "prn", "q8h", "q12h", "q6h", "bid", "tid", "qid", "capsules", "capsule", "tablet", "tabs", "tab", "cap"]
        
        filtered_medicines = []
        for med in found_medicines:
            med_parts = med.lower().split()
            
            if med_parts[0] not in strict_exclusion_list and len(med) > 3: 
                cleaned_med = re.sub(r'\s*\(\w+\)\s*', ' ', med).strip() 
                
                if any(kw in cleaned_med.lower() for kw in ['mg', 'ml', 'g', 'po', 'prn', 'sig', 'capsules', 'tablet']):
                    
                    if any(s in cleaned_med.lower() for s in ['sig', 'po', 'prn']):
                        cleaned_med = cleaned_med.split(' sig')[0].split(' po')[0].strip()
                        
                    filtered_medicines.append(cleaned_med.title())

        processed_response["medicines"] = list(set(filtered_medicines))
        
    return processed_response


# --- FLASK ROUTES (Unchanged) ---

@app.route('/')
def index():
    """Serves the main document upload form page."""
    return render_template('index.html')

@app.route('/result')
def result_page():
    """Serves the analysis result page (data handled client-side)."""
    return render_template('result.html')

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    # ... (analyze_document function body is unchanged)
    if 'document_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    gender = request.form.get('gender')

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    safe_filename = str(uuid.uuid4()) + file_extension

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, safe_filename)
    file.save(file_path)

    images = []
    all_extracted_text_pages = []

    try:
        ext = os.path.splitext(safe_filename)[-1].lower()

        if ext == ".pdf":
            images = pdf_to_images(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            images = [Image.open(file_path)]
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        if not images:
            return jsonify({"error": "No images could be processed from the file."}), 500

        for i, image in enumerate(images):
            if image.mode != 'RGB':
                image = image.convert("RGB")
            page_text = pytesseract.image_to_string(image)
            all_extracted_text_pages.append(f"--- Page {i+1} ---\n{page_text.strip()}")

        full_document_text = "\n\n".join(all_extracted_text_pages)

        first_image_base64 = image_to_base64(images[0])
        llava_parsed_response = send_to_llava_for_tagging_and_details(first_image_base64, full_document_text)

        final_analysis = post_process_tag(llava_parsed_response, full_document_text)

        # --- DATABASE STORAGE ---
        medical_tests_json = json.dumps(final_analysis.get('medical_tests', []))
        medicines_json = json.dumps(final_analysis.get('medicines', []))

        new_record = DocumentAnalysis(
            name=name,
            email=email,
            phone=phone,
            gender=gender,
            tag=final_analysis.get('tag'),
            imaging_type=final_analysis.get('imaging_type'),
            body_part_imaged=final_analysis.get('body_part_imaged'),
            medical_tests=medical_tests_json,
            medicines=medicines_json,
            doctor_name=final_analysis.get('doctor_name'),
            total_bill=final_analysis.get('total_bill'),
            extracted_text=full_document_text
        )

        db.session.add(new_record)
        db.session.commit()
        print(f"Record saved to database with ID: {new_record.id}")
        # --- END DATABASE STORAGE ---

        # Prepare response JSON for result.html via session storage
        final_analysis['user_info'] = {
            "name": name,
            "email": email,
            "phone": phone,
            "gender": gender
        }
        final_analysis['full_document_text'] = full_document_text
        final_analysis['record_id'] = new_record.id

        return jsonify(final_analysis), 200

    except Exception as e:
        print(f"An unexpected error occurred during document analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    finally:
        # Cleanup temp directory and files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

# --- SECURED OPERATOR ROUTES (Unchanged) ---

@app.route('/records', methods=['GET', 'POST'])
def view_records():
    # ... (view_records remains the same)
    error_message = None

    if request.method == 'POST':
        operator_password = request.form.get('operator_password')
        if operator_password == OPERATOR_SECRET_KEY:
            session['logged_in'] = True  # Successful login
            flash('Logged in successfully!', 'success')
            return redirect(url_for('view_records')) 
        else:
            error_message = "Access Denied: Incorrect password."
            return render_template('records.html', records=[], show_login_form=True, error_message=error_message)
    
    # GET request check
    if not session.get('logged_in'):
        return render_template('records.html', records=[], show_login_form=True)

    # If logged in, fetch and display records
    try:
        # Fetching records in reverse order (newest first)
        all_records = DocumentAnalysis.query.order_by(DocumentAnalysis.created_at.desc()).all() 
        records_data = [record.to_dict() for record in all_records]
        return render_template('records.html', records=records_data, show_login_form=False)
    except Exception as e:
        flash(f"Could not load records: {str(e)}", 'danger')
        return render_template('records.html', records=[], show_login_form=False, error_message=f"DB Error: {str(e)}")

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('view_records'))

@app.route('/records/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized access. Please log in."}), 403
    
    try:
        record_to_delete = DocumentAnalysis.query.get_or_404(record_id)
        db.session.delete(record_to_delete)
        db.session.commit()
        return jsonify({"message": f"Record {record_id} deleted successfully"}), 200
    except Exception as e:
        print(f"Error deleting record {record_id}: {e}")
        return jsonify({"error": f"Failed to delete record: {str(e)}"}), 500

@app.route('/records/update/<int:record_id>', methods=['GET', 'POST'])
def update_record(record_id):
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized access. Please log in."}), 403
    
    return jsonify({"message": f"Update route for record {record_id} reached. Logic not fully implemented."}), 200

if __name__ == '__main__':
    check_executables()
    with app.app_context():
        db.create_all()
        print("Database tables checked/created.")
    app.run(debug=True, port=5000)
