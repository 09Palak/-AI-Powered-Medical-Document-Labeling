# 🏥 Medical Document Analyzer

A **Flask-based web application** that allows users to upload medical documents (PDFs or images) and automatically analyze them using **OCR** and **AI-based document understanding**.  
The system classifies documents and extracts structured medical information.

---

## 🚀 Features

- 📄 Upload medical documents (**PDF / JPG / PNG**)
- 🔍 OCR-based text extraction using **Tesseract**
- 🧠 AI-powered document classification
- 🏷️ Automatic document tagging:
  - Prescription
  - Medical Report
  - X-ray
  - MRI
  - CT Scan
  - Ultrasound
  - Medical Bill
- 💊 Extract structured data:
  - Medicines
  - Medical tests
  - Imaging type & body part
- 🗄️ Stores analysis results in **MySQL**
- 🔐 Operator-only dashboard for viewing & managing records

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Backend | Flask (Python) |
| Frontend | HTML, CSS, Bootstrap |
| OCR | Tesseract OCR |
| PDF Processing | pdf2image, Poppler |
| AI / LLM | Vision + Text LLM (via API) |
| Database | MySQL |
| ORM | SQLAlchemy |

---

## 📁 Project Structure

