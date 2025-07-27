# Smart Invoice AI System

A Streamlit-based application that extracts and manages invoice data using the LayoutLM model from the `impira/layoutlm-invoices` pre-trained model. This system processes uploaded invoices (PDFs or images), extracts key fields, allows manual corrections, and provides an admin dashboard for statistics.

## Features
- Extracts invoice fields: invoice number, date, supplier, total amount, VAT, and line items.
- Displays extraction results with confidence scores.
- Provides an editable interface to correct extracted fields.
- Saves corrections to a local SQLite database.
- Includes an admin dashboard with supplier statistics.
- Simulates model fine-tuning with a downloadable model version.
- Supports CSV, PDF, and image file uploads (PNG, JPG, JPEG).
- Offers downloadable CSV results and the best model file.

## Requirements
- **Python 3.8+**
- Required libraries (install via `requirements.txt`):
  - `streamlit==1.24.0`
  - `pandas==2.0.3`
  - `pytesseract==0.3.10`
  - `pdf2image==1.16.3`
  - `pillow==10.0.0`
  - `torch==2.0.1`
  - `transformers==4.30.2`
  - `numpy==1.24.3`

- **Tesseract OCR**: Install Tesseract-OCR on your system (e.g., from [here](https://github.com/UB-Mannheim/tesseract/wiki)) and set the path in the code (`pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`). Adjust the path based on your OS.

- **Poppler**: Required for PDF processing with `pdf2image`. Install via your package manager (e.g., `sudo apt install poppler-utils` on Ubuntu).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Smart-Invoice-AI.git
   cd Smart-Invoice-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Tesseract and Poppler**:
   - Ensure Tesseract is installed and the path is correctly set in the code.
   - Install Poppler for PDF support.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser and go to the provided URL (e.g., `http://localhost:8501`).

## Usage
- **Upload Files**: Use the file uploader to submit invoices in CSV, PDF, or image formats.
- **View Results**: Extracted fields (invoice number, date, supplier, total, VAT, line items) are displayed with confidence scores.
- **Edit Fields**: Manually correct any extracted fields or line items using the edit interface.
- **Save Corrections**: Click "Save Corrections" to store changes in the database.
- **Download Results**: Export extracted data as a CSV file.
- **Admin Dashboard**: View supplier statistics in the sidebar.
- **Fine-Tune Model**: Click "Fine-Tune Model" to simulate model improvement (requires at least 5 corrections).

## Project Structure
- `app.py`: Main Streamlit application file containing UI and integration logic.
- Functions included:
  - `init_db()`: Sets up the SQLite database.
  - `extract_text()`: Extracts text from uploaded files using Tesseract OCR.
  - `parse_line_items()`: Parses line items from text.
  - `extract_fields()`: Extracts invoice fields using LayoutLM and rule-based logic.
  - `save_corrections()`: Saves manual corrections to the database.
  - `fine_tune_model()`: Simulates model fine-tuning and saves the model state.
  - `main()`: Runs the Streamlit app.

## Notes
- The application uses a pre-trained LayoutLM model (`impira/layoutlm-invoices`) for field extraction. Confidence scores are assigned (0.9 for extracted, 0.5 for missing).
- The `models` directory is created to store fine-tuned model files.
- Raw OCR text is displayed for debugging purposes (`st.write("Raw OCR Text:", text)`).

## Future Work
- **Fine-Tune the Model**: Implement a more robust fine-tuning process using real invoice data and corrections stored in the database to improve extraction accuracy, potentially with labeled datasets and hyperparameter tuning.
- **Better Auto Fine-Tune Mechanism**: Develop an automated fine-tuning pipeline that triggers based on a threshold of correction data (e.g., 10+ corrections), integrates with the LayoutLM model, and evaluates performance metrics (e.g., F1 score) to select the best model version.
- **Better App Interface**: Enhance the user interface with a more intuitive layout, including a progress bar for file processing, a collapsible edit panel, and responsive design for mobile devices using custom CSS or Streamlit components.

## License
[MIT License](LICENSE) (Add a `LICENSE` file with MIT terms if desired.)

## Contributing
Feel free to fork this repository, submit issues, or create pull requests for improvements.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/) for the UI.
- Utilizes [Transformers](https://huggingface.co/transformers/) and [LayoutLM](https://huggingface.co/impira/layoutlm-invoices) for NLP tasks.
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract).
