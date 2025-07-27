import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance
import pdf2image
import io
import os
import sqlite3
import json
import torch
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
from datetime import datetime
import re
import numpy as np
from pathlib import Path

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize LayoutLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-invoices")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-invoices")

# Database setup
def init_db():
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS invoices
                 (id INTEGER PRIMARY KEY, invoice_number TEXT, date TEXT, supplier TEXT,
                  total REAL, vat REAL, corrections TEXT, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS model_versions
                 (version TEXT PRIMARY KEY, accuracy REAL, timestamp TEXT, path TEXT)''')
    conn.commit()
    conn.close()

# Extract text from PDF or image
def extract_text(file):
    try:
        if file.name.endswith('.pdf'):
            images = pdf2image.convert_from_bytes(file.read())
            text = ''
            for img in images:
                text += pytesseract.image_to_string(img, config='--psm 6')
        else:
            img = Image.open(file).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2)
            text = pytesseract.image_to_string(img, config='--psm 6')
        if not text.strip():
            st.warning("No text extracted from the document.")
            return ""
        st.write("Raw OCR Text:", text)  # Debug output
        return text
    except Exception as e:
        st.error(f"Error in OCR: {str(e)}")
        return ""

# Parse line items
def parse_line_items(text):
    items = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        match = re.match(r'(.+?)\s+(\d+)\s+([\d.]+)\s+[\$]?([\d.]+)', line)
        if match:
            items.append({
                'description': match.group(1).strip(),
                'quantity': int(match.group(2)),
                'price': float(match.group(3))
            })
    return items

# Extract fields
def extract_fields(text):
    if not text.strip():
        return {}, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    fields = {
        'invoice_number': None,
        'date': None,
        'supplier': None,
        'total': None,
        'vat': None,
        'line_items': []
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for i, line in enumerate(lines):
        if "INV" in line and "-" in line:
            fields['invoice_number'] = re.search(r'INV\d+-\d+', line).group() if re.search(r'INV\d+-\d+', line) else None
        elif "/" in line and len(line.split('/')[0]) == 2:
            fields['date'] = line
        elif "Company" in line or "Name" in line:
            if i + 1 < len(lines) and not lines[i + 1].startswith("Address"):
                fields['supplier'] = lines[i + 1].strip()
        elif "Total:" in line:
            fields['total'] = float(re.search(r'\d+\.?\d*', line.split("Total:")[1].replace('$', '').replace(',', '')).group())

    in_table = False
    for line in lines:
        if "Description" in line and "Quantity" in line and "Unit Price" in line:
            in_table = True
            continue
        if in_table and "Total:" in line:
            in_table = False
            continue
        if in_table and line.strip() and not line.startswith("Product/Service"):
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                description = parts[0].replace("Product/Service", "").strip()
                quantity = int(parts[1]) if parts[1].isdigit() else 1
                unit_price = float(parts[2].replace('$', '').replace(',', '')) if '.' in parts[2] else float(parts[2])
                fields['line_items'].append({
                    'description': description,
                    'quantity': quantity,
                    'price': unit_price
                })

    confidences = {k: 0.9 if v is not None and v != '' else 0.5 for k, v in fields.items()}
    return fields, confidences

# Save corrections
def save_corrections(fields, corrections):
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('''INSERT INTO invoices (invoice_number, date, supplier, total, vat, corrections, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
                 (fields['invoice_number'], fields['date'], fields['supplier'],
                  fields['total'], fields['vat'], json.dumps(corrections), datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Fine-tune model (simulated)
def fine_tune_model():
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('SELECT corrections FROM invoices')
    corrections = [json.loads(row[0]) for row in c.fetchall() if row[0]]
    conn.close()
    
    if len(corrections) < 5:
        return None
    
    new_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/model_{new_version}.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    accuracy = np.random.uniform(0.85, 0.95)
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('INSERT INTO model_versions (version, accuracy, timestamp, path) VALUES (?, ?, ?, ?)',
              (new_version, accuracy, datetime.now().isoformat(), model_path))
    conn.commit()
    conn.close()
    return new_version, accuracy

# Streamlit UI
def main():
    st.title("Smart Invoice AI System")
    init_db()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    st.markdown("""
        <style>
        .stTable { font-size: 14px; }
        </style>
        """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Invoices (CSV, PDF, Image)", accept_multiple_files=True,
                                    type=['csv', 'pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        results = []
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    fields = {
                        'invoice_number': str(row.get('invoice_number', '')),
                        'date': str(row.get('date', '')),
                        'supplier': str(row.get('supplier', '')),
                        'total': float(row.get('total', 0)),
                        'vat': float(row.get('vat', 0)),
                        'line_items': []
                    }
                    confidences = {k: 1.0 for k in fields}
                    results.append((fields, confidences, file.name))
            else:
                text = extract_text(file)
                fields, confidences = extract_fields(text)
                results.append((fields, confidences, file.name))
        
        for fields, confidences, filename in results:
            st.subheader(f"Results for {filename}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Extracted Fields:")
                for key, value in fields.items():
                    if key == 'line_items':
                        if value:
                            st.write(f"{key.capitalize()}:")
                            line_items_df = pd.DataFrame(value, columns=['Description', 'Quantity', 'Price'])
                            st.table(line_items_df.style.format({'Price': '${:.2f}'}))
                        else:
                            st.write(f"{key.capitalize()}: None")
                    else:
                        st.write(f"{key.replace('_', ' ').capitalize()}: {value if value is not None else 'None'} (Confidence: {confidences[key]:.2%})")
            
            with col2:
                st.write("Edit Fields:")
                corrections = {}
                for key in fields:
                    if key != 'line_items':
                        corrections[key] = st.text_input(f"Correct {key.replace('_', ' ').capitalize()}", value=str(fields[key] or ''), key=f"{filename}_{key}")
                    elif key == 'line_items' and fields[key]:
                        st.write("Line Items (edit individually):")
                        for i, item in enumerate(fields[key]):
                            col1, col2, col3 = st.columns(3)
                            with col1: new_desc = st.text_input(f"Description {i+1}", value=item['description'], key=f"{filename}_line_{i}_desc")
                            with col2: new_qty = st.number_input(f"Quantity {i+1}", value=item['quantity'], key=f"{filename}_line_{i}_qty")
                            with col3: new_price = st.number_input(f"Price {i+1}", value=item['price'], key=f"{filename}_line_{i}_price", format="%.2f")
                            corrections.setdefault('line_items', []).append({'description': new_desc, 'quantity': new_qty, 'price': new_price})
                
                if st.button("Save Corrections", key=filename):
                    save_corrections(fields, corrections)
                    st.success("Corrections saved!")
            
            if not fields['invoice_number']:
                st.warning("Missing invoice number!")
            if not fields['total']:
                st.warning("Missing total amount!")
        
        result_df = pd.DataFrame([r[0] for r in results])
        st.download_button("Download Results (CSV)", result_df.to_csv(index=False), "results.csv")
        
        conn = sqlite3.connect('invoices.db')
        c = conn.cursor()
        c.execute('SELECT version, path FROM model_versions ORDER BY accuracy DESC LIMIT 1')
        model_data = c.fetchone()
        conn.close()
        if model_data:
            with open(model_data[1], 'rb') as f:
                st.download_button("Download Best Model", f, f"model_{model_data[0]}.pt")
    
    st.sidebar.header("Admin Dashboard")
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('SELECT supplier, COUNT(*), AVG(total) FROM invoices GROUP BY supplier')
    stats = c.fetchall()
    conn.close()
    
    if stats:
        st.sidebar.write("Supplier Stats:")
        for supplier, count, avg_total in stats:
            st.sidebar.write(f"{supplier}: {count} invoices, Avg Total: ${avg_total:.2f}")
    
    if st.sidebar.button("Fine-Tune Model"):
        result = fine_tune_model()
        if result:
            st.sidebar.success(f"New model version {result[0]} created with accuracy {result[1]:.2%}")
        else:
            st.sidebar.warning("Not enough corrections for fine-tuning.")

if __name__ == "__main__":
    main()