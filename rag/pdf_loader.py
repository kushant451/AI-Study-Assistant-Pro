from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def extract_text_from_multiple(uploaded_files):
    documents = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        documents.append({"filename": file.name, "text": text})
    return documents