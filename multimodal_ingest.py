import os
import shutil
import pickle
import faiss
import pdfplumber
import pytesseract
import re
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------
# TABLE → METRIC-LEVEL CHUNKS (AUTHORITATIVE DATA)
# ------------------------------------------------
def chunk_table_metrics(table, page_num):
    chunks = []

    if not table or len(table) < 2:
        return chunks

    headers = [h.strip() if h else "" for h in table[0]]

    # Detect year column
    year_col = None
    for i, h in enumerate(headers):
        if "year" in h.lower() or h.strip().isdigit():
            year_col = i
            break

    for row in table[1:]:
        if not row:
            continue

        row = [c.strip() if c else "" for c in row]

        year = None
        if year_col is not None and year_col < len(row):
            year = row[year_col]

        for col_idx, (header, value) in enumerate(zip(headers, row)):
            if not header or not value or col_idx == year_col:
                continue

            if value.lower() in ["", "-", "n/a", "na", "null"]:
                continue

            h = header.lower()

            # ---- UNIT INFERENCE (CRITICAL ORDER) ----
            if "growth" in h or "percent" in h or "%" in header or "change" in h:
                units = "%"
            elif "inflation" in h or "unemployment" in h:
                units = "%"
            elif "gdp" in h or "revenue" in h or "expenditure" in h:
                units = "billion QAR"
            elif "debt" in h or "deficit" in h or "surplus" in h:
                units = "billion QAR"
            else:
                units = ""

            metric = header.replace("(%)", "").replace("(bn)", "").replace("(billion)", "").strip()

            if year:
                sentence = f"In {year}, Qatar's {metric} was {value}"
            else:
                sentence = f"Qatar's {metric} was {value}"

            if units:
                sentence += f" {units}"

            sentence += f" (Page {page_num}, Table)"

            chunks.append({
                "content": sentence,
                "page": page_num,
                "type": "table"
            })

    return chunks


# ------------------------------------------------
# REMOVE NUMERIC TEXT WHEN TABLE EXISTS
# ------------------------------------------------
def is_numeric_summary(text):
    return bool(re.search(r"\b\d+(\.\d+)?\s*(percent|%|billion|qar|usd)\b", text.lower()))


# ------------------------------------------------
# PDF EXTRACTION
# ------------------------------------------------
def extract_from_pdf(pdf_path):
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            page_has_tables = False

            # ---- TABLES FIRST (AUTHORITATIVE) ----
            try:
                tables = page.extract_tables()
                for table in tables:
                    metric_chunks = chunk_table_metrics(table, page_num)
                    if metric_chunks:
                        page_has_tables = True
                        chunks.extend(metric_chunks)
            except:
                pass

            # ---- TEXT (ONLY NARRATIVE, NOT NUMERIC) ----
            try:
                text = page.extract_text()
                if text and text.strip():
                    if not (page_has_tables and is_numeric_summary(text)):
                        chunks.append({
                            "content": text,
                            "page": page_num,
                            "type": "text"
                        })
            except:
                pass

            # ---- OCR ----
            try:
                for img in page.images:
                    try:
                        cropped = page.crop((img["x0"], img["top"], img["x1"], img["bottom"]))
                        image = cropped.to_image(resolution=300).original
                        ocr = pytesseract.image_to_string(image)

                        if ocr and ocr.strip():
                            chunks.append({
                                "content": ocr,
                                "page": page_num,
                                "type": "image"
                            })
                    except:
                        pass
            except:
                pass

    return chunks


# ------------------------------------------------
# INGESTION PIPELINE
# ------------------------------------------------
def ingest_multimodal():
    print("\nStarting IMF-grade multimodal ingestion...")

    if not os.path.exists("uploads"):
        print("No uploads folder.")
        return 0

    if os.path.exists("db"):
        shutil.rmtree("db")
    os.makedirs("db", exist_ok=True)

    all_chunks = []

    for file in os.listdir("uploads"):
        if file.lower().endswith(".pdf"):
            print("Processing:", file)
            pdf_path = os.path.join("uploads", file)
            extracted = extract_from_pdf(pdf_path)

            for c in extracted:
                c["source"] = file

            all_chunks.extend(extracted)

    if not all_chunks:
        print("No content extracted.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    final_chunks = []
    metadata = []

    for chunk in all_chunks:
        if chunk["type"] == "table":
            final_chunks.append(chunk["content"])
            metadata.append({
                "page": chunk["page"],
                "type": "table",
                "source": chunk["source"]
            })
        else:
            splits = splitter.split_text(chunk["content"])
            for s in splits:
                final_chunks.append(s)
                metadata.append({
                    "page": chunk["page"],
                    "type": chunk["type"],
                    "source": chunk["source"]
                })

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(final_chunks, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("db/chunks.pkl", "wb") as f:
        pickle.dump(final_chunks, f)

    with open("db/meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    faiss.write_index(index, "db/index.faiss")

    shutil.rmtree("uploads")

    print("\n✓ Multimodal index built:", len(final_chunks), "chunks")
    return len(final_chunks)


if __name__ == "__main__":
    ingest_multimodal()