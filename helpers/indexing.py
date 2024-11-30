import os
import json
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data files
nltk.download('punkt')

def extract_docs(content):
    """Extract documents based on <DOCNO> tags and exclude <DOCOLDNO> and <DOCHDR> tags."""
    docs = []
    doc_pattern = re.compile(r'<DOC>(.*?)</DOC>', re.DOTALL)
    docno_pattern = re.compile(r'<DOCNO>(.*?)</DOCNO>', re.DOTALL)
    exclude_pattern = re.compile(r'<DOCOLDNO>.*?</DOCOLDNO>|<DOCHDR>.*?</DOCHDR>', re.DOTALL)
    content_pattern = re.compile(r'>([^<]+)<')

    for doc in doc_pattern.findall(content):
        docno_match = docno_pattern.search(doc)
        if docno_match:
            docno = docno_match.group(1).strip()
            doc = exclude_pattern.sub('', doc)  # Remove <DOCOLDNO> and <DOCHDR> tags
            text = '\n'.join(content_pattern.findall(doc))
            text = re.sub(r'\n+', '\n', text)
            docs.append({'id': docno, 'contents': text})

    return docs

def stem_text(text):
    """Perform stemming on the text."""
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

def create_original_jsonl(input_folder, output_file):
    """Convert corpus into jsonl with original content."""

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for root, _, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        docs = extract_docs(content)
                        for doc in docs:
                            jsonl_file.write(json.dumps(doc) + '\n')
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

def create_stemmed_jsonl(original_jsonl_file, stemmed_output_file):
    """Convert corpus into jsonl with stemmed content."""

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(stemmed_output_file), exist_ok=True)

    with open(original_jsonl_file, 'r', encoding='utf-8') as original_file, \
         open(stemmed_output_file, 'w', encoding='utf-8') as stemmed_jsonl_file:
        for line in original_file:
            try:
                json_record = json.loads(line)
                text = json_record.get('contents', '')
                stemmed_text = stem_text(text)
                stemmed_json_record = {
                    'id': json_record.get('id', ''),
                    'contents': stemmed_text
                }
                stemmed_jsonl_file.write(json.dumps(stemmed_json_record) + '\n')
                print(f"Finished writing id: {json_record.get('id', '')}") # logging
            except Exception as e:
                print(f"Error processing line: {e}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_folder = os.path.join(project_dir, 'WT2G')
    original_jsonl_file = os.path.join(project_dir, 'data/collections/collections.jsonl')
    stemmed_output_file = os.path.join(project_dir, 'data/collections_stemmed/collections_stemmed.jsonl')

    if os.path.exists(input_folder):
        # create_original_jsonl(input_folder, original_jsonl_file)
        create_stemmed_jsonl(original_jsonl_file, stemmed_output_file)
        print("Processing completed.")
    else:
        print(f'{input_folder} does not exist.')