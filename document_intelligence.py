import os
import json
import datetime
import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIntelligenceSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10):
        self.model_name = model_name
        self.top_k = top_k
        self.model = None

    def load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def extract_text_chunks(self, pdf_path: str) -> List[Dict]:
        chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document_name = os.path.basename(pdf_path)
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    sections = self._extract_sections_from_page(text)
                    for section in sections:
                        if self._is_meaningful_section(section['text']):
                            chunks.append({
                                'document': document_name,
                                'text': section['text'],
                                'section_title': section['title'],
                                'page': page_num,
                                'content_type': self._classify_content_type(section['text'])
                            })
        except Exception as e:
            logger.error(f"Failed to process PDF '{pdf_path}': {e}")
        return chunks

    def _extract_sections_from_page(self, text: str) -> List[Dict]:
        sections = []
        lines = text.split('\n')
        current = {'title': '', 'text': ''}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self._is_section_header(line):
                if current['text'].strip():
                    sections.append({
                        'title': current['title'] or line[:100],
                        'text': current['text'].strip()
                    })
                current = {'title': line, 'text': ''}
            else:
                current['text'] += line + ' '
        if current['text'].strip():
            sections.append({
                'title': current['title'] or 'Content',
                'text': current['text'].strip()
            })
        if not sections and text.strip():
            title = self._extract_meaningful_title(text)
            sections.append({
                'title': title,
                'text': text.strip()
            })
        return sections

    def _is_section_header(self, line: str) -> bool:
        if len(line) < 3 or len(line) > 150:
            return False
        patterns = [
            r'^[A-Z][A-Za-z\s&-]+$',
            r'^\d+\.\s*[A-Z]',
            r'^[IVX]+\.\s*[A-Z]',
            r'^[A-Z\s]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+',
        ]
        for pattern in patterns:
            if re.match(pattern, line):
                if line.isupper() and len(line) > 50:
                    continue
                if line.count('.') > 3:
                    continue
                return True
        return False

    def _extract_meaningful_title(self, text: str) -> str:
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if 20 <= len(sentence) <= 100:
                return sentence
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 10 <= len(line) <= 100:
                return line
        words = text.split()[:10]
        return ' '.join(words) + ('...' if len(text.split()) > 10 else '')

    def _is_meaningful_section(self, text: str) -> bool:
        noise = [
            len(text.split()) < 10,
            (text.count('\n') / len(text) > 0.1) if len(text) > 0 else False,
            (len(re.findall(r'\d', text)) / len(text) > 0.3) if len(text) > 0 else False,
        ]
        return not any(noise)

    def _classify_content_type(self, text: str) -> str:
        text = text.lower()
        if 'abstract' in text or 'summary' in text:
            return 'abstract'
        elif 'introduction' in text or 'background' in text:
            return 'introduction'
        elif 'method' in text or 'approach' in text:
            return 'methodology'
        elif 'result' in text or 'finding' in text:
            return 'results'
        elif 'conclusion' in text or 'discussion' in text:
            return 'conclusion'
        elif 'reference' in text or 'bibliography' in text:
            return 'references'
        else:
            return 'content'

    def create_persona_query(self, persona: Dict, job: Dict) -> str:
        role = persona.get('role', '')
        expertise = persona.get('expertise_areas', [])
        focus = persona.get('focus_areas', [])
        task = job.get('task', '')
        requirements = job.get('requirements', [])
        query_parts = [task]
        if expertise:
            query_parts.append(f"Expertise in: {', '.join(expertise)}")
        if focus:
            query_parts.append(f"Focus on: {', '.join(focus)}")
        if requirements:
            query_parts.append(f"Requirements: {', '.join(requirements)}")
        query_parts.append(f"Role: {role}")
        return " | ".join(query_parts)

    def rank_sections(self, sections: List[Dict], query: str) -> List[Dict]:
        if not sections or not self.model:
            return []
        texts = [s['text'] for s in sections]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        weighted_scores = []
        for i, (score, section) in enumerate(zip(similarities, sections)):
            weight = {
                'abstract': 1.2,
                'introduction': 1.1,
                'methodology': 1.3,
                'results': 1.3,
                'conclusion': 1.2,
                'references': 0.5,
                'content': 1.0
            }.get(section.get('content_type', 'content'), 1.0)
            weighted_scores.append((score * weight, i))
        weighted_scores.sort(reverse=True, key=lambda x: x[0])
        ranked = []
        for rank, (_, idx) in enumerate(weighted_scores[:self.top_k]):
            section = sections[idx].copy()
            section['importance_rank'] = rank + 1
            section['relevance_score'] = float(similarities[idx])
            ranked.append(section)
        return ranked

    def process_documents(self, input_data: Dict) -> Dict:
        start_time = datetime.datetime.now()
        documents = input_data.get('documents', [])
        persona = input_data.get('persona', {})
        job = input_data.get('job_to_be_done', {})
        pdf_dir = input_data.get('pdf_directory', 'pdfs')
        if not self.model:
            self.load_model()
        query = self.create_persona_query(persona, job)
        all_sections = []
        processed_docs = []
        for doc in documents:
            filename = doc.get('filename', doc) if isinstance(doc, dict) else doc
            pdf_path = os.path.join(pdf_dir, filename)
            if os.path.exists(pdf_path):
                sections = self.extract_text_chunks(pdf_path)
                all_sections.extend(sections)
                processed_docs.append(filename)
            else:
                logger.warning(f"File not found and skipped: {pdf_path}")
        top_sections = self.rank_sections(all_sections, query)
        output = {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona.get('role', str(persona)),
                "job_to_be_done": job.get('task', str(job)),
                "processing_timestamp": start_time.isoformat()
            },
            "extracted_sections": [
                {
                    "document": s["document"],
                    "section_title": s["section_title"],
                    "importance_rank": s["importance_rank"],
                    "page_number": s["page"]
                } for s in top_sections
            ],
            "subsection_analysis": [
                {
                    "document": s["document"],
                    "refined_text": s["text"][:500] + "..." if len(s["text"]) > 500 else s["text"],
                    "page_number": s["page"]
                } for s in top_sections
            ]
        }
        return output

def main():
    INPUT_JSON = "challenge1b_input.json"
    OUTPUT_JSON = "challenge1b_output.json"
    PDF_DIR = "pdfs"
    try:
        with open(INPUT_JSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file '{INPUT_JSON}' not found.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from '{INPUT_JSON}': {e}")
        return
    input_data['pdf_directory'] = PDF_DIR
    system = DocumentIntelligenceSystem()
    result = system.process_documents(input_data)
    try:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write output to '{OUTPUT_JSON}': {e}")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
