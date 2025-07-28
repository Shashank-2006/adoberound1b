# 📄 Intelligent Document Analysis System

This project extracts, analyzes, and ranks the most relevant sections from a collection of PDFs based on a **persona** and a **job-to-be-done** using semantic similarity and sentence embeddings.

---

## ✅ Features

- Extracts clean, structured sections from scanned and text PDFs
- Uses [Sentence Transformers](https://www.sbert.net/) for ranking based on persona context
- Outputs ranked section titles and refined summaries
- Dockerized and easy to deploy

---

## 🔍 Approach

1. **PDF Parsing**: Each page is parsed using `pdfplumber`. Sections are detected based on layout heuristics and headers.
2. **Persona Query Construction**: Combines role, expertise, focus, and task into a composite semantic query.
3. **Embedding and Ranking**: Uses `sentence-transformers` (MiniLM model) to compute similarity between query and document chunks.
4. **Ranking**: Sections are ranked with content-type-based weighting (e.g., abstract/methodology/conclusion gets higher weight).
5. **Output JSON**: Final output includes metadata, top section titles with page numbers, and refined content samples.

---

## 🧠 Model Used

- `sentence-transformers/all-MiniLM-L6-v2`
    - Lightweight and fast
    - Good balance of accuracy and inference speed

---

## 🧪 Sample Input

`challenge1b_input.json`

```json
{
  "documents": ["file1.pdf", "file2.pdf"],
  "persona": {
    "role": "Curriculum Designer",
    "expertise_areas": ["AI", "Education Technology"],
    "focus_areas": ["Learning outcomes", "Skill development"]
  },
  "job_to_be_done": {
    "task": "Design a new AI curriculum for undergraduates",
    "requirements": ["Focus on hands-on skills", "Use latest industry trends"]
  }
}

## HOW TO RUN

