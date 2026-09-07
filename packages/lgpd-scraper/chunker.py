import json
import os
import re
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INPUT_FILE = "data/raw/lgpd_raw.html"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lgpd_chunks.json")

def clean_text(text: str) -> str:
    """Removes extra whitespaces and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks(html_content: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all(['p', 'span'])
    
    all_chunks = []
    current_article = None
    child_counter = 0
    
    for p in paragraphs:
        text = clean_text(p.get_text())
        if not text:
            continue
            
        if text.startswith("Art."):
            parts = text.split(" ", 2)
            art_num = parts[1] if len(parts) > 1 else "desconhecido"
            
            current_article = {
                "id": f"artigo_{art_num}",
                "artigo": art_num,
                "texto_completo": text, # Vai acumular todos os filhos
                "texto_chunk": text,    # Apenas o texto do caput
                "tipo": "parent"
            }
            all_chunks.append(current_article)
            child_counter = 0
            
        elif current_article:
            is_child = text.startswith("§") or text.startswith("Parágrafo único") or re.match(r'^[IXV]+\s*-', text)
            
            if is_child:
                child_counter += 1
                child_id = f"{current_article['id']}_child_{child_counter}"
                
                child_chunk = {
                    "id": child_id,
                    "artigo": current_article["artigo"],
                    "parent_id": current_article["id"],
                    "texto_chunk": text,
                    "tipo": "child"
                }
                all_chunks.append(child_chunk)
                current_article["texto_completo"] += f" {text}"

    # Atualizamos todos os filhos com o texto completo do parent (essencial para Retrieval)
    for chunk in all_chunks:
        if chunk["tipo"] == "child":
            parent = next((p for p in all_chunks if p["id"] == chunk["parent_id"]), None)
            if parent:
                chunk["parent_texto_completo"] = parent["texto_completo"]

    return all_chunks

def run_chunker() -> None:
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file {INPUT_FILE} not found. Run collect.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()

    logging.info("Parsing LGPD HTML...")
    chunks = extract_chunks(html_content)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Successfully processed {len(chunks)} articles. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_chunker()
