"""
Semantic Markdown Chunker
Segments markdown documents by section headers and questions into coherent chunks.
"""

import re
from typing import List, Dict, Any

def chunk_markdown_documents(documents: List[Dict[str, Any]], max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Chunks markdown documents by header levels (## and ###) to preserve semantic coherence.
    Each chunk contains:
      - chunk_id: str
      - title: str
      - section: str
      - text: str
      - source: str
      - category: str
    """
    chunks = []
    chunk_counter = 0

    for doc in documents:
        content = doc["content"]
        source = doc["relative_path"]
        category = doc["category"]
        doc_title = doc["title"]

        # Split by ## or ### headers
        sections = re.split(r'\n(?=#{1,3}\s+)', content)

        for sec in sections:
            sec_text = sec.strip()
            if not sec_text:
                continue

            # Extract header line if present
            header_match = re.match(r'^(#{1,3})\s+(.+)$', sec_text, re.MULTILINE)
            section_title = header_match.group(2) if header_match else doc_title

            # If section is excessively long, split by paragraphs
            if len(sec_text) > max_chunk_size:
                paragraphs = sec_text.split("\n\n")
                current_block = []
                current_len = 0

                for p in paragraphs:
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    if current_len + len(p_clean) > max_chunk_size and current_block:
                        chunk_text = f"[{doc_title} - {section_title}]\n" + "\n\n".join(current_block)
                        chunk_counter += 1
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_counter}",
                            "title": doc_title,
                            "section": section_title,
                            "text": chunk_text,
                            "source": source,
                            "category": category
                        })
                        current_block = [p_clean]
                        current_len = len(p_clean)
                    else:
                        current_block.append(p_clean)
                        current_len += len(p_clean)

                if current_block:
                    chunk_text = f"[{doc_title} - {section_title}]\n" + "\n\n".join(current_block)
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_counter}",
                        "title": doc_title,
                        "section": section_title,
                        "text": chunk_text,
                        "source": source,
                        "category": category
                    })
            else:
                chunk_text = f"[{doc_title} - {section_title}]\n{sec_text}"
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter}",
                    "title": doc_title,
                    "section": section_title,
                    "text": chunk_text,
                    "source": source,
                    "category": category
                })

    return chunks
