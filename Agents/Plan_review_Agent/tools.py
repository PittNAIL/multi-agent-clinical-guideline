# Agents/Plan_review_Agent/tools.py
import json
import docx
import os

def fetch_guideline_for_diagnosis(diagnosis: str) -> str:
    """
    Fetches the NCCN guideline information for a specific diagnosis
    by reading a .docx file located in the same directory.
    
    Args:
        diagnosis: The cancer diagnosis (e.g., "osteosarcoma"). Case-insensitive.
        
    Returns:
        A string containing the relevant guideline text, or an error message.
    """
    try:
        # Construct the path to the DOCX file relative to this script's location
        script_dir = os.path.dirname(__file__)
        doc_path = os.path.join(script_dir, 'Human Altered Decision Trees.docx')
        
        if not os.path.exists(doc_path):
            return json.dumps({"error": f"Guideline document not found at {doc_path}"})
            
        document = docx.Document(doc_path)
        
        guideline_text = []
        found_diagnosis = False
        
        # We'll search for the diagnosis as a heading.
        # Once found, we'll capture all subsequent paragraphs until we hit
        # another heading or the end of the document.
        
        # Normalize the search term
        search_term = diagnosis.lower().strip()
        
        for para in document.paragraphs:
            # Check if we've found the relevant section
            if found_diagnosis:
                # If we hit another heading, we stop. This assumes headings are used for new sections.
                # A simple heuristic for a heading is bold text or a specific style.
                # We'll check if the text is bold.
                if para.runs and para.runs[0].bold:
                    break
                guideline_text.append(para.text)
            
            # Check if the paragraph is the start of the section we want
            elif search_term in para.text.lower():
                # We assume the diagnosis is mentioned in a heading-like paragraph
                if para.runs and para.runs[0].bold:
                    found_diagnosis = True
                    # We can include the heading itself if we want
                    guideline_text.append(para.text)

        if not guideline_text:
            return json.dumps({"error": f"No guideline section found for diagnosis '{diagnosis}' in the document."})
            
        # Join the collected paragraphs into a single text block
        return "\n".join(guideline_text)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to read or parse the guideline document: {str(e)}"})
