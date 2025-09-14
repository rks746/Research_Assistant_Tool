from transformers import pipeline

# Load summarization pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    # Split the text by empty lines, filter out empty strings
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    summary_text = ""
    
    for chunk in parts:
        input_len = len(chunk.split())
        
        # If the chunk is too small, skip summarization and just append it
        if input_len < 30:
            summary_text += chunk + "\n\n"
            continue
        
        # Adjust max_length dynamically
        max_len = min(150, max(30, input_len // 2))
        
        # Run summarization
        summary = summarizer(chunk, max_length=max_len, min_length=15, do_sample=False)[0]['summary_text']
        
        summary_text += summary + "\n\n"
    
    return summary_text.strip()

