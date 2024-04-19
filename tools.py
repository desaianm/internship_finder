import anthropic
import json
from PyPDF2 import PdfReader
from openai import OpenAI
import os


def resume_into_json(resume):
    api_key = os.getenv("CLAUDE_API_KEY")
    client = anthropic.Anthropic(
        api_key=api_key,
    )


    pdf_reader = PdfReader(resume)
    text=""
    for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    prompt = f" Act as Master in extracting data from resume.don't give any explantion. please analyze and convert resume data from this {text} into  json, remove data like name, email or personal information and  please return only json file  "

    response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
            ]
        ).content[0].text
    
    return json.loads(response)

def company_url(company):

    if company == "Astranis":
        return "https://www.jeezai.com/companies/astranis-space-technologies"
    
    company = (company.lower()).replace(" ", "-")

    return f"https://www.jeezai.com/companies/{company}/"
