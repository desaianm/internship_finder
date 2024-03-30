import anthropic
import json
from PyPDF2 import PdfReader
from openai import OpenAI
import os


def resume_into_json(resume):
    pdf_reader = PdfReader(resume)
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    client = OpenAI()

    prompt = f" don't give any explantion. please analyze and convert file data from this {text} into  json and in response please return only json file please don't enter data in fields if irrelevant to template"

    response = client.chat.completions.create(
            model="gpt-4-0125-preview",
              response_format={ "type": "json_object" },  # Adjust the model identifier as needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ],temperature=0.4
    )
        
    return json.loads(response.choices[0].message.content)

def company_url(company):

    if company == "Astranis":
        return "https://www.jeezai.com/companies/astranis-space-technologies"
    
    company = (company.lower()).replace(" ", "-")

    return f"https://www.jeezai.com/companies/{company}/"
