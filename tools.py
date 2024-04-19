import json
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import cohere
import requests

with open("resume_temp.json") as f:
    file = json.load(f)

temp = json.dumps(file)

def resume_into_json(resume):
    cohere_api_key = os.getenv("CO_API_KEY")
    co = cohere.Client(cohere_api_key)

    pdf_reader = PdfReader(resume)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    prompt = f"Act as Master in extracting data from resume. Don't give any explanation. Please analyze and convert resume data from this {text} into JSON, remove data like name, email, or personal information, and please return only the JSON file."

    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=5000,
        num_generations=1,
        temperature=0.2,
        
    )

    return json.loads(response.generations[0].text)

def company_url(company):

    if company == "Astranis":
        return "https://www.jeezai.com/companies/astranis-space-technologies"
    
    company = (company.lower()).replace(" ", "-")

    return f"https://www.jeezai.com/companies/{company}/"


def get_company_info(company):
    data = requests.post(
        "https://advanced-research-agents.onrender.com",
        json={
            "query": company,
        }
    )
    return data.json()