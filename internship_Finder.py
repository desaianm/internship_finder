import dspy
import json
import nltk
from PyPDF2 import PdfReader
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
import weaviate
from typing import Optional, List
import os
import streamlit as st
from generate_query import GenerateQuery
from generate_analysis import GenerateAnalysis


gpt4 = dspy.OpenAI(model="gpt-4-0125-preview")

url = "https://internships-hc3oiv0y.weaviate.network"
apikey = os.getenv("WCS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Connect to Weaviate
weaviate_client = weaviate.connect_to_wcs(
    cluster_url=url,  
    auth_credentials=weaviate.auth.AuthApiKey(apikey),
        headers={
        "X-OpenAI-Api-Key": openai_api_key  
    }  
    
)

questions = weaviate_client.collections.get("Internship")
class JobListing(BaseModel):
    city: str
    date_published: datetime  # Assuming the date can be parsed into a datetime object
    apply_link: HttpUrl  # Pydantic will validate this is a valid URL
    company: str
    location: Optional[str]  # Assuming 'location' could be a string or None
    country: str
    name: str

class Out_Internship(BaseModel):
    output: list[JobListing] = Field(description="list of internships")  

def search_datbase(query):
    response = questions.query.hybrid(
        query=query,
        limit=7
    )

    interns = []

    # adding internships to list
    for item in response.objects:
        interns.append(item.properties) 
    

    context = json.dumps(interns)
    return json.loads(context)



class Internship_finder(dspy.Module):
    lm = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.3)

    dspy.settings.configure(lm=lm, rm=weaviate_client)

    def __init__(self, my_bar):
        super().__init__()
        self.my_bar = my_bar
        self.generate_query = [dspy.ChainOfThought(GenerateQuery) for _ in range(3)]
        self.generate_analysis = dspy.Predict(GenerateAnalysis, max_tokens=4000)

    def forward(self, resume):
        #resume to pass as context 
        
        passages = []

        for hop in range(3):
            query = self.generate_query[hop](context=str(resume)).query
            info=search_datbase(query)
            passages.append(info)

        context = deduplicate(passages)    
        context.append(resume)
        self.my_bar.progress(60,text="Generating Analysis")
            
        analysis = self.generate_analysis(resume=str(resume), context=context).output
              
        return analysis

def deduplicate(context):
        """
        Removes duplicate elements from the context list while preserving the order.
        
        Parameters:
        context (list): List containing context elements.
        
        Returns:
        list: List with duplicates removed.
        """
        json_strings = [json.dumps(d, sort_keys=True) for d in context]
    
        # Use a set to remove duplicate JSON strings
        unique_json_strings = set(json_strings)
    
        # Convert JSON strings back to dictionaries
        unique_dicts = [json.loads(s) for s in unique_json_strings]
        return unique_dicts

def check_answer(assessment_answer):
    if assessment_answer == "no":
        return False
    return True

def get_resume():
    with open('resume.json', 'r') as file: 
        resume = json.load(file)
     
    return resume

