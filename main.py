import dspy
import os
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import json
import streamlit as st
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from tools import resume_into_json, company_url
import nltk
from PyPDF2 import PdfReader
import time

gpt4 = dspy.OpenAI(model="gpt-4-0125-preview",temperature=0.2)

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


retriever_model = WeaviateRM("Internship", weaviate_client=weaviate_client)
questions = weaviate_client.collections.get("Internship")

dspy.settings.configure(lm=gpt4,rm=retriever_model)
# Weaviate client configuration
st.title("Internship Finder")
my_bar = st.progress(0)

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
        limit=10
    )

    interns = []

    # adding internships to list
    for item in response.objects:
        interns.append(item.properties) 
    

    context = json.dumps(interns)
    return json.loads(context)

def check_resume(resume):
    if (resume != None):
        pdf_reader = PdfReader(resume)
        text=""
        for page_num in range(len(pdf_reader.pages)):
                
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()
    nltk.download('punkt')  # Ensure the tokenizer is available
    tokens = nltk.word_tokenize(text)
    
    # Check if the total character count of all tokens exceeds the limit
    total_length = sum(len(token) for token in tokens)
    if total_length >= 16000:
        return False  # Return False if the total length of tokens exceeds the limit

    tokens_to_check = ["summary", "skills", "experience", "projects", "education"]
    
    # Convert tokens to lower case for case-insensitive comparison
    text_tokens_lower = [token.lower() for token in tokens]

    # Check if any of the specified tokens are in the tokenized text
    tokens_found = [token for token in tokens_to_check if token.lower() in text_tokens_lower]

    # Return False if none of the specified tokens were found, True otherwise
    return bool(tokens_found)



class Internship_finder(dspy.Module):
    lm = dspy.OpenAI(model='gpt-3.5-turbo-0125', temperature=0.2)

    dspy.settings.configure(lm=lm, rm=weaviate_client)

    def __init__(self):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(generate_query) for _ in range(5)]
        self.generate_analysis = dspy.Predict(generate_analysis,max_tokens=4000) 

    def forward(self, resume):
        #resume to pass as context 
        
        passages = []

        for hop in range(3):
            query = self.generate_query[hop](context=str(resume)).query
            info=search_datbase(query)
            passages.append(info)

        context = deduplicate(passages)  
        my_bar.progress(60,text="Doing Analysis")
            
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


class generate_analysis(dspy.Signature):
    """
    Role: As a matchmaking manager, your task is to connect students with AI engineering internship opportunities that best align with their academic and professional background. Your matchmaking algorithm should analyze and compare the details of student resumes against internship listings to identify the most suitable matches.

    Input:

    Student Resume (Text): Contains details like degree, major, university, relevant coursework, work experience, technical skills, and ML/AI project experience.
    Internship Listings: A collection of AI engineering internship descriptions detailing required skills, experience, and educational requirements.
    Matching Criteria:

    Educational Background: Prioritize matches where the student's education level (undergraduate, graduate, PhD) and major directly align with the internship's requirements. Leverage relevant coursework to strengthen matches related to the internship's focus.
    Skill and Experience Match: Look for a strong correlation between the skills (e.g., programming languages, ML/AI frameworks, tools) required by the internship and those listed on the student's resume. Emphasize matches where specific tools and frameworks are common to both the student's resume and the internship requirements.
    Project Experience: Evaluate the student's project work and internships for their relevance to the internship's technical demands, highlighting projects that showcase the technological competencies sought by the internship.
    Additional Considerations: Exclude internships that emphasize research roles. Instead, focus on opportunities related to "development," "engineering," "application," or "implementation," which are more aligned with practical experiences.

    Output: The system should strictly return a JSON array mentioned below.  Do not include any explanatory text or additional information outside of this JSON structure.
    {
        "name":"",
        "company":"",
        "apply_link":"",
        "match_analysis":""
        }

    If no suitable matches are found, return None.
    """
    
    context = dspy.InputField(desc="Internships")
    resume = dspy.InputField(desc="resume")
    output = dspy.OutputField(desc="list of listings",type=list[JobListing])

class generate_query(dspy.Signature):
    """
    Generate query to search in the weaviate database hybrid search by following below rules:
    1. Analyze the resume, extract keywords from skills, education, experience, projects
    2. then use the keywords to generate query to search in the weaviate database
    3. query should be keyword based to find the best internships for the resume
    """

    context = dspy.InputField(desc="Resume")
    query = dspy.OutputField(desc="query in simple string format")


def main():
    
        
    file = st.file_uploader("Upload Resume to get started", type=["pdf"])
    my_bar.progress(0,text="Starting...") 
    if file is not None:
        msg = st.toast("Resume Uploaded")
        if check_resume(file):
            with st.status("Extracting Details from  Resume"):
                resume = resume_into_json(file)
                st.write(resume)

            analysis = Internship_finder()
            
            my_bar.progress(30,text="Finding Internships")   
            
            generate_analysis = analysis(resume)
            print(generate_analysis)

            if generate_analysis !="None":
                st.subheader("List of Internships:")
                col_company, col_url = st.columns([2,6])
                interns = json.loads(generate_analysis)
                my_bar.progress(100, "Internships Found !!")
                with col_company:
                        for intern in interns:
                            st.link_button(intern["company"],company_url(intern["company"]))
                    
                with col_url:
                        for intern in interns:
                            st.link_button(intern["name"], intern["apply_link"])
                            with st.status("Match Analysis"):
                                st.write(intern["match_analysis"])
            else:
                my_bar.progress(100, "Sorry, No Internships Found for you !!")
                st.write(" We are adding more internships every day, please check back later.")
            
            
        else:
            st.warning("Invalid File Uploaded !!")
            my_bar.progress(0,text="Invalid File Uploaded")


if __name__ == "__main__":
    main()
