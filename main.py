import dspy
from dspy import dsp
import os
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import json
import streamlit as st
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from tools import company_url, resume_into_json
import nltk
from PyPDF2 import PdfReader
import cohere

co_api_key = os.getenv("CO_API_KEY")
nltk.download('punkt')

# Weaviate client configuration
url = "https://internship-finder-52en6hka.weaviate.network"
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

cohere = dsp.Cohere(model='command-r-plus',api_key=co_api_key)

retriever_model = WeaviateRM("Internship", weaviate_client=weaviate_client)

dspy.settings.configure(lm=cohere,rm=retriever_model)
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
    url = "https://internship-finder-52en6hka.weaviate.network"
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

    response = questions.query.hybrid(
        query=query,
        limit=10
    )

    interns = []

    # adding internships to list
    for item in response.objects:
        interns.append(item.properties) 
    
    
    context = json.dumps(interns)
    weaviate_client.close()
    return json.loads(context)

def check_resume(resume):
    if (resume != None):
        pdf_reader = PdfReader(resume)
        text=""
        for page_num in range(len(pdf_reader.pages)):
                
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()
    
    
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
    cohere = dsp.Cohere(model='command-r-plus',api_key=co_api_key)

    dspy.settings.configure(lm=cohere)
    def __init__(self):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(generate_query) for _ in range(3)]
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
    Your Role:


    You are a Matchmaking Manager, an expert at connecting students with their ideal internship opportunities.


    Input:


    You will be provided with a student's resume and a list of potential internship opportunities. Your task is to carefully analyze and match the student's credentials with the requirements of each internship, following the specific criteria outlined below.


    Matching Criteria:


    Educational Background:


    Degree Level and Major: Seek exact matches or close alignments between the student's degree level (bachelor's, master's, etc.) and major with the educational requirements specified in the internships.
    Related Fields of Study: Consider closely related fields of study as a potential match. For example, a student majoring in Computer Science could be a good fit for internships seeking IT or Software Engineering majors.
    Relevant Coursework: Give bonus points to internships that specifically mention or prefer certain courses that the student has completed. For example, if an internship seeks candidates with a background in Data Structures and the student has taken an advanced course in that area, it strengthens the match.

    Skill and Experience Match:


    Required Skills: Look for strong overlaps between the technical skills listed on the student's resume and the required skills outlined in the internship descriptions.
    Tools and Frameworks: Prioritize internships that specifically mention tools, programming languages, or frameworks that the student has hands-on experience with. For example, if an internship seeks proficiency in Python, and the student has worked on Python projects, it is a strong match.
    Applied Skills: Value projects or previous work experiences that demonstrate the practical application of the required skills. For instance, if an internship seeks candidates with web development skills, and the student has built and deployed websites, it is a clear indication of a good fit.

    Project Relevance:


    Project Experience: Analyze the student's project portfolio to identify technical skills and areas of expertise that align with the internships' requirements.
    AI/ML and Data Focus: Match internships that specifically seek experience or interest in AI/ML model development, data analysis, or similar areas. Look for keywords like "machine learning," "data engineering," or "data-driven solutions" in the internship descriptions.
    Ensure that the internships do not include "research"  in their titles, skills, or descriptions.
    Practical Implementation: Prioritize internships that emphasize hands-on experience in development, engineering, application development, or implementation roles over theoretical or research-focused roles.
    For Match Analysis: do a detailed match analysis for each internship, highlighting the key points of alignment between the student's profile and the internship requirements. Provide a brief summary of the match analysis for each internship.
    
    Output Format:

    Strictly follow the output format as described below:
    keep max tokens 4000
    strictly just provide a JSON array with the top-matched internships nothing else, following this format :
    
    {
    "name": "",
    "company": "",
    "apply_link": "",
    "match_analysis":""
    }
    No Matches: If no internships are a good fit, return None.


    """
    
    context = dspy.InputField(desc="Internships")
    resume = dspy.InputField(desc="resume")
    output = dspy.OutputField(desc="list of internships",type=list[JobListing])

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

            if generate_analysis != "None":
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
