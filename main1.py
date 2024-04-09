import dspy
import os
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import json
import streamlit as st
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from tools import resume_into_json, company_url
import nltk
from PyPDF2 import PdfReader
import time
from internship_Finder import Internship_finder

def check_resume(resume):
    if resume is not None:
        pdf_reader = PdfReader(resume)
        text = ""
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
    else:
        return False

def main():
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

    retriever_model = WeaviateRM("Internship", weaviate_client=weaviate_client)
    questions = weaviate_client.collections.get("Internship")

    dspy.settings.configure(lm=gpt4, rm=retriever_model)

    st.title("Internship Finder")
    my_bar = st.progress(0)

    file = st.file_uploader("Upload Resume to get started", type=["pdf"])
    my_bar.progress(0, text="Starting...") 
    if file is not None:
        msg = st.toast("Resume Uploaded")
        if check_resume(file):
            with st.status("Extracting Details from  Resume"):
                resume = resume_into_json(file)
                st.write(resume)

            internship_finder = Internship_finder(my_bar)
            
            my_bar.progress(30, text="Finding Internships")   
            
            generate_analysis = internship_finder(resume)

            st.subheader("List of Internships :")

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

            
            if interns is None:
                msg.toast("No Internships Found")    
            msg.toast("Internships Found !!")
        else:
            st.warning("Invalid File Uploaded !!")
            my_bar.progress(0, text="Invalid File Uploaded")

if __name__ == "__main__":
    main()