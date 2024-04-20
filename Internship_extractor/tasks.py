from textwrap import dedent
from crewai import Task

class Tasks():
    def extract_info(self, agent, json_file, output_file):
        return Task(
            description=dedent(f"""\
                    Extracts key information of each company in order such as company,apply_link,date_published,location,country,city,state,roles,skills,eligibility,degree,field,experience,summary,etc from the internships.csv file {json_file} resumes and make a overall summary of the job posting .
                    Please also include most important skill(not basic skills) needed for the job .and give summary a little longer. Also don't write [Here] in apply_link.
                    In experience field just give important experience need for the job and don't make it longer
                    like """),
            expected_output=dedent("""\
                    Give output like strictly like this json file like {
    "name": "",
    "company": "",
    "apply_link": "",
    "date_published": "",
    "country": "",
    "city": "",
    "skills": [],
    "degree": "",
    "field": [],
    "experience": [],
    "summary": ""
}
Please give summary a little longer of about 60 words"""),
            agent=agent,
            output_file=output_file  # Pass the output file path to the Task
        )