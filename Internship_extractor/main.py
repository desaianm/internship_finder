import csv
from dotenv import load_dotenv
from crewai import Crew
from tasks import Tasks
from agents import Agents
import json
import pandas as pd
import time
import os

load_dotenv()
tasks = Tasks()
agents = Agents()

csv_file = 'internships.csv'
df = pd.read_csv(csv_file)

final_output_file = 'final_internships.json'  # Specify the final output JSON file

# Create or open the final output file in append mode
with open(final_output_file, 'a') as final_output:
    for index, row in df.iterrows():
        company_data = row.to_dict()
        company_json = json.dumps(company_data)

        output_json_file = f'output_{index}.json'  # Specify the output JSON file path based on index or other criteria

        with open(output_json_file, 'w') as json_file:
            json.dump(company_data, json_file)

        research_agent = agents.research_agent()
        review = tasks.extract_info(research_agent, company_json,output_json_file)

        crew = Crew(
            agents=[research_agent],
            tasks=[review]
        )

        # Kick off the process
        result = crew.kickoff()

        # Append the content of the output file to the final output file
        with open(output_json_file, 'r') as output_file:
            final_output.write(output_file.read() + ',\n')

        # Delete the last output file
        os.remove(output_json_file)

        time.sleep(5) 