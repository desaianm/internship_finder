from crewai import Agent
from crewai_tools import WebsiteSearchTool, FileReadTool, CSVSearchTool

from crewai_tools import BaseTool

# class CSVReaderTool(BaseTool):
#     name: str = "CSV Reader Tool"
#     description: str = "A tool for reading data from a CSV file row by row."

#     def _run(self, csv_file_path: str) -> str:
#         try:
#             # Implementation to read CSV file row by row with explicit encoding
#             with open(csv_file_path, 'r', encoding='utf-8') as file:
#                 for line in file:
#                     # Process each row as needed
#                     print(line)
#             return "CSV file read successfully"
#         except UnicodeDecodeError as e:
#             return f"Error reading CSV file: {e}"
    
# Define tools
web_search_tool = WebsiteSearchTool()
file_read_tool = FileReadTool(
    file_path='input.json',
    description='A tool to read the internship job description file.'
)
# csv_reader_tool = CSVReaderTool(file_path='internships.csv')
class Agents:
    def research_agent(self):
        return Agent(
            role='Research Analyst',
            goal='Analyze the internship details and provided descriptions to extract a complete summary of the company job/internship posting.',
            tools=[web_search_tool, file_read_tool],
            backstory='Expert in analyzing internship descriptions and identifying key values and needs from various sources.',
            verbose=True
        )

    def writer_agent(self):
        return Agent(
            role='Job Description Writer',
            goal='Use insights from the Research Analyst to create a detailed, engaging, and enticing internship posting.',
            tools=[web_search_tool, file_read_tool],
            backstory="Skilled in crafting compelling internship descriptions that attract the right candidates.",
            verbose=True
        )

