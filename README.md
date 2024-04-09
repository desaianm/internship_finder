# Resume Finder

This is a web application designed to help users find internship opportunities based on their resumes. The application leverages the power of OpenAI's Dspy library and Weaviate for natural language processing and semantic search capabilities.

## Features

- **Resume Upload**: Users can upload their resumes in PDF format.
- **Resume Analysis**: The application extracts information from the uploaded resume, including educational background, work experience, technical skills, and projects.
- **Internship Matching**: Using the extracted resume information, the application searches for relevant internship opportunities.
- **Interactive Interface**: The web interface allows users to interactively view and explore matched internship listings.

## Technologies Used

- [Streamlit](https://streamlit.io/): Python library for building interactive web applications.
- [OpenAI Dspy](https://openai.com/dspy): Natural language processing library for text generation and analysis.
- [Weaviate](https://www.semi.technology/): Semantic vector search engine for performing advanced searches.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/resume-finder.git
    cd resume-finder
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up API keys:**

    - Obtain an API key for OpenAI and Weaviate and replace them in the appropriate places in the code.

4. **Run the application:**

    ```bash
    streamlit run main.py
    ```

## Usage

1. Upload your resume in PDF format.
2. Wait for the application to analyze your resume and display relevant internship opportunities.
3. Explore the internship listings and click on links to view more details or apply.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests with improvements or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
