# AI Data Analysis Agent

This project is a simple AI-powered data analysis tool designed for quick insights on CSV or Excel files. You provide the data, and the agent handles the rest.

## What It Does

- Accepts data in **CSV** or **Excel** format  
- Generates a clear summary of the dataset  
- Identifies patterns, trends and basic statistical insights  
- Suggests sample questions you can ask about the data  
- Allows you to ask your own custom questions  
- Can generate **SQL queries** based on the contents of your file


## Example Questions You Can Ask

- “What are the most important columns in this dataset?”  
- “Are there any correlations?”  
- “Give me a summary of the top categories.”  
- “Write an SQL query to filter records based on a condition.”

## Goal

The goal of this project is to provide a minimal, beginner-friendly AI agent that makes data exploration easier without needing complex setup or advanced machine-learning knowledge.


### How to get Started?

1. Clone the GitHub repository

```bash
git clone https://github.com/parshvak26/AI-Agent-for-data-analysis.git
cd Basic-AI-Agent-for-data-analysis
```
2. Install the required dependencies

```bash
pip install -r requirements.txt
```
3. Get your OpenAI API Key

- Sign up for an [Google gemini account] and obtain your API key, and add it in code below Import

4. Run this command
```CMD
python DataAnalysisAgent.py --file YOUR_FILE_HERE.CSV
```
