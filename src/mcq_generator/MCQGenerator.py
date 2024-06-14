import os
import json
import pandas as pd
import traceback
from langchain import LLMChain, PromptTemplate, HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import SequentialChain

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")

# repo_id_ls = ['meta-llama/Meta-Llama-3-8B-Instruct', "mistralai/Mistral-7B-v0.1", "facebook/m2m100_1.2B"]

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    temperature = 0.7,
    max_length = 128,
    token = api_key
)

#First chain getting all the initial inputs
TEMPLATE = '''
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz \
of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.

### RESPONSE_JSON
{response_json}
'''

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = TEMPLATE
 )
quiz_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key = "quiz", verbose = True)

#Second chain
TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables = ["subject", "quiz"],
    template = TEMPLATE2
)

review_chain = LLMChain(llm = llm, prompt = quiz_evaluation_prompt, output_key = "review", verbose = True)

#Creating a complete chain
chain = SequentialChain(
    chains = [quiz_chain, review_chain],
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    output_variables = ["quiz", "review"]
)

data_file_path = r"/Users/vineetdhokare/Documents/genai/mcqgen/data.txt"
with open(data_file_path, 'r') as file:
    INPUT_TEXT = file.read()

#Executing the llm chain
# quiz_output = chain(
#     {"text":INPUT_TEXT, "number":2, "subject":"machine learning", "tone":"medium", "response_json":json.dumps(RESPONSE_JSON)}
# )