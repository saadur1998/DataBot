import streamlit as st

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import os

import pandas as pd

form = st.sidebar.form(key='my_form')
user_api_key = form.text_input(
    label="#### Your OpenAI API key",
    placeholder="Paste your openAI API key, sk-",
    type="password")
uploaded_file = form.file_uploader("Choose a file")
text_prompt = form.text_area('Text Prompt', value='Write your query here')
submit_button = form.form_submit_button(label='Submit')
if submit_button :
    df = pd.read_csv(uploaded_file)

    st.title("OpenAI DataBot"
             )
    st.subheader("A basic chatbot for interacting with dataset using OpenAI's API")
    
    st.subheader("Your data:")
    st.write(df)
    
    #horizontal line
    st.markdown("""---""")

    st.subheader("AI generated response:")
    os.environ['OPENAI_API_KEY']=user_api_key
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0),
        df,
        verbose=True,allow_dangerous_code=True)
    
    result = agent.run(text_prompt)
    st.write(result)
