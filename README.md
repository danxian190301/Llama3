# Query SQL Database Using Natural Language with Llama 3 and LangChain （RFM data)
<details>
  <summary>Tables of Contents</summary>


  1. Overview
  2. Setting Up the Environment
  3. Creating a Custom Model to Generate SQL
  4. Building the LangChain Integration
  5. Establishing Database Connection
  6. Generating SQL Queries using Custom LLM
  7. Creating the Full Chain
  8. Building the Streamlight App
  9. Testing the App

</details>

## Overview
![screenshot](https://github.com/danxian190301/Llama3/blob/main/1_QreqBevJQX7M4G9hsv1oeA.webp)

Create a Streamlit application that allows us to extract insights from SQL databases using natural language. The project uses Llama 3, an open-source Large Language Model (LLM) that runs locally, ensuring your data remains in your network. To host the LLM locally, we use Ollama, a prerequisite for this project. This tutorial will help you to:
- Create a custom LLM.
- Create LangChain chains (AI agents).
- Create Streamlit applications.
- Integrate NLP and AI with Oracle.

## Setting Up the Environment
Use Oracle as our database. For setup guidance on [Ollama](https://www.youtube.com/watch?v=CE9umy2NlhE) follow this [guide](https://www.youtube.com/watch?v=CE9umy2NlhE). We also need [OpenWebUI](https://www.youtube.com/watch?v=YUYZd71hg3w), refer to this [link](https://www.youtube.com/watch?v=YUYZd71hg3w) for the setup.
Based on this project, the model we use is Llama3.
![screenshot](https://github.com/danxian190301/Llama3/blob/main/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240805132008.png)

## Creating a Custom Model to Generate SQL
The next step is to create a custom LLM that generates SQL queries based on user input. We will use OpenWebUI for this task. Here is how to set it up:

1. Create a New Model File:
- Name and describe your custom model.
- Define the model content, basing it on Llama 3 with a temperature set to zero.
- In the system message, instruct the model to generate SQL for Oracle, incorporating the database schema and relationships.
2. Model Content Definition:
- Provide the database schema.
- Include guidelines on query structure and table joins.
- Focus on SQL query generation, ignoring formatting in the response.
3. Save the Model:
- Add prompt suggestions and relevant categories.
- Save the model file to make it available within the Ollama ecosystem.

## Building the LangChain Integration
LangChain is an open-source framework for building applications based on the Large Language Model. It allows us to integrate LLMs with internal data sources and generate responses through automated chains of operations. We use LangChain to implement natural language interaction with the database. We coded our solution in a Python app.py file.
First, ensure the following packages are installed in your environment:
- langchain
- langchain-community
- streamlit
- cx_Oracle
```python
!pip install streamlit sqlalchemy pandas requests cx_Oracle
!pip install langchain
!pip install langchain --upgrade # Upgrade langchain to the latest version
!pip install langchain-community # Install the langchain-community package
```
```python
import re
import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

## Establishing Database Connection
Save your database credentials in local variables and create a function to establish a connection using LangChain’s SQLDatabase wrapper.
```python
# Database configuration
db_username = 'xxxx'
db_password = 'xxxxx'
db_host = 'xxxxxxxxxxx'
db_port = 'xxxx'
db_service = 'xxxxxxxx'

def init_database() -> create_engine:
    db_uri = f"oracle+cx_oracle://{db_username}:{db_password}@{db_host}:{db_port}/{db_service}"
    return create_engine(db_uri)
```
## Generating SQL Queries using Custom LLM
Create a custom function to generate SQL based on user input. This function will use Ollama to access the custom model.
```python
# Function to generate SQL queries
def llm_query(question: str) -> str:
    try:
        # Initialize the model with the correct name
        llm = ChatOllama(model="llama3")

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template("{topic}")

        # Define the chain for processing
        chain = prompt | llm | StrOutputParser()

        # Invoke the chain with the user question
        sql = chain.invoke({"topic": f"{question}"})

        # Clean up the SQL query (remove unwanted characters)
        sql = re.sub(r'(?:(?<=_) | (?=_))', '', sql)

        # Return the generated SQL query
        return sql
    except Exception as e:
        return f"An error occurred: {e}"
```

## Creating the Full Chain
Develop a function that allows natural language interaction with the database. This function will take user input, database connection, and chat history as inputs.
```python
# Function to get response from the database
def get_response(user_query: str, db: create_engine, chat_history: list) -> str:
    sql_query = llm_query(user_query)

    # Check if the SQL query is valid
    if "An error occurred:" in sql_query:
        return sql_query  # Return the error message as the response

    try:
        # Execute the SQL query
        with db.connect() as connection:
            result = connection.execute(text(sql_query)).fetchall()
            sql_response = result if result else "No results found."
    except Exception as e:
        sql_response = f"Database error: {e}"

    template = """
    You are an experienced data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema, question, sql query, and sql response, write a natural language response.

    Conversation History: {chat_history}
    User question: {question}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3", temperature=0)

    chain = (
        RunnablePassthrough.assign(
            response=lambda vars: sql_response,
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
```
## Building the Streamlit App
Create Streamlit components to store chat history and manage user interactions.

Initialize the database & Handle User Input：
```python
# Streamlit app
def main():
    st.title("SQL Query Assistant")

    # Initialize the database connection
    if 'db' not in st.session_state:
        st.session_state.db = init_database()
        st.success("Connected to database!")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Type a question...")

    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("User"):
            st.markdown(user_query)

        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)

        with st.chat_message("AI"):
            st.markdown(response)

if __name__ == "__main__":
    main()
```

## Testing the App
Test our app with a few queries to ensure it’s functioning correctly.

The question we ask is： Analysis ORDER_STATUS, analyzing the distribution of order statuses over time?

![screenshot](https://github.com/danxian190301/Llama3/blob/main/3.png)

The app should answer the questions accurately.

![screenshot](https://github.com/danxian190301/Llama3/blob/main/2.png)

Below is a screenshot of the complete process where we simulated asking a question and the AI assistant answering it.

![screenshot](https://github.com/danxian190301/Llama3/blob/main/4.png)

LangChain's integration with LLM opens up a wide range of possibilities for data analysis, especially pattern-specific data analysis. Users can now get answers using natural language, thus enhancing and complementing existing business intelligence solutions.



  
