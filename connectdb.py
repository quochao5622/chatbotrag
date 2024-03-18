import os

import pandas as pd
import streamlit as st
import psycopg2
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma

from brain import get_embedding

load_dotenv()
def connect_to_postgresql():
    try:
        connection = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_DATABASE")
        )
        return connection
    except psycopg2.Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

# Function to execute a query and fetch data
@st.cache_data
def execute_query(connection=connect_to_postgresql(), query=""):
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    return data

@st.cache_data
def query_qas():
    query = '''SELECT * FROM qas'''
    data = execute_query(query=query)
    cols = [
        'id',
        'title',
        'embedding_title',
        'content',
        'source'
    ]
    df = pd.DataFrame(data, columns=cols)
    return df
@st.cache_resource
def load_chroma():
    try:
        collection = Chroma(persist_directory="./chromadb",
                        embedding_function=get_embedding(),
                        collection_name="rag")
    except Exception as e:
        st.error(f"Error connecting to Chroma: {e}")
        return None
    return collection