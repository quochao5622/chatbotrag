import streamlit as st
import psycopg2

# Function to connect to the PostgreSQL database
def connect_to_postgresql():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432",
            database="postgres"
        )
        return connection
    except psycopg2.Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

# Function to execute a query and fetch data
def execute_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    return data

