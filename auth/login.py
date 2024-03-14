import streamlit as st
from connectdb import connect_to_postgresql
import bcrypt

def authenticate_with_database(username, input_password):
    connection = connect_to_postgresql()
    cursor = connection.cursor()

    query = f"SELECT * FROM accounts WHERE username = '{username}';"
    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        stored_hashed_password = result[2]  # Assuming the hashed password is in the second column

        # Ensure input_password is of type bytes
        input_password_bytes = input_password.encode('utf-8')

        if bcrypt.checkpw(input_password_bytes, stored_hashed_password.encode('utf-8')):
            st.session_state['loggedIn'] = True
            return True
        else:
            st.session_state['loggedIn'] = False

    cursor.close()
    connection.close()
    return False

# Function to handle login and logout    
def login_partial():
    st.sidebar.header("Login")

    # Check if the user is already logged in
    if not st.session_state.get('loggedIn', False):
        # User is not logged in, show login form
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")

        if st.sidebar.button("Login"):
            db_authentication = authenticate_with_database(username, password)
            if db_authentication:
                st.session_state['loggedIn'] = True
                
                st.rerun()  # Force a re-run to update the UI
            else:
                st.sidebar.error('Username/password is incorrect')

def logout_partial():
    st.sidebar.success('Login successful!')

    if st.sidebar.button("Logout"):
        st.session_state['loggedIn'] = False
        st.sidebar.success('Logout successful!')
        st.rerun()  # Force a re-run to update the UI
    if st.sidebar.button('Clear Cache'):
        st.cache_resource.clear()

def login():
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
    if not st.session_state['loggedIn']:
        login_partial()
    else:
        logout_partial()

    return st.session_state['loggedIn']