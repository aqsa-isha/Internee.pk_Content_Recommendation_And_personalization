from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import streamlit as st
import pandas as pd

# Load environment variables
load_dotenv()

def main():
    # Load Google API Key from environment
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY is None or GOOGLE_API_KEY.strip() == "":
        st.error("Google API Key is not set. Please configure it in your .env file.")
        return

    # Path to the movies.csv file
    path = "movies.csv"
    if not os.path.exists(path):
        st.error(f"CSV file not found at path: {path}. Please provide a valid file.")
        return
    
    # Load the movies.csv data
    movies_df = pd.read_csv(path)
    
    # Set up Streamlit UI
    st.set_page_config(page_title="Enhanced Movie Recommender")
    custom_css()  # Add custom CSS styling
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎥 Personalized Movie Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Let us help you find your next favorite movie! 🍿✨</p>", unsafe_allow_html=True)
    
    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    
    # Create an agent using the CSV file
    try:
        agent = create_csv_agent(
            llm=llm,
            path=path,  
            verbose=True,
            allow_dangerous_code=True  
        )
    except Exception as e:
        st.error(f"Error while creating the agent: {e}")
        return

    # User preference tracking (session state)
    if "preferences" not in st.session_state:
        st.session_state["preferences"] = []

    # Query input field
    st.markdown("<h3 style='color: #007BFF;'>🎬 What type of movie do you want to watch?</h3>", unsafe_allow_html=True)
    query = st.text_input("", placeholder="Type a genre, movie name, or keyword here...")

    # Process query and store preferences
    if query and query.strip() != "":
        with st.spinner(text="✨ Finding personalized recommendations..."):
            try:
                # Run the query through the agent
                response = agent.run(query)
                
                # Extract key genres or keywords for analysis
                st.session_state["preferences"].append(query)
                
                # Display personalized recommendations
                st.markdown("<h3 style='color: #FFC107;'>🎯 Personalized Recommendations</h3>", unsafe_allow_html=True)
                st.write(response)
                
                # Display additional diverse suggestions
                st.markdown("<h3 style='color: #FF5722;'>🌟 Diverse Suggestions</h3>", unsafe_allow_html=True)
                diverse_recommendations = suggest_diverse_movies(movies_df)
                st.write(diverse_recommendations)
                
            except Exception as e:
                st.error(f"Error during query processing: {e}")

    # Sidebar: User preferences and feedback
    st.sidebar.markdown("<h3 style='color: #6c757d;'>📝 Your Preferences</h3>", unsafe_allow_html=True)
    st.sidebar.write(st.session_state["preferences"])
    
    st.sidebar.markdown("<h3 style='color: #6c757d;'>💬 Feedback</h3>", unsafe_allow_html=True)
    feedback = st.sidebar.radio("Rate the recommendations:", ["👍 Good", "👎 Bad", "🤔 Neutral"])
    if st.sidebar.button("Submit Feedback"):
        st.success("Thank you for your feedback! 🌟")

def suggest_diverse_movies(df):
    """
    Suggest diverse movies based on different genres and other attributes.
    """
    try:
        # Ensure the DataFrame has the required columns
        required_columns = ["title", "genre", "year"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return f"Error: The following required columns are missing from the CSV: {', '.join(missing_columns)}"
        
        # Get diverse sample
        diverse_sample = df.sample(5)  # Randomly sample 5 diverse movies
        return diverse_sample[["title", "genre", "year"]]
    except Exception as e:
        return f"Error suggesting diverse movies: {e}"

def custom_css():
    """
    Add custom CSS styles to the Streamlit app.
    """
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 0;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #4CAF50;
            padding: 10px;
            font-size: 16px;
        }
        .stTextInput > div > div > input:focus {
            outline: none;
            border: 2px solid #007BFF;
            box-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stSidebar > div {
            background-color: #343a40;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
