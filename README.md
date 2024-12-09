# 🎥 Movie Recommender using Gemini Pro

This project is a **Streamlit-based Movie Recommender System** powered by **Google Gemini Pro LLM** and a CSV dataset of movies. Users can input their preferences to get personalized recommendations and explore diverse movie suggestions.

## Features

- Personalized movie recommendations based on user queries.
- Diverse suggestions using a dataset of movies.
- Intuitive UI for preferences tracking and feedback.
- Powered by **Google Gemini Pro** for advanced language understanding.

---

## 🛠️ Setup and Run the Application

```bash
# Step 1: Clone the Repository and Navigate to the Project Directory
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

# Step 2: Create and Activate a Virtual Environment
# On Windows:
python -m venv myenv
myenv\Scripts\activate
# On macOS/Linux:
python3 -m venv myenv
source myenv/bin/activate

# Step 3: Install Dependencies
pip install -r requirements.txt

# Step 4: Run the Application
streamlit run app.py
