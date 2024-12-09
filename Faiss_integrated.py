from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Define the base directory
base_path = "./Movie_Recommendation_System"  # Relative path

# Ensure the directory exists
if not os.path.exists(base_path):
    os.makedirs(base_path)  # Create the directory if it doesn't exist

# Full path to the CSV file
csv_file_path = os.path.join(base_path, "movies.csv")

# Create Google Gemini LLM model with the API key
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key
if not api_key.strip():
    raise ValueError("API key for Google Gemini is missing!")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings("hku-nlp/instructor-base")

# Define the FAISS index file path
vectordb_file_path = os.path.join(base_path, "faiss_index")


def create_vector_db():
    """
    Function to create a FAISS vector database from the CSV file.
    """
    try:
        # Check if the CSV file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found at {csv_file_path}")

        # Load data from the CSV file
        print("Loading data from the CSV file...")
        loader = CSVLoader(file_path=csv_file_path)
        data = loader.load()

        # Create a FAISS instance for the vector database from the data
        print("Creating the FAISS vector database...")
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Save the vector database locally
        vectordb.save_local(vectordb_file_path)
        print(f"Vector database saved at {vectordb_file_path}")

    except Exception as e:
        print(f"Error while creating vector database: {e}")


def get_qa_chain():
    """
    Function to create a Retrieval QA chain using the FAISS vector database.
    """
    try:
        # Check if the FAISS index exists
        if not os.path.exists(vectordb_file_path):
            raise FileNotFoundError(f"FAISS index file not found at {vectordb_file_path}. "
                                     "Make sure to run create_vector_db() first.")

        # Load the FAISS vector database
        print("Loading the FAISS vector database...")
        vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        # Define the prompt template
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create the Retrieval QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        return chain

    except Exception as e:
        print(f"Error while loading QA chain: {e}")
        return None


if __name__ == "__main__":
    # Create the vector database (run this only once or when the CSV changes)
    try:
        print("Creating vector database...")
        create_vector_db()
    except Exception as e:
        print(f"Error during vector database creation: {e}")

    # Load the QA chain
    try:
        print("Loading QA chain...")
        chain = get_qa_chain()

        if chain is not None:
            # Query the chain
            query = "What are some good action movies?"
            result = chain({"query": query})

            print("\nAnswer:")
            print(result["result"])  # Prints the generated answer

            print("\nSource Documents:")
            for doc in result["source_documents"]:
                print(doc.page_content)  # Optionally print the source documents
    except Exception as e:
        print(f"Error during QA chain execution: {e}")
