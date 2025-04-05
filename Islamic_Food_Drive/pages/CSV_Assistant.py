import streamlit as st
import pandas as pd
import importlib
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check and import libraries
def import_with_error_handling(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        st.error(f"Error importing {module_name}: {e}")
        st.error("Please ensure all required libraries are installed.")
        st.stop()

# --- Page Config ---
st.set_page_config(page_title="CSV Assistant", page_icon="📊", layout="wide")
st.title("Ask about the CSV Data!")

# Set up OpenAI client with API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Load & Cache Data ---
@st.cache_data
def load_csv_data():
    return pd.read_csv("updated_merged_data_with_distance.csv")

try:
    df = load_csv_data()
except Exception as e:
    st.error(f"Error loading CSV data: {e}")
    st.stop()

# --- Generate Data Summary ---
@st.cache_data
def generate_data_summary(df):
    # Get basic statistics
    num_rows = len(df)
    num_columns = len(df.columns)
    column_names = ", ".join(df.columns.tolist())
    
    # Get sample data
    sample_data = df.head(5).to_string()
    
    # Create summary
    summary = f"""
    Dataset Summary:
    - Total records: {num_rows}
    - Total columns: {num_columns}
    - Column names: {column_names}
    
    Sample data (first 5 rows):
    {sample_data}
    
    This dataset contains information about food hamper distribution, including client details,
    pickup information, and distance metrics.
    """
    return summary

data_summary = generate_data_summary(df)

# --- Generate Narrative from Data ---
def generate_narrative_from_enriched(df, limit=5):
    df = df.sort_values("pickup_month").dropna(subset=["pickup_month", "monthly_hamper_demand"])
    df = df.head(limit)

    narrative = "Here are recent hamper pickup summaries:\n"
    for _, row in df.iterrows():
        narrative += (
            f"In {row['pickup_month']}, approximately {int(row['monthly_hamper_demand'])} hampers were needed "
            f"to serve {int(row['unique_clients'])} clients. "
            f"The average client traveled {row['avg_distance_km']:.1f} km and had {int(row['total_visits'])} visits. "
            f"Households had around {int(row['total_dependents'])} dependents total. "
            f"Returning rate was {row['returning_proportion']:.2%}.\n"
        )
    return narrative

hamper_narrative = generate_narrative_from_enriched(df)

# --- Query OpenAI ---
def query_openai(query, context):
    try:
        # Using the cheapest model - text-ada-001
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # This is one of the cheapest models available
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that answers questions about food hamper distribution data. Use the following context to answer questions:\n\n{context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI: {e}"

# --- UI: User Input & Display ---
st.markdown("Ask questions about the food hamper distribution data in the CSV file.")

query = st.text_input("💬 Ask a question about the data:")
if query:
    with st.spinner("Generating answer..."):
        # Combine data summary and narrative for context
        context = data_summary + "\n\n" + hamper_narrative
        answer = query_openai(query, context)

    st.markdown("### 🤖 Assistant's Response")
    st.success(answer)

    with st.expander("📄 Data Context"):
        st.text(context)

# --- Show Data Sample ---
with st.expander("📊 View Data Sample"):
    st.dataframe(df.head(10))

# --- Show Data Statistics ---
with st.expander("📈 Data Statistics"):
    st.write("Numerical Columns Statistics:")
    st.dataframe(df.describe())
    
    st.write("Column Information:")
    column_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isna().sum()
    })
    st.dataframe(column_info) 