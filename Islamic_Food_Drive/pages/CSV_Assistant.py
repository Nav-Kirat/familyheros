import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": query}
    ]
)



# --- Page Config ---
st.set_page_config(page_title="CSV Assistant", page_icon="📊", layout="wide")
st.title("📦 Ask about the Food Hamper Data")

# --- Load API Key ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()
openai.api_key = api_key

# --- Load CSV Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("updated_merged_data_with_distance.csv")
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors='ignore')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# --- Data Summary ---
@st.cache_data
def generate_summary(df):
    summary = f"Dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.\n"
    summary += "Columns:\n"
    for col in df.columns:
        summary += f"- {col} ({df[col].dtype}, {df[col].isna().sum()} nulls)\n"
    sample = df.head(5).to_markdown(index=False)
    return summary + "\nSample (first 5 rows):\n" + sample

data_summary = generate_summary(df)

# --- Enrich Context with Narratives ---
def generate_narrative(df):
    if {"pickup_month", "monthly_hamper_demand", "unique_clients", "avg_distance_km", "total_visits", "total_dependents", "returning_proportion"}.issubset(df.columns):
        df = df.sort_values("pickup_month").dropna(subset=["pickup_month", "monthly_hamper_demand"])
        df = df.head(5)
        narrative = "Recent hamper activity:\n"
        for _, row in df.iterrows():
            narrative += (
                f"In {row['pickup_month']}, {int(row['monthly_hamper_demand'])} hampers were needed "
                f"for {int(row['unique_clients'])} clients. "
                f"Avg distance: {row['avg_distance_km']:.1f} km, visits: {int(row['total_visits'])}, "
                f"dependents: {int(row['total_dependents'])}, returning rate: {row['returning_proportion']:.2%}.\n"
            )
        return narrative
    return ""

hamper_narrative = generate_narrative(df)

# --- Query OpenAI ---
def query_openai(query, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant for food hamper analytics.\n\n{context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI error: {e}"

# --- User Input UI ---
query = st.text_input("💬 Ask a question about the data:")
if query:
    with st.spinner("Analyzing..."):
        context = data_summary + "\n\n" + hamper_narrative
        response = query_openai(query, context)

    st.markdown("### 🤖 Assistant's Response")
    st.success(response)

    with st.expander("📄 Context Used"):
        st.text(context)

# --- Show Sample & Stats ---
with st.expander("📊 View Sample Data"):
    st.dataframe(df.head(10))

with st.expander("📈 Data Statistics"):
    st.write("Numerical Summary:")
    st.dataframe(df.describe())

    st.write("Column Info:")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes,
        "Nulls": df.isna().sum(),
        "Non-Nulls": df.notna().sum()
    })
    st.dataframe(col_info)
