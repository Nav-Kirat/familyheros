import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

st.set_page_config(page_title="CSV + Org Assistant", page_icon="📊", layout="wide")

# --- Load API Key ---
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("Missing OpenAI API key in .env file or Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Load Data ---
@st.cache_data
def load_csv():
    df = pd.read_csv("updated_merged_data_with_distance.csv")
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors='ignore')
    return df

@st.cache_data
def load_markdown():
    with open("Islamicfamily.md", "r", encoding="utf-8") as f:
        return f.read()

df = load_csv()
doc_context = load_markdown()

# --- UI Header ---
st.title("🧠 Ask About Food Hamper Data or Islamic Family Organization")

# --- CSV Summary ---
@st.cache_data
def generate_summary(df):
    summary = f"CSV has {df.shape[0]:,} rows and {df.shape[1]} columns:\n"
    summary += ", ".join(df.columns)
    return summary

data_summary = generate_summary(df)

# --- Smart Query Handler ---
def smart_answer(query):
    q = query.lower()

    # 1. CSV-specific rules
    if "total clients" in q:
        if "unique_client" in df.columns:
            return f"Total unique clients: {df['unique_client'].nunique():,}"

    elif "more than 3 dependents" in q:
        if "dependents_qty" in df.columns:
            count = df[df["dependents_qty"] > 3]["unique_client"].nunique()
            return f"Clients with more than 3 dependents: {count:,}"

    elif "day" in q and "pick" in q:
        if "pickup_day" in df.columns:
            df["pickup_day"] = pd.to_datetime(df["pickup_day"], errors="coerce")
            top_day = df["pickup_day"].dt.day_name().value_counts().idxmax()
            return f"Most common pickup day: {top_day}"

    elif "average distance" in q and "month" in q:
        if "pickup_month" in df.columns and "distance_km" in df.columns:
            avg = df.groupby("pickup_month")["distance_km"].mean().round(2)
            return "\n".join([f"{k}: {v} km" for k, v in avg.items()])

    # 2. Organization (markdown-based) knowledge
    org_keywords = ["islamic family", "mission", "the hub", "green room", "refugee", "support", "mental health", "prison"]
    if any(word in q for word in org_keywords) or query.endswith("?"):
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that answers questions about a community organization using this context:\n\n{doc_context}"},
            {"role": "user", "content": query}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

    # 3. Fallback to AI summary with CSV context
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. The user has access to this CSV summary:\n{data_summary}"},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content

# --- User Input ---
user_q = st.text_input("💬 Ask a question about the data or organization:")
if user_q:
    with st.spinner("Thinking..."):
        result = smart_answer(user_q)
        st.markdown("### 🤖 Response")
        st.success(result)

# --- Optional Preview Sections ---
with st.expander("📊 CSV Sample"):
    st.dataframe(df.head())

# with st.expander("📄 Organization Overview (Markdown)"):
#     st.text(doc_context[:1500] + "...")
