#!/usr/bin/env python3
"""
genai_data_agent.py

A CLI data-analysis agent using Google GenAI (Gemini) + DuckDB for SQL execution.

Usage:
    export GOOGLE_API_KEY="your_key_here"
    python genai_data_agent.py --file /path/to/data.csv

What it does:
 - Loads CSV or Excel into pandas
 - Preprocesses columns (dates, numerics), quotes string fields
 - Cleans text columns with a precise NLP pipeline (unicode normalization, URL/email removal,
   contraction expansion, punctuation removal, lemmatization using spaCy or NLTK)
 - Loads processed CSV into DuckDB as table "uploaded_data"
 - Computes summary stats locally, then asks Gemini to generate insights
 - Prints Gemini's insights, then prompts for follow-up questions
 - For follow-ups: sends schema + question to Gemini and asks it to return DuckDB SQL only,
   executes the SQL, prints the SQL and results.

Notes:
 - Requires: pandas, duckdb, google-genai (library name: google), spacy and/or nltk (optional).
 - If spaCy model 'en_core_web_sm' is not available, the script will attempt NLTK fallback.
"""

import os
import sys
import argparse
import tempfile
import re
import html
import unicodedata
import json
from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd
import duckdb

os.environ["GOOGLE_API_KEY"] = "Your_API_KEY"


# Google GenAI client
try:
    from google import genai
except Exception as e:
    print("Missing `google.genai` library. Install it with `pip install google-genai`.")
    raise

# Optional NLP libs
SPACY_AVAILABLE = False
NLTK_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# Simple contractions map (expand common contractions)
_CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he's": "he is", "she's": "she is",
    "it's": "it is", "i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
    "let's": "let us", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "they're": "they are", "we're": "we are", "we've": "we have", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "you've": "you have", "you're": "you are",
    "you'll": "you will", "could've": "could have", "should've": "should have", "would've": "would have"
}

# Regexes used in text cleaning
URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'\S+@\S+\.\S+', flags=re.IGNORECASE)
NON_PRINTABLE_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')
MULTI_WHITESPACE_RE = re.compile(r'\s+')

# Initialize NLP pipeline (try spacy, fallback to nltk)
SPACY_NLP = None
if SPACY_AVAILABLE:
    try:
        SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        # user might not have the model; try to download if possible (may fail offline)
        try:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            SPACY_NLP = None

if NLTK_AVAILABLE:
    # try to download required packages quietly if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/wordnet')
    except Exception:
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass
    _NLTK_LEMMATIZER = WordNetLemmatizer()
else:
    _NLTK_LEMMATIZER = None


def expand_contractions(text: str) -> str:
    # Expand contractions using the small map above (case-insensitive)
    def _replace(match):
        word = match.group(0).lower()
        return _CONTRACTIONS.get(word, match.group(0))
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in _CONTRACTIONS.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(_replace, text)


def clean_text(text: str, do_lemmatize: bool = True) -> str:
    """
    Aggressive, precise text cleaning:
      - HTML unescape
      - Unicode normalization
      - Remove URLs and emails
      - Remove control/non-printable chars
      - Expand contractions (basic)
      - Lowercase
      - Remove punctuation & special chars (keep a-z0-9 and whitespace)
      - Collapse whitespace
      - Lemmatize (spaCy preferred; NLTK fallback)
    """
    if text is None:
        return ""
    # Ensure string
    text = str(text)

    # HTML entities, unicode normalization
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs and emails
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)

    # Remove non-printable characters
    text = NON_PRINTABLE_RE.sub(" ", text)

    # Expand contractions
    text = expand_contractions(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation and non-alphanumeric (keep spaces)
    # But keep apostrophes already expanded; now remove everything not a-z0-9 or whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Collapse whitespace
    text = MULTI_WHITESPACE_RE.sub(' ', text).strip()

    # Lemmatize tokens
    if do_lemmatize:
        if SPACY_NLP is not None:
            try:
                doc = SPACY_NLP(text)
                lemmas = [tok.lemma_ for tok in doc if tok.lemma_ != ""]
                text = " ".join(lemmas)
            except Exception:
                # fallback to simple NLTK lemmatizer
                if _NLTK_LEMMATIZER is not None:
                    tokens = word_tokenize(text)
                    lemmas = [_NLTK_LEMMATIZER.lemmatize(t) for t in tokens]
                    text = " ".join(lemmas)
        elif _NLTK_LEMMATIZER is not None:
            tokens = word_tokenize(text)
            lemmas = [_NLTK_LEMMATIZER.lemmatize(t) for t in tokens]
            text = " ".join(lemmas)
        else:
            # no lemmatizer available; keep cleaned tokens
            pass

    return text


def preprocess_dataframe(df: pd.DataFrame, text_columns: List[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    Preprocess dataframe:
     - Parse date-like columns
     - Try to convert object columns with numbers to numeric
     - Ensure string columns are quoted properly when saved to CSV
     - Clean text columns using clean_text()
    Returns path to temporary CSV and the processed dataframe
    """
    df = df.copy()

    # Standard na_values already handled if read with pandas; ensure string cols are str
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

    # Parse dates: heuristically any column with 'date' in name
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Numeric conversion for object columns where possible
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass

    # Detect text columns if none explicitly provided: assume object/string columns longer than 20 chars on average
    if text_columns is None:
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].map(lambda x: len(str(x)) if pd.notnull(x) else 0).mean()
                if avg_len >= 20:
                    text_columns.append(col)

    # Clean text columns precisely
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).map(lambda t: clean_text(t))

    # Write to temp CSV with quoting
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    df.to_csv(tmp.name, index=False, quoting=1)  # quoting=csv.QUOTE_ALL -> 1
    return tmp.name, df


def compute_local_stats(df: pd.DataFrame, text_columns: List[str]) -> Dict:
    """
    Compute local summary statistics to provide context for GenAI.
    Returns a dict with load-bearing stats.
    """
    stats = {}
    stats['rows'] = int(df.shape[0])
    stats['columns'] = int(df.shape[1])
    stats['column_names'] = list(df.columns)

    # Missing value counts (top 10)
    missing = df.isnull().sum().sort_values(ascending=False)
    stats['missing_top'] = missing[missing > 0].head(10).to_dict()

    # Numeric column summaries
    numeric = df.select_dtypes(include=['number'])
    stats['numeric_describe'] = numeric.describe().to_dict() if not numeric.empty else {}

    # Categorical top values
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_top = {}
    for c in cat_cols:
        top = df[c].fillna("<<NULL>>").value_counts().head(10)
        cat_top[c] = top.to_dict()
    stats['categorical_top'] = cat_top

    # Date ranges
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_ranges = {}
    for c in date_cols:
        try:
            date_ranges[c] = {"min": str(df[c].min()), "max": str(df[c].max())}
        except Exception:
            pass
    stats['date_ranges'] = date_ranges

    # Text column stats: avg length + top words
    text_stats = {}
    for tc in text_columns:
        if tc in df.columns:
            lens = df[tc].fillna("").map(len)
            all_text = " ".join(df[tc].fillna("").astype(str).tolist())
            tokens = all_text.split()
            counter = Counter(tokens)
            top_words = counter.most_common(20)
            text_stats[tc] = {
                "avg_len": float(lens.mean()) if len(lens) else 0,
                "top_words": top_words
            }
    stats['text_stats'] = text_stats

    return stats


def pretty_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)


def genai_generate_insights(client, stats: Dict, max_output_chars: int = 4000) -> str:
    """
    Call Google GenAI to generate insights from precomputed stats.
    We pass the summary stats JSON and ask for concise, prioritized insights and recommended follow-up SQL questions.
    """
    prompt = f"""
You are an expert data analyst. Below is a JSON summary of a dataset loaded into a DuckDB table named `uploaded_data`.
Produce:
  1) Up to 8 concise insights (1-2 sentences each) prioritized by importance and actionability.
  2) Up to 6 suggested follow-up natural-language queries that a user might ask next (one per line).
Be concrete, do not hallucinate column names beyond what's in the JSON. Use the JSON as truth.

JSON:
{pretty_json(stats)}
"""
    # Use GenAI to generate content
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    # Different SDKs return different shapes; try common ones
    text = ""
    if hasattr(resp, "text"):
        text = resp.text
    elif isinstance(resp, dict) and 'candidates' in resp:
        text = resp['candidates'][0].get('content', '')
    else:
        try:
            text = str(resp)
        except Exception:
            text = ""
    return text[:max_output_chars]


def genai_generate_sql(client, schema: Dict[str, str], user_question: str, dialect_hint: str = "DuckDB") -> str:
    """
    Ask GenAI to generate a DuckDB-compatible SQL query for the user_question.
    The model is instructed to return only SQL in a triple backtick block or plain.
    """
    schema_json = pretty_json(schema)
    prompt = f"""
You are a SQL generator for the {dialect_hint} SQL dialect. Table name: uploaded_data.
Schema (column -> type):
{schema_json}

Task: Given the user's natural-language question below, produce a single SQL statement that answers it using the 'uploaded_data' table.
Constraints:
 - Return only the SQL statement in a single fenced code block (triple backticks) or plain text, nothing else.
 - Use standard SQL functions available in DuckDB (regexp_replace, lower, trim, CAST, etc.) if text cleaning is needed.
 - If multiple steps are required, produce a single SQL query using WITH clauses.
User question:
{user_question}
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    text = ""
    if hasattr(resp, "text"):
        text = resp.text
    else:
        try:
            text = str(resp)
        except Exception:
            text = ""

    # Extract SQL from the response (strip fences if present)
    sql = text.strip()
    # Remove leading/trailing triple backticks and language hints
    sql = re.sub(r"^```(sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql, flags=re.IGNORECASE)
    return sql.strip()


def execute_sql_and_print(conn: duckdb.DuckDBPyConnection, sql: str, show_limit: int = 50):
    try:
        df = conn.execute(sql).fetchdf()
        pd.set_option("display.max_rows", None)
        print("\n--- SQL USED ---")
        print(sql)
        print("\n--- RESULT (top rows) ---")
        if df.empty:
            print("(no rows returned)")
        else:
            print(df.head(show_limit).to_string(index=False))
    except Exception as e:
        print("Error executing SQL:", e)


def infer_schema_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Simple mapping from pandas dtypes to SQL types (DuckDB-ish)
    """
    mapping = {}
    for c in df.columns:
        typ = df[c].dtype
        if pd.api.types.is_integer_dtype(typ):
            mapping[c] = "BIGINT"
        elif pd.api.types.is_float_dtype(typ):
            mapping[c] = "DOUBLE"
        elif pd.api.types.is_datetime64_any_dtype(typ):
            mapping[c] = "TIMESTAMP"
        else:
            mapping[c] = "TEXT"
    return mapping


def main(args):
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable and try again.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load file
    fp = args.file
    if not os.path.exists(fp):
        print(f"File not found: {fp}")
        sys.exit(1)

    print("Loading file:", fp)
    if fp.lower().endswith(".csv"):
        df = pd.read_csv(fp, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
    elif fp.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(fp, na_values=['NA', 'N/A', 'missing'])
    else:
        print("Unsupported file format. Use CSV or Excel.")
        sys.exit(1)

    # Detect text columns (user could optionally supply, but heuristic is fine)
    candidate_text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].fillna("").map(lambda x: len(str(x))).mean()
            if avg_len >= 20:
                candidate_text_columns.append(col)

    print(f"Detected text columns for cleaning: {candidate_text_columns}")

    # Preprocess and clean text
    tmp_csv_path, df_processed = preprocess_dataframe(df, text_columns=candidate_text_columns)

    # Connect to DuckDB and load table
    conn = duckdb.connect(database=":memory:")
    # DuckDB can read CSV directly, but we already have df_processed; create table from pandas df
    conn.register("uploaded_data_df", df_processed)
    conn.execute("CREATE TABLE uploaded_data AS SELECT * FROM uploaded_data_df;")

    # Compute local stats and ask GenAI to generate insights
    stats = compute_local_stats(df_processed, candidate_text_columns)
    print("\nGenerating high-level insights using GenAI (Gemini)...")
    insights = genai_generate_insights(client, stats)
    print("\n--- INSIGHTS ---")
    print(insights)

    # Now prompt user for follow-up natural language queries, convert to SQL, execute
    while True:
        print("\nDo you have another question about this data? (enter blank to exit)\nExamples: 'Top 5 customers by revenue', 'Average order value by month', 'Show rows where review_text mentions refund'")
        user_q = input("\nYour question (or press Enter to quit): ").strip()
        if user_q == "":
            print("No more questions. Goodbye.")
            break

        # Build schema hint
        schema = infer_schema_types(df_processed)
        sql = genai_generate_sql(client, schema, user_q, dialect_hint="DuckDB")
        if not sql:
            print("GenAI did not return SQL. Here's a best-effort fallback: try a simple SELECT * ...")
            print("Fallback SQL: SELECT * FROM uploaded_data LIMIT 50;")
            continue

        print("\nExecuting the SQL produced by GenAI...")
        execute_sql_and_print(conn, sql, show_limit=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenAI Data Agent (DuckDB + Gemini)")
    parser.add_argument("--file", "-f", required=True, help="Path to CSV or Excel file")
    args = parser.parse_args()
    main(args)
