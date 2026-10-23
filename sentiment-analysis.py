# import all necessary modules and libraries
import streamlit as st
import pandas as pd
from transformers import pipeline

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="AI Sentiment Analysis Tool",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 2. CUSTOM CSS DESIGN
# ==================================================
st.markdown("""
<style>

/* Application background */
.stApp {
    background: linear-gradient(
        135deg,
        #F3E8FF 0%,
        #EDE9FE 50%,
        #E0F2FE 100%
    );
    font-family: 'Trebuchet MS', sans-serif;
}

/* Main page container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    margin: auto;
}

/* Main title */
h1 {
    color: #4C1D95 !important;
    text-align: center !important;
    font-family: 'Trebuchet MS', sans-serif !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}

/* Section headings */
h2, h3 {
    color: #5B21B6 !important;
    font-family: 'Trebuchet MS', sans-serif !important;
    font-weight: 700 !important;
}

/* Paragraphs and labels */
.stApp p,
.stApp label,
.stMarkdown {
    font-family: 'Trebuchet MS', sans-serif;
}

/* Center the introductory description */
.intro-text {
    text-align: center;
    color: #475569;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Text area */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 2px solid #C4B5FD !important;
    border-radius: 12px !important;
    font-family: 'Trebuchet MS', sans-serif !important;
    font-size: 16px !important;
}

/* Input focus */
.stTextArea textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px #DDD6FE !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #6D28D9 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.2rem !important;
    font-family: 'Trebuchet MS', sans-serif !important;
    font-weight: bold !important;
    transition: 0.3s ease;
}

/* Button hover */
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #4C1D95 !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #5B21B6 !important;
    font-family: 'Trebuchet MS', sans-serif !important;
    font-size: 16px;
    font-weight: bold;
    padding: 12px 18px;
}

.stTabs [aria-selected="true"] {
    color: #6D28D9 !important;
    border-bottom-color: #6D28D9 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.65);
    padding: 15px;
    border: 1px solid #C4B5FD;
    border-radius: 12px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #C4B5FD;
    border-radius: 10px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #C4B5FD;
}

/* Center buttons inside the main application */
.stButton {
    text-align: center;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #DDD6FE;
}

/* Responsive title for smaller screens */
@media (max-width: 768px) {
    h1 {
        font-size: 30px !important;
    }

    .block-container {
        padding-top: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. APPLICATION TITLE AND DESCRIPTION
# ==================================================
st.title("💜 AI Sentiment Analysis Application")

st.markdown("""
<div class="intro-text">
Analyze customer reviews using AI-powered sentiment analysis.
<br>
Discover whether feedback is positive, negative, or neutral.
</div>
""", unsafe_allow_html=True)

st.divider()

# ==================================================
# 4. LOAD THE SENTIMENT MODEL
# ==================================================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

try:
    with st.spinner("Loading the AI model... Please wait."):
        sentiment_pipeline = load_sentiment_model()

except Exception as e:
    st.error(
        "Unable to load the AI model. Check your internet connection "
        "and installed dependencies, then try again."
    )
    st.exception(e)
    st.stop()

# ==================================================
# 5. CREATE APPLICATION TABS
# ==================================================
tab1, tab2 = st.tabs([
    "📝 Analyze the Text",
    "📊 Analyze the CSV Dataset"
])

# ==================================================
# TAB 1: SINGLE REVIEW ANALYSIS
# ==================================================
with tab1:
    st.header("Analyze a Single Review")

    st.write(
        "Enter a customer review below to predict its sentiment."
    )

    user_input = st.text_area(
        "Please enter the customer review:",
        value="The service and the product were amazing. I loved it!",
        height=150,
        placeholder="Type a customer review here..."
    )

    if st.button(
        "✨ Analyze Sentiment",
        key="single_analysis",
        use_container_width=True
    ):
        if user_input.strip():
            try:
                with st.spinner("Analyzing your review..."):
                    result = sentiment_pipeline(
                        user_input.strip(),
                        truncation=True
                    )[0]

                label = result["label"]
                score = result["score"]

                st.subheader("Analysis Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Predicted Sentiment",
                        value=label.upper()
                    )

                with col2:
                    st.metric(
                        label="Model Confidence",
                        value=f"{score:.2%}"
                    )

                if label.lower() == "positive":
                    st.success(
                        f"😊 This review is classified as "
                        f"**{label.upper()}**."
                    )

                elif label.lower() == "negative":
                    st.error(
                        f"😞 This review is classified as "
                        f"**{label.upper()}**."
                    )

                else:
                    st.info(
                        f"😐 This review is classified as "
                        f"**{label.upper()}**."
                    )

                st.caption(
                    "The confidence score is the model's estimated "
                    "confidence in its predicted label."
                )

            except Exception as e:
                st.error(f"Could not analyze the review: {e}")

        else:
            st.warning(
                "Please enter some text before analyzing sentiment."
            )

# ==================================================
# TAB 2: CSV DATASET ANALYSIS
# ==================================================
with tab2:
    st.header("Analyze the CSV Dataset")

    st.write(
        "Upload a CSV file containing customer reviews. "
        "Your file must contain a column named `review_text`."
    )

    uploaded_file = st.file_uploader(
        "Choose your customer reviews CSV file",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.warning("The uploaded CSV file contains no data.")

            else:
                st.subheader("Preview of Your Dataset")
                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                text_column = "review_text"

                if text_column not in df.columns:
                    st.error(
                        f"The required column '{text_column}' "
                        "was not found in your CSV file. "
                        "Please check your column names."
                    )

                else:
                    st.write(
                        f"**Total records:** {len(df)}"
                    )

                    if st.button(
                        "🚀 Analyze the Entire Dataset",
                        key="csv_analysis",
                        use_container_width=True
                    ):
                        try:
                            with st.spinner(
                                "Analyzing reviews... Please wait."
                            ):
                                # Handle missing values
                                reviews = (
                                    df[text_column]
                                    .fillna("")
                                    .astype(str)
                                    .str.strip()
                                )

                                # Identify non-empty reviews
                                valid_indices = [
                                    i for i, review
                                    in enumerate(reviews)
                                    if review
                                ]

                                # Prepare result columns
                                df["AI_Sentiment"] = "No text"
                                df["AI_Confidence"] = float("nan")

                                # Process valid reviews in batches
                                if valid_indices:
                                    valid_reviews = [
                                        reviews.iloc[i]
                                        for i in valid_indices
                                    ]

                                    predictions = sentiment_pipeline(
                                        valid_reviews,
                                        batch_size=16,
                                        truncation=True
                                    )

                                    for idx, prediction in zip(
                                        valid_indices,
                                        predictions
                                    ):
                                        df.at[
                                            df.index[idx],
                                            "AI_Sentiment"
                                        ] = prediction["label"]

                                        df.at[
                                            df.index[idx],
                                            "AI_Confidence"
                                        ] = prediction["score"]

                            st.success(
                                "✅ Sentiment analysis completed!"
                            )

                            # Display analyzed data
                            st.subheader("Analysis Results")

                            st.dataframe(
                                df.head(10),
                                use_container_width=True
                            )

                            # Sentiment distribution
                            st.subheader("Sentiment Distribution")

                            sentiment_counts = (
                                df.loc[
                                    df["AI_Sentiment"] != "No text",
                                    "AI_Sentiment"
                                ]
                                .value_counts()
                            )

                            if not sentiment_counts.empty:
                                st.bar_chart(sentiment_counts)

                                # Summary metrics
                                metric_cols = st.columns(
                                    len(sentiment_counts)
                                )

                                for col, (sentiment, count) in zip(
                                    metric_cols,
                                    sentiment_counts.items()
                                ):
                                    with col:
                                        st.metric(
                                            label=sentiment.upper(),
                                            value=int(count)
                                        )
                            else:
                                st.info(
                                    "No non-empty reviews were found."
                                )

                            # Download analyzed dataset
                            csv_data = df.to_csv(
                                index=False
                            ).encode("utf-8-sig")

                            st.download_button(
                                label="⬇️ Download Sentiment Results",
                                data=csv_data,
                                file_name="Sentiment_Analysis_file.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as e:
                            st.error(
                                f"An error occurred during analysis: {e}"
                            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")

        except UnicodeDecodeError:
            st.error(
                "Unable to read the CSV file encoding. "
                "Please upload a UTF-8 encoded CSV file."
            )

        except Exception as e:
            st.error(f"Unable to process the uploaded file: {e}")

# ==================================================
# 6. FOOTER
# ==================================================
st.divider()

st.markdown("""
<div style="
    text-align: center;
    color: #6D28D9;
    font-family: 'Trebuchet MS', sans-serif;
    padding: 10px;
">
    <p><strong>AI Sentiment Analysis Tool</strong></p>
    <p>Powered by Streamlit, Hugging Face Transformers and DistilBERT.</p>
</div>
""", unsafe_allow_html=True)
