import streamlit as st
import docx
import re
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# 1. App Configuration & Setup
st.set_page_config(page_title="Word Analyzer", page_icon="🧚", layout="centered")
st.title("🧚 Word & Adjective Frequency Analyzer")
st.write("Upload a `.docx` file to extract, clean, and visualize word frequencies.")

# Safely download NLTK data behind a Streamlit cache layer
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

download_nltk_data()

# 2. Text Extraction Function
def extract_text_from_docx(file_buffer):
    """Reads all text from an uploaded Word document buffer."""
    doc = docx.Document(file_buffer)
    return ' '.join([para.text for para in doc.paragraphs])

# 3. Dynamic Plotting Function
def plot_top_frequencies(word_count_frequency, top_n=15, title="Top Word Frequencies", palette='rocket'):
    """Plots a bar chart of the top_n items from a Counter object inside Streamlit."""
    most_common = word_count_frequency.most_common(top_n)

    if not most_common:
        st.warning(f"No valid data found to plot for: {title}")
        return

    words_to_plot, counts_to_plot = zip(*most_common)

    # Build the Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=list(words_to_plot), 
        y=list(counts_to_plot), 
        palette=palette, 
        hue=list(words_to_plot), 
        legend=False, 
        ax=ax
    )

    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel("Words,words,words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)

    # Add numerical count strings slightly above each bar
    max_count = max(counts_to_plot)
    for i, count in enumerate(counts_to_plot):
        ax.text(i, count + (max_count * 0.01), str(count), ha='center', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig) # Embed the graph into the webpage

# 4. Sidebar Controls
st.sidebar.header("🔧 Settings")
top_n_slider = st.sidebar.slider("Number of words to show", min_value=5, max_value=30, value=15)

# 5. File Upload Interface
uploaded_file = st.file_uploader("Please upload your .docx file:", type=["docx"])

if uploaded_file is not None:
    with st.spinner("Processing document pipelines..."):
        try:
            # Extract raw text
            text = extract_text_from_docx(uploaded_file)
            
            if not text.strip():
                st.error("The uploaded file does not contain any readable text paragraphs.")
                st.stop()

            # Clean and tokenize (KEEP CASE intact for part-of-speech tagging accuracy)
            tokens = nltk.word_tokenize(text)
            words = [word for word in tokens if word.isalpha()]
            tagged_words = nltk.pos_tag(words)

            # Define stop words list
            stop_words = set(stopwords.words('english'))
            custom_filters = {'also', 'would', 'could', 'may', 'one', 'two'}
            stop_words.update(custom_filters)

            filtered_words = []
            adjectives = []
            adverbs = []

            # Filter data using safe lowercase string checking
            for word, tag in tagged_words:
                lower_word = word.lower()
                if lower_word not in stop_words:
                    filtered_words.append(lower_word)
                    
                    # Identify adjectives (JJ) or adverbs (RB)
                    if tag.startswith(('JJ')):
                        adjectives.append(lower_word)
                    if tag.startswith(('RB')):
                        adverbs.append(lower_word)

            # Count overall frequencies
            word_frequency = Counter(filtered_words)
            adjective_frequency = Counter(adjectives)
            adverb_frequency = Counter(adverbs)

            # Generate the charts
            st.subheader("📊 Visualization Analysis Framework")
            
            plot_top_frequencies(
                word_count_frequency=word_frequency,
                top_n=top_n_slider,
                title=f"Top {top_n_slider} words in document",
                palette='rocket'
            )

            st.markdown("---")

            plot_top_frequencies(
                word_count_frequency=adjective_frequency,
                top_n=top_n_slider,
                title=f"Top {top_n_slider} adjectives in document",
                palette='viridis'
            )

            st.markdown("---")

            plot_top_frequencies(
                word_count_frequency=adverb_frequency,
                top_n=top_n_slider,
                title=f"Top {top_n_slider} adjectives in document",
                palette='viridis'
            )
          

        except Exception as e:
            st.error(f"An unexpected tracking error occurred: {e}")
else:
    st.info("System Standby: Awaiting a valid document to begin chart rendering cycles.")
