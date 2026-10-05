import streamlit as st
import docx
import re
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import spacy
from spacy.cli import download

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
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

#download_nltk_data()

def setup_spaCy(model):
    nlp = spacy.load(model)
    nlp.add_pipe("merge_entities")
    return nlp

nlp = setup_spaCy("en_core_web_md")

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

def extract_and_count_frequencies(text):
    doc = nlp(text)
    
    words = []
    proper_nouns = []
    pronouns = []          # Added for pronoun extraction
    verbs = []
    gerunds=[]
    adjectives = []
    adjectival_phrases = []
    adverbs = []
    adverbial_phrases = []
    
    # 1. Extract Proper Nouns (using spaCy's named entities or Part-of-Speech tags)
    for ent in doc.ents:
        proper_nouns.append(ent.text.strip())
        
    for token in doc:
        if token.pos_ == "PROPN" and not token.ent_type_:
            proper_nouns.append(token.text)

    words = [token.text for token in doc if token.is_alpha and not token.is_stop]
    
    # 2. Extract Verbs, Pronouns, and Phrases
    for token in doc:
     
        # --- PRONOUN EXTRACTION ---
        # POS tag 'PRON' captures personal (he/she), relative (who/which), and demonstrative (this/that)
        # POS tag 'DET' combined with possessive pronouns (my/their/your) captures possessive determiners
        if token.pos_ == "PRON" or (token.pos_ == "DET" and token.morph.get("Poss", [""])[0] == "Yes"):
            pronouns.append(token.text.lower())

        # --- VERB EXTRACTION ---
        elif token.pos_ in ["VERB", "AUX"] and not token.is_stop:
            verbs.append(token.text.lower())

        elif token.pos_ == "NOUN" and token.tag_ == "VBG" and not token.is_stop:
            gerunds.append(token.text.lower())

        # --- ADJECTIVAL PHRASE EXTRACTION ---
        elif token.pos_ == "ADJ" and not token.is_stop:
            adjectives.append(token.text.lower())
            phrase_tokens = [child for child in token.children if child.dep_ == "advmod"] + [token]
            if len(phrase_tokens) > 1:
                phrase_text = " ".join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
                adjectival_phrases.append(phrase_text)
                
        # --- ADVERBIAL PHRASE EXTRACTION ---
        elif token.pos_ == "ADV" and not token.is_stop:
            adverbs.append(token.text.lower())
            phrase_tokens = [child for child in token.children if child.dep_ == "advmod"] + [token]
            if len(phrase_tokens) > 1:
                phrase_text = " ".join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
                adverbial_phrases.append(phrase_text)

    lyadverbs = [ w for w in adverbs if w.endswith("ly") ]
    # Compute Frequencies

    all_words_count = Counter(words) if words.count else Counter()
    proper_nouns_count = Counter(proper_nouns) if proper_nouns.count else Counter()
    pronoun_counts = Counter(pronouns) if pronouns.count else Counter()            # Added to return dict
    verb_counts = Counter(verbs) if verbs.count else Counter()
    gerund_counts = Counter(gerunds) if gerunds.count else Counter()
    adverb_counts = Counter(adverbs) if adverbs.count else Counter()
    lyadverbs_counts = Counter(lyadverbs) if lyadverbs.count else Counter()
    adverbial_phrase_counts = Counter(adverbial_phrases) if adverbial_phrases.count else Counter()
    adjectives_counts = Counter(adjectives) if adjectives.count else Counter()
    adjectivial_phrase_counts = Counter(adjectival_phrases) if adjectival_phrases.count else Counter()

    reduced_words_count = Counter()
    if all_words_count:
        reduced_words_count += all_words_count
        reduced_words_count -= proper_nouns_count
        reduced_words_count -= pronoun_counts
        reduced_words_count -= verb_counts
        reduced_words_count -= adverb_counts
        reduced_words_count += lyadverbs_counts
    
    return {
        "All": all_words_count,
        "All - Proper Nouns - Pronouns - Verbs - Adverbs + Adverbs ending in 'ly'": reduced_words_count,
        "Proper Nouns": proper_nouns_count,
        "Pronouns": pronoun_counts,
        "Verbs": verb_counts,
        "Gerunds": gerund_counts,
        "Adverbs": adverb_counts,
        "'ly' Adverbs": lyadverbs_counts,
        "Adverbial Phrases": adverbial_phrase_counts,
        "Adjectives": adjectives_counts,
        "Adjectival Phrases": adjectivial_phrase_counts
    }

def nltk_freqeuncy_counter(text):
    
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
                # --- FIX FOR NOUN ADJUNCTS ---
                # Check synsets to see how the word is classified in WordNet
                synsets = wordnet.synsets(lower_word)
                
                # If WordNet knows the word, check if it can even function as an adjective
                if synsets:
                    has_adj_sense = any(s.pos() in ('a', 's') for s in synsets)
                    has_noun_sense = any(s.pos() == 'n' for s in synsets)
                    
                    # If it's a known noun but has NO recorded adjective senses (like "temple"),
                    # skip adding it to the adjective frequency counter.
                    if has_noun_sense and not has_adj_sense:
                        continue  # Treat it as a noun adjunct, not a true adjective
                # -----------------------------
                adjectives.append(lower_word)
            
            if tag.startswith('RB') or tag == 'WRB':
                adverbs.append(lower_word)

    # Count overall frequencies
    
    return {
        "All Words": Counter(filtered_words),
        "Adjectives": Counter(adjectives),
        "Adverbs": Counter(adverbs)
    }

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

            frequencies = extract_and_count_frequencies(text)
            #frequencies = nltk_freqeuncy_counter(text) #inactivated based on nltk

            # Generate the charts
            st.subheader("📊 Visualization Analysis Framework")
            
            for category, counts in frequencies.items():

                if not counts:
                    continue

                plot_top_frequencies(
                    word_count_frequency=counts,
                    top_n=top_n_slider,
                    title=f"Top {top_n_slider} {category} in document",
                    palette='rocket'
                )
                st.markdown("---")

        except Exception as e:
            st.error(f"An unexpected tracking error occurred: {e}")
else:
    st.info("System Standby: Awaiting a valid document to begin chart rendering cycles.")
