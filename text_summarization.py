import spacy
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

@st.cache
def get_ngrams(text, n=1, stop_words=None):
    """
    :param text: iterable of strings
    :param n:    number of words for n-gram
    :param stop_words: stop words to remove, if any
    :return:     n-gram, as specified by n
    """
    if stop_words:
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word', stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')

    sparse = vectorizer.fit_transform(text)
    frequencies = np.sum(sparse, axis=0).T
    return pd.DataFrame(frequencies, index=vectorizer.get_feature_names(), columns=['frequency'])


@st.cache(allow_output_mutation=True)
def vectorize(df, member='jennie', type='tf'):
    """
    The comments were already pre-processed, and a column was created consisting of just noun
    :param df: the dataframe
    :param member: target member
    :param type: plain count or tfidf
    :return: vectorized strings and vectorizer
    """
    try:
        just_nouns = df[(df[member] == True) & \
                        (df['text_length'] > 10) & \
                        (df['source_lang'] == 'en')]['just_nouns'].fillna('')
    except:
        raise ValueError("Could not produce dataset")

    if type == 'tf':
        tf_vectorizer = CountVectorizer()
    else:
        tf_vectorizer = TfidfVectorizer()
    tf = tf_vectorizer.fit_transform(
        just_nouns
    )

    return tf, tf_vectorizer




@st.cache
def fit_lda(n_components, tf):
    lda = LatentDirichletAllocation(n_components=n_components,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    return lda


@st.cache
def fit_nmf(n_components, tfidf):
    nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd', max_iter=500).fit(tfidf)
    return nmf


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        st.write(message)
    st.write()



def get_topics_for_member(df, member, type='tfidf'):
    # remember to lower
    try:
        vectorized, vectorizer = vectorize(df, member.lower(), type)
        n_components = 8
        n_top_words = 8
        nmf = fit_nmf(n_components, vectorized)
        print_top_words(nmf, vectorizer.get_feature_names(), n_top_words)
    except:
        st.write("Unsuccessful in finding topics :( ")


@st.cache
def query_comments(df, topic, member, text_length=1):
    try:
        return df[
            (df['Original Comment'].str.contains("|".join(topic.split(", ")), case=False)) & \
            (df[member.lower()] == True) & \
            (df['source_lang'] == 'en') & \
            (df['text_length'] >= text_length)

            ]
    except:
        return pd.DataFrame()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_nlp():
    nlp = spacy.load('en_core_web_md')
    return nlp


@st.cache
def top_sentence(text, limit):
    # create spacy nlp doc out of given text
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    nlp = load_nlp()
    doc = nlp(text.lower())

    # create list of all words
    keyword = [token.lemma_ for token in doc
               if token.pos_ in pos_tag
               and token.text not in nlp.Defaults.stop_words
               and token.text not in punctuation]

    # generate term-frequency counts
    freq_word = Counter(keyword)
    max_freq = freq_word.most_common(1)[0][1]

    # normalize term frequencies by max_freq
    for w in freq_word:
        freq_word[w] = (freq_word[w] / max_freq)

    # build idf
    num_sentences = 0
    idf = {}
    for sent in doc.sents:
        seen = set()
        num_sentences += 1
        for word in sent:
            # only add words once, even if they appear multiple times in sentence
            if word.lemma_ not in seen:
                if word.lemma_ not in idf:
                    idf[word.lemma_] = 1
                else:
                    idf[word.lemma_] += 1
                seen.add(word.lemma_)

    # combine the normalize(idf) and tfidf calculation step
    tfidf = {}
    for w in freq_word:
        tfidf[w] = freq_word[w] * (np.log((num_sentences + 1) / (idf[w] + 1)) + 1)

    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.lemma_ in tfidf.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += tfidf[word.lemma_]
                else:
                    sent_strength[sent] = tfidf[word.lemma_]

    summary = []

    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)

    i = 0
    while i < len(sorted_x) and i < limit:
        summary.append(" ".join(str(sorted_x[i][0]).capitalize().split()))
        i += 1

    return summary


def get_top_sentences(df, topic, member, text_length=15):
    data = query_comments(df, topic, member, text_length)
    if data.empty:
        st.write(f"Couldn't generate summary on *{topic}* for *{member}* :(")
    else:
        sentences = pd.DataFrame(top_sentence(". ".join(data['Original Comment'].tolist()), 5),
                                 columns=[f"5 Sentence Machine-Generated Summary on {topic} for {member}"]
                                 )
        st.table(sentences)