import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import spacy
from collections import Counter
from string import punctuation

from sentiment_plotting import *
from text_summarization import *

st.title("Text Analysis of Comments on BlackPink's latest video \"How You Like That\"")
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
all_solo = load_data()
pattern = ['jennie', 'jisoo', 'rose', 'lisa', 'yg']

st.markdown(
    "Between June 26th and July 20th, I managed to collect more than **a million** comments from BlackPink's latest  "
    "video, \"How You Like That\".")
st.write("After preparing the data, I ultimately ended up with 200,000 comments to work with.")

st.write("With these comments, I wanted to answer several questions:")
st.markdown(">*Which member is most talked about?  \n"
            "What is the general sentiment towards each member?  \n"
            "What are some of the things they're receiving praise for?  \n"
            "Some of the things they've been criticized for?*")
st.markdown("## **Which member is the most talked about?** ")
st.plotly_chart(plot_total_comments(all_solo, pattern))
st.markdown("## **What is the general sentiment towards each member?**")

total_view_option = st.radio(
    "See share of positive and negative comments as is or as ratio",
    ('As is', 'Ratio')
)
st.plotly_chart(pos_to_neg(all_solo, pattern, total_view_option))

st.markdown(
    "Lisa has almost the same number of comments as the sum of Jisoo's and Jennie's! I was very surprised to find out  "
    "that Rose would receive the least number of comments. Then again, I admit that I don't really know too much  "
    "about each member's popularity. And despite these numbers, it's safe to say that all members of BlackPink are  "
    "household names. In terms of raw numbers, Lisa also has the greatest number of negative comments. If you view  "
    "the ratio option, however, you'll see that YG has the greatest proportion of negative to positive comments."
)
st.markdown(
    "Note that the number of positive and negative comments do not add up to the total number of comments  "
    "for each member. This is because social media texts are inherently noisy, and although the library we used for  "
    "sentiment analysis did a great job, it wasn't able to predict the sentiment of every word. Since the library  "
    "handles these out-of-vocabulary words by treating them as neutral words, it's likely that including neutral  "
    "comments might misrepresent the actual number of positive, neutral, and negative comments."
)
st.write("Thus, for all graphs, the underlying data consists of the positive and negative comments only.")

st.markdown("## **What about sentiment over time?**")

member_time_option = st.radio(
    "Choose a member, or the company:",
    ('Jennie', 'Jisoo', 'Lisa', 'Rose', 'YG'),
    key='member_time_option'
)

view_option = st.radio(
    "Choose whether you want to see positive and negative comments as is or as ratio",
    ('As is', 'Ratio')
)

select_time_interval = st.radio(
    "Select a time interval:",
    ('10 min', '1 hr', '6 hr', '12 hr', '1 day', '1 week')
)

time_range = generate_time_range_slider(select_time_interval)
fig = plot_pos_vs_neg_over_time(all_solo, member_time_option, view_option, time_range, select_time_interval)
st.plotly_chart(fig)
st.markdown("Note that the distribution of comments also follows the power law; the majority of the comments can be  "
            "found in the first three days to week since the video has been posted.")

st.markdown("## **A Closer Look**")

member_topic_option = st.radio(
    "Choose a member, or the company:",
    ('Jennie', 'Jisoo', 'Lisa', 'Rose', 'YG'),
    key='member_topic_option'
)

if st.checkbox(f'Show me the machine generated topics for {member_topic_option}'):
    get_topics_for_member(all_solo, member_topic_option)

topic_of_interest = st.text_input("Your topic(s) of interest: \
(you can query multiple topics! Just separate each topic by a comma and space!)", "🔥, hair, 爱")

sample_and_plot_topic_for_member(all_solo, topic_of_interest, member_topic_option)

# get_top_sentences(all_solo, topic_of_interest, member_topic_option)

st.markdown("## **Closing Remarks**")
st.markdown("Remember that people can be inconsistent when it comes to sentiment analysis, and machine approaches  "
            "are no different. We also need to consider the domain under which words are used. For example, typically  "
            "the word 'bias' should be considered negative. But for kpop, and fandoms in general, bias instead means  "
            "the member you favor the most in a group. This explains why the sentiment for 'bias' is roughly evenly  "
            "split. The word 'bias' is usually found with other positive words, but if it stands on its own, it will  "
            "make a statement negative.")
st.markdown("From the sentiment analysis, to the topic modeling, to the text summarization, all of these analyses  "
            "were done with baseline tools. In the near future, I would like to use more cutting edge solutions to  "
            "get more accurate results. However, the quality of data is important as well. Since social media texts  "
            "are inherently very messy, it's possible that baseline methods would've worked well, but got hampered by  "
            "noise. Most of these concerns would be resolved if, for example, people used correct syntax and grammar  "
            "on YouTube.")


