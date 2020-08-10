import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from text_summarization import query_comments


@st.cache
def load_data():
    """
    Loads a dataframe from memory that has been preprocessed such that each comment contains mention of only one of
    the member columns
    :return: a pandas dataframe
    """
    return pd.read_csv('all_pos_neg.csv', lineterminator='\n',
                       parse_dates=['Updated At'],
                       date_parser=lambda x: pd.to_datetime(x),
                       index_col='Updated At')

    return all


@st.cache
def get_by_sentiment(df, target_sentiment=1):
    return df.query(f'sentiment_category == {target_sentiment}')


@st.cache
def get_count_of_comments(df, pattern):
    """
    Because each row mentions exactly one member, we can just sum the boolean columns
    :param df: dataframe containing pattern boolean columns corresponding to whether the comment mentions one in pattern
    :param pattern: a list of boolean columns
    :return: sum of frequencies of truths in boolean columns in dataframe form
    """
    return df[pattern].sum().sort_values(ascending=False).to_frame()


def plot_total_comments(df, pattern):
    count = get_count_of_comments(df, pattern)
    comments = px.bar(x=count.index,
                      y=count,
                      color_discrete_sequence=['pink'],
                      title="Comments per Member or YG"
                      )
    comments.update_xaxes(title="Member or YG")
    comments.update_yaxes(title="Number of Comments")
    comments.update_layout(plot_bgcolor='rgb(0, 0, 0)',
                           xaxis_showgrid=False, yaxis_showgrid=False)
    return comments


def plot_pos_to_neg(x_vals, pos_vals, neg_vals):
    """
    Helper function to plot two dataframes against each other
    :param x_vals:   x-axis values, discrete and categorical, for each member
    :param pos_vals: count of positive comments for each x_val
    :param neg_vals: count of negative comments for each x_val
    :return: a plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(name="Positive Comments",
               x=x_vals,
               y=pos_vals,
               marker_color='pink'
               ),
        go.Bar(name="Negative Comments",
               x=x_vals,
               y=neg_vals,
               marker_color='black'
               )
    ])

    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)', xaxis_showgrid=False, yaxis_showgrid=False,
    )
    return fig


@st.cache
def pos_to_neg(df, pattern, view_option='Ratio'):
    positives = get_by_sentiment(df, 1)
    negatives = get_by_sentiment(df, -1)

    pos_vals = positives[pattern].sum().sort_values(ascending=False)

    # use pos_index to re-order negatives
    pos_index = pos_vals.index
    neg_vals = negatives[pattern].sum()[pos_index]

    if view_option == 'As is':
        sum = 1
    else:
        sum = (positives[pattern].sum() + negatives[pattern].sum())[pos_index]

    return plot_pos_to_neg(pos_index,
                           (pos_vals / sum).tolist(),
                           (neg_vals / sum).tolist()
                           )


@st.cache
def get_sentiment_isolated(df, option='Lisa', target_sentiment=1):
    """
    Query for the desired sentiment, and then select comments that mention only one member.
    From this, select the column corresponding to option

    :param option: target member
    :param target_sentiment: desired sentiment
    :return: returns a dataframe, replacing boolean columns with 0 or 1 to signify whether the row has given sentiment
    """
    if target_sentiment == 1:
        label = "positive"
    elif target_sentiment == -1:
        label = "negative"
    else:
        label = "neutral"

    sentiment = get_by_sentiment(df, target_sentiment)
    return sentiment[option.lower()] \
        .astype('float64') \
        .to_frame() \
        .rename({option.lower(): f'{label} comments for {option.lower()}'}, axis=1)


@st.cache
def resample_by(pos, neg, time='10min'):
    return pos.resample(time).sum(), neg.resample(time).sum()


def plot_pos_vs_neg_over_time(df,
                              member_option='Jennie',
                              view_option='As is',
                              time_range=None,
                              time_interval='1h'):
    time_intervals = {
        '10 min': '10min',
        '1 hr': '1h',
        '6 hr': '6h',
        '12 hr': '12h',
        '1 day': '1d',
        '1 week': '1w'
    }
    pos_subset = get_sentiment_isolated(df, member_option, target_sentiment=1)
    neg_subset = get_sentiment_isolated(df, member_option, target_sentiment=-1)
    pos_subset, neg_subset = resample_by(pos_subset, neg_subset, time_intervals.get(time_interval))
    joined = pd.concat([neg_subset, pos_subset], axis=1)

    if time_range:
        # restrict given time range to indices that actually exist in the dataframe
        joined = joined.loc[joined.index[joined.index.get_loc(time_range[0], method='nearest')]: \
                            joined.index[joined.index.get_loc(time_range[1], method='nearest')]
                 ]

    if view_option == 'Ratio':
        sum = joined.sum(axis=1)
    else:
        sum = 1

    joined[f'positive comments for {member_option.lower()}'] = joined[
                                                                   f'positive comments for {member_option.lower()}'
                                                               ] / sum
    joined[f'negative comments for {member_option.lower()}'] = joined[
                                                                   f'negative comments for {member_option.lower()}'
                                                               ] / sum

    fig = px.bar(joined,
                 color_discrete_sequence=['black', 'pink'],
                 title=f'Positive to Negative Sentiment for {member_option} over Time',
                 height=600,
                 width=1000,
                 )

    fig.update_xaxes(title="Time",
                     )

    fig.update_yaxes(title="Number of Comments")
    fig.update_layout(hovermode="x",
                      plot_bgcolor='rgb(255, 255, 255)', xaxis_showgrid=False, yaxis_showgrid=False,
                      )
    return fig


def generate_time_range_slider(time_interval):
    if time_interval == '10 min':
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 6, 28, 00, 00))
    elif time_interval in ['1 hr', '6 hr', '12 hr']:
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 6, 30, 00, 00))
    elif time_interval == '1 day':
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 7, 6, 00, 00))
    else:
        value = (datetime(2020, 6, 25, 00, 00), datetime(2020, 7, 21, 00, 00))

    return st.slider(
        label="Time Range",
        min_value=datetime(2020, 6, 30, 00, 00),
        value=value,
        max_value=datetime(2020, 7, 21, 00, 00),
        step=timedelta(hours=6),
        format="MM/DD/YY - hh:mm a"
    )


def pos_to_neg_topic_for_member(df, topic, member):
    positives = get_sentiment_isolated(df, member[0], 1)
    negatives = get_sentiment_isolated(df, member[0], -1)

    pos_vals = positives.sum()[0]
    neg_vals = negatives.sum()[0]
    sum = pos_vals + neg_vals

    fig = plot_pos_to_neg(member,
                          [pos_vals / sum],
                          [neg_vals / sum]
                          )
    fig.update_layout(
        title=f"Sentiment for All Comments Related to '{topic}' for {member[0]}"
    )
    return fig


def sample_and_plot_topic_for_member(df, topic, member, text_length=1):
    data = query_comments(df, topic=topic, member=member, text_length=text_length)
    if data.empty:
        st.write(f"Couldn't find anything on *{topic}* for *{member}* :(")
    else:
        st.markdown(f"Found {data.shape[0]} comments regarding **{topic}** for **{member}**. Displaying a few now:")
        if data.shape[0] < 3:
            st.table(data[['Original Comment', 'sentiment_category']])
        else:
            st.table(data.sample(3)[['Original Comment', 'sentiment_category']])
        st.plotly_chart(pos_to_neg_topic_for_member(data, topic, [member]))
