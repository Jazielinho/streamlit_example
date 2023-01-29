

import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import random
from sklearn.preprocessing import normalize
from streamlit_tags import st_tags



dirname = os.path.dirname(__file__)
# dirname = 'C:/Users/User/Downloads/streamlit_example/'
filename_text_df = os.path.join(dirname, 'text_info_df.csv')
filename_topic_df = os.path.join(dirname, 'topics_info_df.csv')
filename_ner_df = os.path.join(dirname, 'ner_df.csv')
filename_sentiment_df = os.path.join(dirname, 'sentiment_df.csv')
filename_topics_over_time_df = os.path.join(dirname, 'topics_over_time_df.csv')


random.seed(123456)

df = pd.read_csv(filename_text_df)

if len(df) >= 10_000:
    df = df.sample(n=10_000).reset_index(drop=True)



topic_df = pd.read_csv(filename_topic_df)
ner_df = pd.read_csv(filename_ner_df)
sentiment_df = pd.read_csv(filename_sentiment_df)
topics_over_time_df = pd.read_csv(filename_topics_over_time_df)

ner_df['text'] = ner_df['text'].str.lower()

WIDTH = 1000
HEIGHT = 600

unique_topics = sorted(list(set(df['topic'])))

topic_name_dict = topic_df.set_index('topic')['name'].to_dict()

names = ['']
for row, info in topic_df.sort_values('topic').iterrows():
    if info['topic'] >= 0:
        if info['coherence'] >= 0.6:
            names.append(str(info['topic']) + ': ' + info['label'])
        else:
            names.append(str(info['topic']) + ': ' + '-'.join(info['words'].split('-')[:5]))

# if -1 not in unique_topics:
# names = names[1:]

dict_names = {enum: k for enum, k in enumerate(names[1:])}
dict_names[-1] = ''

df = df[df['topic'] != -1]

#====================================================================================================================================
st.title('Representación de tópicos')

_df = df[df['topic'] == -1]
fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=_df.x,
        y=_df.y,
        hovertext=_df.doc,
        hoverinfo="text",
        mode='markers+text',
        name="other",
        showlegend=False,
        marker=dict(color='#CFD8DC', size=5, opacity=0.5)
    )
)

for name, topic in zip(names, unique_topics):
    if int(topic) != -1:
        selection = df.loc[df.topic == topic, :]
        selection["text"] = ""

        # if df['topic'].value_counts().to_dict().get(topic) >= 100:
        #     selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), name]

        fig.add_trace(
            go.Scattergl(
                x=selection.x,
                y=selection.y,
                hovertext=selection.doc,
                hoverinfo="text",
                text=selection.text,
                mode='markers+text',
                name=name,
                textfont=dict(
                    size=12,
                ),
                marker=dict(size=5, opacity=0.5)
            )
        )

# Add grid in a 'plus' shape
x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
fig.add_shape(type="line",
              x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
              line=dict(color="#CFD8DC", width=2))
fig.add_shape(type="line",
              x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
              line=dict(color="#9E9E9E", width=2))
fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

# Stylize layout
fig.update_layout(
    template="simple_white",
    title={
        'text': "<b>Documents and Topics",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=22,
            color="Black")
    },
    width=WIDTH,
    height=HEIGHT
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

st.plotly_chart(fig)

#====================================================================================================================================


st.title('Distribución de tópicos')

data = [go.Bar(
   x=[dict_names[x] for x in topic_df['topic']],
   y=topic_df['size'],
)]

fig = go.Figure(data=data)
fig.update_layout(width=WIDTH, height=HEIGHT)
st.plotly_chart(fig)

#====================================================================================================================================
st.title('Evolución de tópicos')

normalize_frequency = st.radio('Normalizar', ('Si', 'No')) == 'Si'

colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
data = topics_over_time_df.sort_values(["Topic", "Timestamp"])
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data["Name"] = data.Topic.map(dict_names)

# Add traces
fig = go.Figure()
for index, topic in enumerate(data.Topic.unique()):
    trace_data = data.loc[data.Topic == topic, :]
    topic_name = trace_data.Name.values[0]
    words = trace_data.Words.values
    if normalize_frequency:
        y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
    else:
        y = trace_data.Frequency
    fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                             mode='lines+markers',
                             marker_color=colors[index % 7],
                             hoverinfo="text",
                             name=topic_name,
                             hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

# Styling of the visualization
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)
fig.update_layout(
    yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
    title={
        'text': "<b>Topics over Time",
        'y': .95,
        'x': 0.40,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=22,
            color="Black")
    },
    template="simple_white",
    width=WIDTH,
    height=HEIGHT,
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    ),
    legend=dict(
        title="<b>Global Topic Representation",
    )
)
st.plotly_chart(fig)

#====================================================================================================================================
st.title('Sentiment por tópicos')

global_sent_df = pd.DataFrame(sentiment_df[['positive_score', 'neutral_score', 'negative_score']].agg('mean')).transpose()
global_sent_df.index = ['global']
topic_sent_df = sentiment_df.groupby('labels')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
topic_sent_df.index = [dict_names[x] for x in topic_sent_df.index]
topic_sent_df = topic_sent_df[topic_sent_df.index != ''].sort_index()
sent_df = pd.concat([global_sent_df, topic_sent_df], axis=0).round(2)

fig = go.Figure(data=[
    go.Bar(name='Positive', x=sent_df.index.to_list(), y=sent_df['positive_score'].to_list()),
    go.Bar(name='Neutral', x=sent_df.index.to_list(), y=sent_df['neutral_score'].to_list()),
    go.Bar(name='Negative', x=sent_df.index.to_list(), y=sent_df['negative_score'].to_list())
])
# Change the bar mode
fig.update_layout(barmode='stack')
fig.update_layout(width=WIDTH, height=HEIGHT)
st.plotly_chart(fig)

#====================================================================================================================================
st.title('PERSONAS')

persons_textids_df = pd.DataFrame(ner_df[ner_df['type'] == 'PERSON'].groupby('text')['text_id'].agg(lambda x: list(set(x))))
persons_textids_df.columns = ['text_id']
persons_textids_df['count'] = persons_textids_df['text_id'].apply(lambda x: len(x))
persons_textids_df = persons_textids_df.sort_values('count', ascending=False).head(30)
list_person_statistics = []
for row, info in persons_textids_df.iterrows():
    sent_mean_df = sentiment_df[sentiment_df['text_id'].isin(info['text_id'])].mean().round(2).to_dict()
    list_person_statistics.append({
        'person': row,
        'cases': info['count'],
        'positive': sent_mean_df['positive_score'],
        'neutral': sent_mean_df['neutral_score'],
        'negative': sent_mean_df['negative_score'],
    })
person_statistics_df = pd.DataFrame(list_person_statistics)


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(name='Positive', x=person_statistics_df['person'].to_list(), y=person_statistics_df['positive'].to_list()))
fig.add_trace(go.Bar(name='Neutral', x=person_statistics_df['person'].to_list(), y=person_statistics_df['neutral'].to_list()))
fig.add_trace(go.Bar(name='Negative', x=person_statistics_df['person'].to_list(), y=person_statistics_df['negative'].to_list()))
fig.add_trace(go.Scatter(name='Cantidad', x=person_statistics_df['person'], y=person_statistics_df['cases']), secondary_y=True)
# Change the bar mode
fig.update_layout(barmode='stack')
fig.update_layout(width=WIDTH, height=HEIGHT)
st.plotly_chart(fig)


#====================================================================================================================================
st.title('PERSONA')
personas_elegidas = st_tags(suggestions=persons_textids_df.index.to_list(), label='Ingresa persona')
if st.button('ejecutar personas'):
    for persona_elegida in personas_elegidas:
        if persona_elegida not in persons_textids_df.index:
            continue
        st.subheader(persona_elegida)
        persona_textids = persons_textids_df[persons_textids_df.index == persona_elegida]['text_id'].to_list()[0]

        _sentiment_df = sentiment_df[sentiment_df['text_id'].isin(persona_textids)]
        global_sent_df = pd.DataFrame(_sentiment_df[['positive_score', 'neutral_score', 'negative_score']].agg('mean')).transpose()
        global_sent_df.index = ['global']
        global_sent_df['size'] = _sentiment_df.shape[0]
        topic_sent_df = _sentiment_df.groupby('labels')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
        topic_sent_df['size'] = [_sentiment_df.groupby('labels').size().to_dict()[x] for x in topic_sent_df.index]
        topic_sent_df.index = [dict_names[x] for x in topic_sent_df.index]
        topic_sent_df = topic_sent_df[topic_sent_df.index != ''].sort_index()
        sent_df = pd.concat([global_sent_df, topic_sent_df], axis=0).round(2)

        sent_df = sent_df[sent_df['size'] >= 30]
        sent_df = sent_df.sort_values('size', ascending=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name='Positive', x=sent_df.index.to_list(), y=sent_df['positive_score'].to_list()))
        fig.add_trace(go.Bar(name='Neutral', x=sent_df.index.to_list(), y=sent_df['neutral_score'].to_list()))
        fig.add_trace(go.Bar(name='Negative', x=sent_df.index.to_list(), y=sent_df['negative_score'].to_list()))
        fig.add_trace(go.Scatter(name='Cantidad', x=sent_df.index, y=sent_df['size']), secondary_y=True)
        # Change the bar mode
        fig.update_layout(barmode='stack')
        fig.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig)


        new_df = df[df['text_id'].isin(persona_textids)]
        new_df = new_df[new_df['topic'] != -1]
        _df = new_df[new_df['topic'] == -1]
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=_df.x,
                y=_df.y,
                hovertext=_df.doc,
                hoverinfo="text",
                mode='markers+text',
                name="other",
                showlegend=False,
                marker=dict(color='#CFD8DC', size=5, opacity=0.5)
            )
        )

        for name, topic in zip(names, unique_topics):
            if int(topic) != -1:
                selection = new_df.loc[new_df.topic == topic, :]
                selection["text"] = ""

                # if len(selection) < 5:
                #     continue

                # if df['topic'].value_counts().to_dict().get(topic) >= 5:
                # selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), name]

                fig.add_trace(
                    go.Scattergl(
                        x=selection.x,
                        y=selection.y,
                        hovertext=selection.doc,
                        hoverinfo="text",
                        text=selection.text,
                        mode='markers+text',
                        name=name,
                        textfont=dict(
                            size=12,
                        ),
                        marker=dict(size=5, opacity=0.5)
                    )
                )

        # Add grid in a 'plus' shape
        x_range = (new_df.x.min() - abs((new_df.x.min()) * .15), new_df.x.max() + abs((new_df.x.max()) * .15))
        y_range = (new_df.y.min() - abs((new_df.y.min()) * .15), new_df.y.max() + abs((new_df.y.max()) * .15))
        fig.add_shape(type="line",
                      x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                      line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="line",
                      x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                      line=dict(color="#9E9E9E", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

        # Stylize layout
        fig.update_layout(
            template="simple_white",
            title={
                'text': "<b>Documents and Topics",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            width=WIDTH,
            height=HEIGHT
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        st.plotly_chart(fig)


