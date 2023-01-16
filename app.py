

import pandas as pd
import os
import plotly.graph_objects as go
import streamlit as st
import random



dirname = os.path.dirname(__file__)
filename_text_df = os.path.join(dirname, 'text_info.csv')
filename_topic_df = os.path.join(dirname, 'topic_info.csv')


random.seed(123456)

df = pd.read_csv(filename_text_df)

df = df.sample(n=2_000).reset_index(drop=True)

topic_df = pd.read_csv(filename_topic_df)


WIDTH = 1200
HEIGHT = 750

unique_topics = sorted(list(set(df['topic'])))

topic_name_dict = topic_df.set_index('topic')['name'].to_dict()

names = ['']
for row, info in topic_df.sort_values('topic').iterrows():
    if info['topic'] >= 0:
        if info['name'] == info['label']:
            names.append(str(info['topic']) + ': ' + info['name'])
        else:
            names.append(str(info['topic']) + ': ' + '-'.join(info['words'].split('-')[:3]))


# names = [str(x) + ': ' + topic_df.set_index('topic')['name'].to_dict().get(x) if x >= 0 else '' for x in unique_topics]
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

        if df['topic'].value_counts().to_dict().get(topic) >= 50:
            selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

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


st.title('Distribución de tópicos')
st.dataframe(topic_df.set_index('topic')[['name', 'size', 'sentiment', 'positive', 'neutral', 'negative', 'words']])


data = [go.Bar(
   x=[str(x) for x in topic_df['topic']],
   y=topic_df['size'],
)]

fig = go.Figure(data=data)
st.plotly_chart(fig)
