

import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import random
from sklearn.preprocessing import normalize
from streamlit_tags import st_tags
import plotly.express as px


#==========================================================================================================================================
st.title('''TWEETS SHAKIRA - PIQUE''')

dirname = os.path.dirname(__file__)

min_topic_size = st.radio('Mínimo tamaño del topic (para la sección de tópicos y personas)', ('500', '100'))

if min_topic_size == '500':
    filename_best_tweets_df = os.path.join(dirname, 'best_tweets_df_500.csv')
    filename_ner_df = os.path.join(dirname, 'ner_df_500.csv')
    filename_text_df = os.path.join(dirname, 'sample_text_info_df_500.csv')
    filename_sentiment_df = os.path.join(dirname, 'sentiment_df_500.csv')
    filename_timestamp_word_count_df = os.path.join(dirname, 'timestamp_word_count_df_500.csv')
    filename_topics_info_df = os.path.join(dirname, 'topics_info_df_500.csv')
    filename_topics_over_time_df = os.path.join(dirname, 'topics_over_time_df_500.csv')
    filename_topics_words_df = os.path.join(dirname, 'topics_words_500.csv')
    filename_word_count_df = os.path.join(dirname, 'word_count_df_500.csv')
    # dirname = 'C:/Users/User/Downloads/streamlit_example/'
else:
    filename_best_tweets_df = os.path.join(dirname, 'best_tweets_df_100.csv')
    filename_ner_df = os.path.join(dirname, 'ner_df_100.csv')
    filename_text_df = os.path.join(dirname, 'sample_text_info_df_100.csv')
    filename_sentiment_df = os.path.join(dirname, 'sentiment_df_100.csv')
    filename_timestamp_word_count_df = os.path.join(dirname, 'timestamp_word_count_df_100.csv')
    filename_topics_info_df = os.path.join(dirname, 'topics_info_df_100.csv')
    filename_topics_over_time_df = os.path.join(dirname, 'topics_over_time_df_100.csv')
    filename_topics_words_df = os.path.join(dirname, 'topics_words_100.csv')
    filename_word_count_df = os.path.join(dirname, 'word_count_df_100.csv')


best_tweets_df = pd.read_csv(filename_best_tweets_df)
ner_df = pd.read_csv(filename_ner_df)
text_df = pd.read_csv(filename_text_df)
sentiment_df = pd.read_csv(filename_sentiment_df)
timestamp_word_count_df = pd.read_csv(filename_timestamp_word_count_df)
topics_info_df = pd.read_csv(filename_topics_info_df)
topics_over_time_df = pd.read_csv(filename_topics_over_time_df)
topics_words_df = pd.read_csv(filename_topics_words_df)
word_count_df = pd.read_csv(filename_word_count_df)

ner_df['text'] = ner_df['text'].str.lower()
timestamp_word_count_df['date'] = pd.to_datetime(timestamp_word_count_df['date'])
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])


names = ['']
for row, info in topics_info_df.sort_values('topic').iterrows():
    if info['topic'] >= 0:
        if info['coherence'] >= 0.6:
            names.append(str(info['topic']) + ': ' + info['label'])
        else:
            names.append(str(info['topic']) + ': ' + '-'.join(info['words'].split('-')[:5]))

# if -1 not in unique_topics:
# names = names[1:]

dict_names = {enum: k for enum, k in enumerate(names[1:])}
dict_names[-1] = ''

WIDTH = 1000
HEIGHT = 600




#==========================================================================================================================================
st.markdown('''# ESTADÍSTICAS GENERALES''')

st.markdown(f'''Número de tweets: {sentiment_df.shape[0]}''')
st.markdown(f'''Periodo de tiempo: {sentiment_df.date.min().date()} - {sentiment_df.date.max().date()}''')

tab1, tab2, tab3 = st.tabs(["GENERAL", "TOPICOS", "PERSONAS"])

with tab1:
    st.markdown('''## CANTIDAD DE TWEETS POR DÍA''')
    count_day_df = sentiment_df.groupby('date').size().reset_index()
    count_day_df.columns = ['date', 'size']
    data = [go.Bar(
       x=count_day_df['date'].to_list(),
       y=count_day_df['size'].to_list(),
    )]
    fig = go.Figure(data=data)
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown('''## PRINCIPALES PALABRAS''')
    data = [go.Bar(
       x=word_count_df['word'].to_list(),
       y=word_count_df['count'].to_list(),
    )]

    fig = go.Figure(data=data)
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown('''### PRINCIPALES PALABRAS A TRAVÉS DEL TIEMPO''')
    data_plot = []
    timestamp_word_count_df = timestamp_word_count_df[timestamp_word_count_df['orden'] <= 10]
    for word in timestamp_word_count_df.groupby('word')['count'].sum().sort_values(ascending=False).index.to_list():
        _df = timestamp_word_count_df[timestamp_word_count_df['word'] == word]
        data_plot.append(
            go.Scatter(name=word, x=_df['date'], y=_df['count'])
        )
    fig = go.Figure(data=data_plot)
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown('''## ANÁLISIS DE SENTIMIENTO''')
    general_sentimiento = sentiment_df[['positive_score', 'neutral_score', 'negative_score']].mean().round(2).reset_index()
    general_sentimiento.columns = ['Sentimiento', 'Porcentaje']
    data_plot = go.Bar(x=general_sentimiento['Sentimiento'], y=general_sentimiento['Porcentaje'])
    fig = go.Figure(data=[data_plot])
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown('''### SENTIMIENTO A TRAVÉS DEL TIEMPO''')
    sentimiento_over_time_df = sentiment_df.groupby('date')[['positive_score', 'neutral_score','negative_score']].mean().reset_index()
    sentimiento_over_time_df.columns = ['date', 'positive', 'neutral', 'negative']
    fig = go.Figure(data=[
        go.Bar(name='Positive', x=sentimiento_over_time_df['date'].to_list(), y=sentimiento_over_time_df['positive'].to_list()),
        go.Bar(name='Neutral', x=sentimiento_over_time_df['date'].to_list(), y=sentimiento_over_time_df['neutral'].to_list()),
        go.Bar(name='Negative', x=sentimiento_over_time_df['date'].to_list(), y=sentimiento_over_time_df['negative'].to_list())
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)


#==========================================================================================================================================
with tab2:
    st.markdown('''# TÓPICOS''')

    _df = text_df[text_df['topic'] == -1]
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

    for topic, name in dict_names.items():
        if int(topic) != -1:
            selection = text_df.loc[text_df.topic == topic, :]
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
    x_range = (text_df.x.min() - abs((text_df.x.min()) * .15), text_df.x.max() + abs((text_df.x.max()) * .15))
    y_range = (text_df.y.min() - abs((text_df.y.min()) * .15), text_df.y.max() + abs((text_df.y.max()) * .15))
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
            'text': "<b>",
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


    st.markdown('## Distribución de tópicos')
    data = [go.Bar(
       x=[dict_names[x] for x in topics_info_df['topic']],
       y=topics_info_df['size'],
    )]

    fig = go.Figure(data=data)
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)


    st.markdown('## Evolución de tópicos')

    normalize_frequency = st.radio('Normalizar', ('Si', 'No')) == 'Si'

    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    topics_over_time_df = topics_over_time_df.sort_values(["Topic", "Timestamp"])
    topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])
    topics_over_time_df["Name"] = topics_over_time_df.Topic.map(dict_names)

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(topics_over_time_df.Topic.unique()):
        trace_data = topics_over_time_df.loc[topics_over_time_df.Topic == topic, :]
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
            'text': "<b>",
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
            title="<b>Tópicos",
        )
    )
    st.plotly_chart(fig)


    st.markdown('## Sentimiento por tópicos')
    topic_sent_df = sentiment_df.groupby('labels')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
    topic_sent_df.index = [dict_names[x] for x in topic_sent_df.index]
    topic_sent_df = topic_sent_df[topic_sent_df.index != ''].sort_index()

    fig = go.Figure(data=[
        go.Bar(name='Positive', x=topic_sent_df.index.to_list(), y=topic_sent_df['positive_score'].to_list()),
        go.Bar(name='Neutral', x=topic_sent_df.index.to_list(), y=topic_sent_df['neutral_score'].to_list()),
        go.Bar(name='Negative', x=topic_sent_df.index.to_list(), y=topic_sent_df['negative_score'].to_list())
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig)


    st.markdown('# Información por tópico')
    elige_topico = st.multiselect('Elige el tópico que tienes interés', [x for x in dict_names.keys() if x != -1], [0, 1])
    if st.button('Ejecutar'):

        for topico_analysis in elige_topico:
            st.markdown(f'''## Topico: {topico_analysis}''')
            st.markdown(f'''### Nombre: {dict_names[int(topico_analysis)]}''')
            fig = go.Figure()
            _df = text_df[text_df['topic'] != int(topico_analysis)]
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
            selection = text_df.loc[text_df.topic == int(topico_analysis), :]
            selection["text"] = ""
            selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), dict_names[int(topico_analysis)]]
            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=dict_names[int(topico_analysis)],
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )
            # Add grid in a 'plus' shape
            x_range = (text_df.x.min() - abs((text_df.x.min()) * .15), text_df.x.max() + abs((text_df.x.max()) * .15))
            y_range = (text_df.y.min() - abs((text_df.y.min()) * .15), text_df.y.max() + abs((text_df.y.max()) * .15))
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
                    'text': "<b>",
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

            st.markdown(f'''### Palabras más importantes de Tópico''')
            _topics_words_df = topics_words_df[topics_words_df['topic'] == int(topico_analysis)]
            data = [go.Bar(
                x=_topics_words_df['word'].to_list(),
                y=_topics_words_df['peso'].to_list(),
            )]

            fig = go.Figure(data=data)
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig)

            st.markdown(f'''### Tweets más importantes del Tópico ''')
            st.dataframe(best_tweets_df[best_tweets_df['topic'] == int(topico_analysis)]['doc'])

            st.markdown(f'''### Evolución de cantidad de tweets y sentimiento''')
            _sentiment_df = sentiment_df[sentiment_df['labels'] == int(topico_analysis)]
            topic_sent_df = _sentiment_df.groupby('date')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
            _size_df = _sentiment_df.groupby('date').size()
            topic_sent_df = pd.concat([topic_sent_df, _size_df], axis=1)
            sent_df = topic_sent_df.round(2)
            sent_df = sent_df.sort_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(name='Positive', x=sent_df.index.to_list(), y=sent_df['positive_score'].to_list()))
            fig.add_trace(go.Bar(name='Neutral', x=sent_df.index.to_list(), y=sent_df['neutral_score'].to_list()))
            fig.add_trace(go.Bar(name='Negative', x=sent_df.index.to_list(), y=sent_df['negative_score'].to_list()))
            fig.add_trace(go.Scatter(name='Cantidad', x=sent_df.index, y=sent_df[0]), secondary_y=True)
            # Change the bar mode
            fig.update_layout(barmode='stack')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig)


#====================================================================================================================================
with tab3:
    st.markdown('# Principales Personas')

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
    st.markdown('## Información por persona')
    personas_elegidas = st.multiselect('Elige la persona que tienes interés', persons_textids_df.index, [persons_textids_df.index[0]])
    if st.button('Run'):
        for persona_elegida in personas_elegidas:
            if persona_elegida not in persons_textids_df.index:
                continue

            st.markdown(f'### Información: {persona_elegida}')
            persona_textids = persons_textids_df[persons_textids_df.index == persona_elegida]['text_id'].to_list()[0]

            st.markdown(f'''#### Evolución de cantidad de tweets y sentimiento para: {persona_elegida}''')
            _sentiment_df = sentiment_df[sentiment_df['text_id'].isin(persona_textids)]
            topic_sent_df = _sentiment_df.groupby('date')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
            _size_df = _sentiment_df.groupby('date').size()
            topic_sent_df = pd.concat([topic_sent_df, _size_df], axis=1)
            sent_df = topic_sent_df.round(2)
            sent_df = sent_df.sort_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(name='Positive', x=sent_df.index.to_list(), y=sent_df['positive_score'].to_list()))
            fig.add_trace(go.Bar(name='Neutral', x=sent_df.index.to_list(), y=sent_df['neutral_score'].to_list()))
            fig.add_trace(go.Bar(name='Negative', x=sent_df.index.to_list(), y=sent_df['negative_score'].to_list()))
            fig.add_trace(go.Scatter(name='Cantidad', x=sent_df.index, y=sent_df[0]), secondary_y=True)
            # Change the bar mode
            fig.update_layout(barmode='stack')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig)


            st.markdown(f'''#### Cantidad de tweets y sentimiento por tópicos para: {persona_elegida}''')
            _sentiment_df = sentiment_df[sentiment_df['text_id'].isin(persona_textids)]
            global_sent_df = pd.DataFrame(_sentiment_df[['positive_score', 'neutral_score', 'negative_score']].agg('mean')).transpose()
            global_sent_df.index = ['global']
            global_sent_df['size'] = _sentiment_df.shape[0]
            topic_sent_df = _sentiment_df.groupby('labels')[['positive_score', 'neutral_score', 'negative_score']].agg('mean')
            topic_sent_df['size'] = [_sentiment_df.groupby('labels').size().to_dict()[x] for x in topic_sent_df.index]
            topic_sent_df.index = [dict_names[x] for x in topic_sent_df.index]
            topic_sent_df = topic_sent_df[topic_sent_df.index != ''].sort_index()
            sent_df = pd.concat([global_sent_df, topic_sent_df], axis=0).round(2)

            # sent_df = sent_df[sent_df['size'] >= 30]
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


            st.markdown(f'''#### Cantidad de tweets y sentimiento por tópicos para: {persona_elegida}''')

            new_df = text_df[text_df['text_id'].isin(persona_textids)]
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

            for topic, name in dict_names.items():
                if int(topic) != -1:
                    selection = new_df.loc[new_df.topic == topic, :]
                    if len(selection) <= 0:
                        continue
                    selection["text"] = ""
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


