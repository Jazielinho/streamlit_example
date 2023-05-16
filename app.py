
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import normalize



# directory = '/media/jahaziel/Datos/proyectos/Smarketing/dataset/Chess_second/2023-05-14'

# directorio actual automático
directory = __file__.split('app.py')[0]
# print(directory)

WIDTH = 1000
HEIGHT = 600

min_topic_sizes = list(set([x.split('.csv')[0].split('_')[-1] for x in os.listdir(directory) if x.endswith('.csv')]))
min_topic_sizes = sorted([int(x) for x in min_topic_sizes])
min_topic_sizes = [str(x) for x in min_topic_sizes]

st.set_page_config(page_title='Chess Champions', page_icon=':chess_pawn:', layout='wide')

st.title(f'''World chess championship''')

min_topic_size = st.radio('Mínimo tamaño de Topic', tuple(min_topic_sizes))

topics_over_time_df = pd.read_csv(f'''{directory}/topics_over_time_{min_topic_size}.csv''')
best_text_by_topics_df = pd.read_csv(f'''{directory}/best_text_by_topics_{min_topic_size}.csv''')
topic_word_weights_df = pd.read_csv(f'''{directory}/topic_word_weights_{min_topic_size}.csv''')
topic_labels_df = pd.read_csv(f'''{directory}/topic_labels_{min_topic_size}.csv''')
topic_sentiment_info_df = pd.read_csv(f'''{directory}/topic_sentiment_info_{min_topic_size}.csv''')
info_hashtags_df = pd.read_csv(f'''{directory}/info_hashtags_{min_topic_size}.csv''')
info_persons_df = pd.read_csv(f'''{directory}/info_persons_{min_topic_size}.csv''')
text_info_df = pd.read_csv(f'''{directory}/text_info_{min_topic_size}.csv''')
embeddings_df = pd.read_csv(f'''{directory}/embeddings_{min_topic_size}.csv''')
info_emojis_df = pd.read_csv(f'''{directory}/info_emojis_{min_topic_size}.csv''')

text_info_df['date'] = pd.to_datetime(text_info_df['date']).dt.date
topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp']).dt.date

for var in ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']:
    text_info_df[var] = text_info_df[var].fillna(0).astype(int)

text_info_df = text_info_df.merge(topic_sentiment_info_df, on='id')
text_info_df = text_info_df.merge(embeddings_df, on='id')

text_info_df = text_info_df[text_info_df['date'] >= datetime.date(2023, 4, 1)]
topics_over_time_df = topics_over_time_df[topics_over_time_df['Timestamp'] >= datetime.date(2023, 4, 1)]

del topic_sentiment_info_df, embeddings_df

dict_topic_labels = dict(zip(topic_labels_df['topic_id'], topic_labels_df['topic_label']))

# ====================================================================================================
st.markdown(f'''## Estadísticas generales''')
st.markdown(f'''
* Número de textos publicados: {text_info_df.shape[0]}
* Periodo de tiempo: {text_info_df['date'].min()} - {text_info_df['date'].max()}
* Número de hashtags encontrados: {len(set(info_hashtags_df['hashtag']))}
* Número de personas encontradas: {len(set(info_persons_df['person']))}
''')

# ====================================================================================================

tab1, tab2, tab3, tab4 = st.tabs(['GENERAL', 'TOPICOS', 'PERSONAS', 'HASHTAGS'])

with tab1:
    st.markdown(f'''#### Estadísticas''')
    general_statistics_df = text_info_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
    general_statistics_df.columns = ['Estadística', 'Valor']
    st.table(general_statistics_df)

    st.markdown(f'''#### Estadísticas a lo largo del tiempo''')
    statistics_date_df = text_info_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum', 'impression_count': 'sum'}).reset_index()
    statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
    statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
    fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Textos más populares''')
    columna_ordenar = st.selectbox('Elegir columna de ordenamiento', ['impression_count', 'retweet_count', 'reply_count', 'like_count', 'quote_count'])
    popular_texts_df = text_info_df.sort_values(columna_ordenar, ascending=False).head(10).reset_index(drop=True)
    popular_texts_df = popular_texts_df[['date', 'content', 'impression_count', 'retweet_count', 'reply_count', 'like_count', 'quote_count']]
    popular_texts_df.columns = ['Fecha', 'Texto', 'Impresiones', 'Retweets', 'Respuestas', 'Likes', 'Quotes']
    st.table(popular_texts_df)

    st.markdown(f'''#### Sentimiento de los textos ''')
    general_sentiment_df = text_info_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
    general_sentiment_df.columns = ['Sentimiento', 'Valor']
    st.table(general_sentiment_df)

    sentiment_date_df = text_info_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)


with tab2:

    fig = go.Figure()
    _df = text_info_df[text_info_df['topic'] == -1]
    fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['content'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
    all_topics = sorted(text_info_df['topic'].unique())
    for topic in all_topics:
        if int(topic) == -1:
            continue
        selection = text_info_df[text_info_df['topic'] == topic]
        fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['content'], hoverinfo='text', mode='markers+text', name=dict_topic_labels[topic], marker=dict(size=5, opacity=0.5)))
    x_range = [text_info_df['x'].min() - abs(text_info_df['x'].min() * 0.15), text_info_df['x'].max() + abs(text_info_df['x'].max() * 0.15)]
    y_range = [text_info_df['y'].min() - abs(text_info_df['y'].min() * 0.15), text_info_df['y'].max() + abs(text_info_df['y'].max() * 0.15)]
    fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
    fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

    topic_count_sentiment_df = text_info_df.groupby('topic').agg({'id': 'count', 'negative_score': 'mean', 'neutral_score': 'mean', 'positive_score': 'mean'}).reset_index()
    topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['topic'] != -1].sort_values('topic')
    topic_count_sentiment_df['topic_label'] = [dict_topic_labels[x] for x in topic_count_sentiment_df['topic']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
    fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Estadísticas por tópicos''')
    topic_statistics_df = text_info_df.groupby('topic').agg({'impression_count': 'sum', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum'}).reset_index()
    topic_statistics_df = topic_statistics_df[topic_statistics_df['topic'] != -1].sort_values('topic')
    topic_statistics_df['topic_label'] = [dict_topic_labels[x] for x in topic_statistics_df['topic']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=topic_statistics_df['topic_label'], y=topic_statistics_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['retweet_count'], name='Número de retweets'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['reply_count'], name='Número de respuestas'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['like_count'], name='Número de likes'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['quote_count'], name='Número de quotes'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Evolución de tópicos ''')
    normalize_frequency = st.radio('Normalizar', ['No', 'Sí']) == 'Sí'
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    topics_over_time_df = topics_over_time_df.sort_values(["Topic", "Timestamp"])
    topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])
    topics_over_time_df["Name"] = topics_over_time_df.Topic.map(dict_topic_labels)
    fig = go.Figure()
    for index, topic in enumerate(topics_over_time_df.Topic.unique()):
        trace_data = topics_over_time_df.loc[topics_over_time_df.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y, mode='lines+markers', marker_color=colors[index % 7], hoverinfo="text", name=topic_name, hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout( yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency", template="simple_white", hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(width=WIDTH * 2, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown(f'''### Información por tópico ''')
    topicos_elegidos = st.multiselect('Elige los tópicos', [x for x in list(dict_topic_labels.keys()) if x != -1], [0, 1])
    if st.button('Ejecutar'):
        for topico_elegido in topicos_elegidos:
            st.markdown(f'''#### Tópico {topico_elegido}: {dict_topic_labels[topico_elegido]}''')
            st.markdown(f'''##### Palabras más frecuentes''')
            best_words = topic_word_weights_df[topic_word_weights_df['topic'] == topico_elegido].sort_values('weight', ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=best_words['word'], y=best_words['weight'], name='Peso'))
            fig.update_layout(xaxis_title="Palabras", yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Textos más representativos''')
            best_texts = best_text_by_topics_df[best_text_by_topics_df['labels'] == topico_elegido]['content']
            st.write(best_texts)

            topic_text_df = text_info_df[text_info_df['topic'] == topico_elegido]

            st.markdown(f'''##### Estadísticas''')
            general_statistics_df = topic_text_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
            general_statistics_df.columns = ['Estadística', 'Valor']
            st.table(general_statistics_df)

            statistics_date_df = topic_text_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum','impression_count': 'sum'}).reset_index()
            statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
            statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
            fig.update_layout(barmode='stack',legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Sentimiento de los textos ''')
            general_sentiment_df = topic_text_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            general_sentiment_df.columns = ['Sentimiento', 'Valor']
            st.table(general_sentiment_df)

            sentiment_date_df = topic_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown(f'''## Personajes con más menciones ''')
    count_person_df = info_persons_df['person'].value_counts().reset_index().head(20)
    count_person_df.columns = ['person', 'count']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=count_person_df['person'], y=count_person_df['count'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de menciones por personaje')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    personas_mencionadas = st.multiselect('Elige las personas', count_person_df['person'], count_person_df['person'][:2])

    if st.button('Analizar personas mencionadas'):
        for persona_elegida in personas_mencionadas:
            persona_text_df = text_info_df[text_info_df['id'].isin(info_persons_df[info_persons_df['person'] == persona_elegida]['id'])]

            st.markdown(f'''#### Persona: {persona_elegida}''')

            st.markdown(f'''##### Estadísticas''')
            general_statistics_df = persona_text_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
            general_statistics_df.columns = ['Estadística', 'Valor']
            st.table(general_statistics_df)

            statistics_date_df = persona_text_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum','impression_count': 'sum'}).reset_index()
            statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
            statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Sentimiento de los textos ''')
            general_sentiment_df = persona_text_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            general_sentiment_df.columns = ['Sentimiento', 'Valor']
            st.table(general_sentiment_df)

            sentiment_date_df = persona_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Participación en tópicos ''')

            topic_count_sentiment_df = persona_text_df.groupby('topic').agg({'id': 'count', 'negative_score': 'mean', 'neutral_score': 'mean', 'positive_score': 'mean'}).reset_index()
            topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['topic'] != -1].sort_values('topic')
            topic_count_sentiment_df['topic_label'] = [dict_topic_labels[x] for x in topic_count_sentiment_df['topic']]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
            fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
            st.plotly_chart(fig, use_container_width=True)

            # fig = go.Figure()
            # _df = persona_text_df[persona_text_df['topic'] == -1]
            # fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['content'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
            # all_topics = sorted(persona_text_df['topic'].unique())
            # for topic in all_topics:
            #     if int(topic) == -1:
            #         continue
            #     selection = persona_text_df[persona_text_df['topic'] == topic]
            #     fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['content'], hoverinfo='text', mode='markers+text', name=dict_topic_labels[topic], marker=dict(size=5, opacity=0.5)))
            # x_range = [persona_text_df['x'].min() - abs(persona_text_df['x'].min() * 0.15), persona_text_df['x'].max() + abs(persona_text_df['x'].max() * 0.15)]
            # y_range = [persona_text_df['y'].min() - abs(persona_text_df['y'].min() * 0.15), persona_text_df['y'].max() + abs(persona_text_df['y'].max() * 0.15)]
            # fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
            # fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
            # fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
            # fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
            # fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
            # fig.update_xaxes(visible=False)
            # fig.update_yaxes(visible=False)
            # fig.update_layout(width=WIDTH, height=HEIGHT)
            # st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown(f'''## HASHTAGS más usados ''')
    count_hashtag_df = info_hashtags_df['hashtag'].value_counts().reset_index().head(20)
    count_hashtag_df.columns = ['hashtag', 'count']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=count_hashtag_df['hashtag'], y=count_hashtag_df['count'], name='Cantidad de hashtag', mode='lines+markers'), secondary_y=True)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de hashtag')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    hashtag_mencionadas = st.multiselect('Elige hashtags', count_hashtag_df['hashtag'], count_hashtag_df['hashtag'][:2])

    if st.button('Analizar hashtags'):
        for hashtag_elegida in hashtag_mencionadas:
            st.markdown(f'''#### hashtags: {hashtag_elegida}''')

            hashtag_text_df = text_info_df[text_info_df['id'].isin(info_hashtags_df[info_hashtags_df['hashtag'] == hashtag_elegida]['id'])]

            st.markdown(f'''##### Estadísticas''')
            general_statistics_df = hashtag_text_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
            general_statistics_df.columns = ['Estadística', 'Valor']
            st.table(general_statistics_df)

            statistics_date_df = hashtag_text_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum','impression_count': 'sum'}).reset_index()
            statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
            statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Sentimiento de los textos ''')
            general_sentiment_df = hashtag_text_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            general_sentiment_df.columns = ['Sentimiento', 'Valor']
            st.table(general_sentiment_df)

            sentiment_date_df = hashtag_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)













