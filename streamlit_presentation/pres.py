import random
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import components
import streamlit.components.v1 as components
from wordcloud import WordCloud
import plotly.graph_objs as go
import squarify
import joblib
from PIL import Image

df_1 = pd.read_csv('streamlit_presentation/datasets/df_name_Date_text_topic_keywords.csv')
df_2 = pd.read_csv('streamlit_presentation/datasets/df_organizations_persons_locations_krasivi_text.csv')
df = pd.concat([df_1, df_2], axis = 1)
df["Date_1"] = pd.to_datetime(df["Date"])
dict_NER = joblib.load('streamlit_presentation/dict_NER_final.pkl')
locations = joblib.load('streamlit_presentation/word_clouds/final_dict_loc.pkl')
organizations = joblib.load('streamlit_presentation/word_clouds/final_dict_org.pkl')
persons = joblib.load('streamlit_presentation/word_clouds/final_dict_per.pkl')

st.set_page_config(layout="wide")

st.title('Деконструкция новостей')
st.header('Пакет инструментов для динамичного и адаптивного анализа большого корпуса новостей ТГ каналов')
st.markdown('Подробная информация о проекте в [GitHub](https://github.com/aveletuk/Deconstructing_news_with_NLP)!')

st.header('РАЗВЕДОЧНЫЙ АНАЛИЗ АНАЛИЗ (Exploratory data analysis, EDA)')
with st.expander('Описание датасета'):
    st.subheader('6 популярных новостных ТГ каналов')
    st.subheader("Новости с 1 июля 2021 по 1 июля 2022")
    st.subheader("Более 110.000 текстов")
    w = pd.read_csv('streamlit_presentation/datasets/df_name_Date_text_topic_keywords.csv', index_col=['Date'], parse_dates=['Date'], dayfirst=True).sort_index()
    
    time = w.index

    trace = go.Histogram(
        x=time,
        marker=dict(
            color='#3F3D8A'
        ),
        opacity=0.75
    )

    layout = go.Layout(
        title=' Общее количество текстов в датасете',
        height=600,
        width=1500,
        xaxis=dict(
            title='Дата'
        ),
        yaxis=dict(
            title='Количество текстов за сутки с 01/07/2021 по 01/07/2022'
        ),
        bargap=0.2,
    )

    data = [trace]

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    df['bar_1'] = [i[:-3] for i in df['Date']]

    x = 0.
    y = 0.
    width = 300
    height = 300
    type_list = list(df['name'].unique())
    values = [len(df[df['name'] == i]) for i in type_list]

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    color_brewer = ['#99B2DD','#365942','#3A405A','#fc9505','#FF5D73','#3F3D8A']
    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 2 ),
                fillcolor = color_brewer[counter]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = "{} : {}".format(type_list[counter], values[counter]),
                showarrow = False
            )
        )
        counter = counter + 1
        if counter >= len(color_brewer):
            counter = 0

    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']) for r in rects ], 
        y = [ r['y']+(r['dy']) for r in rects ],
        text = [ str(v) for v in values ], 
        mode = 'text'
    )
            
    layout = dict(
        height=800, 
        width=1500,
        xaxis=dict(showgrid=False,zeroline=False),
        yaxis=dict(showgrid=False,zeroline=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        font=dict(color="#FFFFFF", size = 20)
    )

    # With hovertext
    figure = dict(data=[trace0], layout=layout)
    st.plotly_chart(figure, filename='squarify-tree', use_container_width=True)

st.header('ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ (Latent Dirichlet Allocation, LDA)')

with st.expander('Интерактивная визуализация результатов тематического моделирования'):
    with st.spinner('Creating pyLDAvis Visualization ...'):
        with open('streamlit_presentation/pyLDAvis_pkl/lda.html') as f:
            html_string = f.read()
        components.html(html_string, width=1500, height=800)

with st.expander('Дополнительная информация'):
    st.image('streamlit_presentation/pictures/mf.png', use_column_width=True)
    st.markdown('Почитайте документацию ' 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html')
    st.markdown('Или статьи какие-нибудь:')
    st.markdown('   Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022')
    st.markdown('   Hoffman, M., Bach, F., & Blei, D. (2010). Online learning for latent dirichlet allocation. Advances in Neural Information Processing Systems, 23.')

st.header('РАСПОЗНАВАНИЕ ИМЕНОВАННЫХ СУЩНОСТЕЙ (Named Entity Recognition, NER)')
with st.expander('Анализ именнованных сущностей'):
    # col_name = st.selectbox("Выберите параметры:", sorted(df.columns[0:3]), index =0)
    # #переименовать колонки
    your_word = st.text_input('Задайте сущность', 'Хабаровск')
    try:
        st.write("Сущность", your_word.upper(), "встречается в датасете", len(dict_NER[your_word]), " раз")
    
        st.write("**Рандомная новость про**", your_word.upper())

        random_state = random.randint(0,(len(dict_NER[your_word])))
        news = df["krasivi_text"][dict_NER[your_word][random_state]]

        st.write("**Новость:**", news)

        start_date = st.date_input("Задайте начальную дату", value=pd.to_datetime("2021-07-01", format="%Y-%m-%d"))
        end_date = st.date_input("Задайте конечную дату", value=pd.to_datetime("2021-07-05", format="%Y-%m-%d"))

        # convert the dates to string
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        st.write("Количество новостей про", your_word.upper(), f"с {start} по {end}:", len(df["text"].loc[dict_NER[your_word]].loc[(df['Date'] >= start) & (df['Date'] <= end)]))

        st.write("В каких темах чаще всего появляется", your_word.upper()) 
        # st.write(df.loc[dict_NER[your_word]]['topic'].value_counts())

        cnt_ = df.loc[dict_NER[your_word]]['topic'].value_counts()
        cnt_ = cnt_.sort_index() 
        fig = {
    "data": [
        {
        "values": cnt_.values,
        "labels": cnt_.index,
        "domain": {"x": [0, 1]},
        "name": "Количество новостей в год",
        "hole": .3,
        "type": "pie",
        "pull": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        },],
    "layout": {
            "annotations": [
                { "font": { "size": 20},
                "showarrow": False,
                "text": "Темы",
                    "x": 2,
                    "y": 2
                },
            ]
        }
    }
        st.plotly_chart(fig, use_container_width=True)
        
    except KeyError:
        st.write("Ой! Что-то пошло не так. Попробуйте написать слово с маленькой буквы. Если не поможет, то слова нет в датасете.")

with st.expander('Облака сущностей'):
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
                    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

    knopka1 = ["Топонимы", locations]
    knopka2 = ["Организации", organizations]   
    knopka3 = ["Люди", persons]
    knopka4 = "ВЫБИРАЙ СУЩНОСТИ!!!!!!!!!!!!!!!!!!!"
   
    option = st.selectbox('Выберите категорию', (knopka4, knopka1[0],knopka2[0],knopka3[0]))
    
    if option == knopka1[0]:
        option = knopka1[1]
        img = Image.open("streamlit_presentation/word_clouds/comment.png")
        wave_mask = np.array(img)
        wordcloud = WordCloud(mask=wave_mask, random_state=80, contour_width=2, contour_color='orange').generate(" ".join(option))
        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(wordcloud, interpolation="bilinear")
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.style.use('dark_background')
        fig.savefig("streamlit_presentation/word_clouds/word_cloud.png")
        why_i_hate_streamlit = Image.open('streamlit_presentation/word_clouds/word_cloud.png')
        # why_i_hate_streamlit = why_i_hate_streamlit.resize((1000, 1000))
        st.image(why_i_hate_streamlit)
        # st.pyplot(fig)

    if option == knopka2[0]:
        option = knopka2[1]
        img = Image.open("streamlit_presentation/word_clouds/mozg_3.tif")
        wave_mask = np.array(img)
        wordcloud = WordCloud(mask=wave_mask, random_state=80, contour_width=2, contour_color='orange').generate(" ".join(option))
        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(wordcloud, interpolation="bilinear")
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.style.use('dark_background')
        fig.savefig("streamlit_presentation/word_clouds/word_cloud.png")
        why_i_hate_streamlit = Image.open('streamlit_presentation/word_clouds/word_cloud.png')
        # why_i_hate_streamlit = why_i_hate_streamlit.resize((1000, 1000))
        st.image(why_i_hate_streamlit)
        # st.pyplot(fig)
        

    if option == knopka3[0]:
        option = knopka3[1]
        img = Image.open("streamlit_presentation/word_clouds/mozg_3.tif")
        wave_mask = np.array(img)
        wordcloud = WordCloud(color_func=grey_color_func, mask=wave_mask, random_state=80, contour_width=1, contour_color='grey').generate(" ".join(option))
        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(wordcloud, interpolation="bilinear")
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.style.use('dark_background')
        fig.savefig("streamlit_presentation/word_clouds/word_cloud.png")
        why_i_hate_streamlit = Image.open('streamlit_presentation/word_clouds/word_cloud.png')
        # why_i_hate_streamlit = why_i_hate_streamlit.resize((1000, 1000))
        st.image(why_i_hate_streamlit)
        # st.pyplot(fig)

    else:
        st.write("Облака слов - прикольная визуализация текстовых данных на стыке исследовательского анализа, инфографики и дата-дизайна.")

with st.expander('Дополнительная информация'):
    st.markdown('Имена людей, названия организаций, топонимы и другие имена собственные называют «именованные сущности» (named entities), а саму задачу — «распознавание именованных сущностей» (named entity recognition).')
    st.markdown('Мы используем библиотеку [Natasha](https://natasha.github.io/ner/), которая играючи справляется со всеми базовыми задачами обработки естественного русского языка: сегментация на токены, морфологический анализ, лемматизация и извлечение именованных сущностей. С помощью Natasha мы вытащили из нашего датасета имена людей, названия организаци и топонимы.')
    st.image('streamlit_presentation/pictures/natasha.png', caption='Библиотека Natasha объединяет 9 репозиториев под одним интерфейсом.', use_column_width=True)
    