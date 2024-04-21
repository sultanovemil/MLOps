
from scripts.model_training import model_training
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardiovascular-Disease App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def user_input_features():
    age = st.sidebar.slider('Возраст',
                            min_value=10,
                            max_value=100,
                            step=1,
    )
    gender= st.sidebar.selectbox('Пол',
                                options=('Мужской', 'Женский'),
    )
    height = st.sidebar.slider('Рост (см)',
                            min_value=100,
                            max_value=200,
                            value=150,
                            step=1,
    )
    weight = st.sidebar.slider('Вес (кг)',
                            min_value=30,
                            max_value=200,
                            value=70,
                            step=1,
    )
    ap_hi = st.sidebar.slider('Систолическое артериальное давление',
                            min_value=50,
                            max_value=200,
                            value=120,
                            step=1,
    )
    ap_lo = st.sidebar.slider('Диастолическое артериальное давление',
                            min_value=50,
                            max_value=200,
                            value=80,
                            step=1,
    )
    cholesterol = st.sidebar.selectbox('Общий холестерин (ммоль/л.)',
                                options=('<5','5-63', '>6.3'),
    )
    gluc = st.sidebar.selectbox('Глюкоза (ммоль/л.)',
                                options=('3.5—5.5','5.6-9', '>9'),
    )
    smoke = st.sidebar.selectbox('Курение',
                                options=('Да','Нет'),
    )
    alco = st.sidebar.selectbox('Употребление алкоголя',
                                options=('Да','Нет'),
    )
    active = st.sidebar.selectbox('Физическая активность',
                                options=('Да','Нет'),
    )    
    
    
    def map_gluc(gluc):
        if gluc == '3.5—5.5':
            return '1'
        elif gluc == '5.6-9':
            return '2'
        else:
            return '3'

    
    def map_cholesterol(cholesterol):
        if cholesterol == '<5':
            return '1'
        elif cholesterol == '5-63':
            return '2'
        else:
            return '3'
    
    
    age = age * 365
    data = {'age': age,
            'gender': '1' if gender == 'Женский' else '0',
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': map_cholesterol(cholesterol),
            'gluc': map_gluc(gluc),
            'smoke': '1' if smoke == 'Да' else '0',
            'alco':  '1' if alco == 'Да' else '0',
            'active': '1' if active == 'Да' else '0',                      
    }
    features = pd.DataFrame(data, index=[0])
    return features


@st.cache_data()
def get_model():
    model, metric = model_training()
    model_json = {'model': model,
                  'metric': metric}
    return model_json


def main():

    st.write(""" # Приложение для определения наличия сердечно-сосудистого заболевания (ССЗ) :heartpulse: """)
    st.sidebar.header("Параметры ввода")
    st.divider()
    user_data = user_input_features()
    st.write(" # Ваши данные")
    new_column_names = {'age': 'Возраст (дней)',
                        'gender': 'Пол',
                        'height': 'Рост (см)' ,
                        'weight': 'Вес (кг)',
                        'ap_hi': 'Систолическое давление',
                        'ap_lo': 'Диастолическое давление',
                        'cholesterol': 'Общий холестерин',
                        'gluc': 'Глюкоза',
                        'smoke': 'Курение',
                        'alco':  'Алкоголь',
                        'active': 'Физическая активность',
                        'cardio': 'x',                      
        }
    user_data_rus = user_data.rename(columns=new_column_names)
    st.dataframe(user_data_rus)
    
    with st.spinner('Загрузка модели ...'):
        model = get_model()
        st.success('Модель загружена!')
    
    st.divider()

    diag_btn = st.button("Диагностика", type="primary")
    if diag_btn == True:         
        result = ' '.join(map(str, model['model'].predict(user_data)))
        result = "Положительный" if result == "positive" else "Отрицательный"
        metric = model['metric']
        col1, col2 = st.columns(2)
        col1.metric(label=" # :heartpulse: Результат",
                    value=result,
        )
        col2.metric(label=" # Метрика",
                    value=str(metric),
        )
 
      
if __name__ == "__main__":
    main()