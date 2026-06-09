import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Прогноз дождя в Австралии", page_icon="🌧️", layout="wide")
st.title('🌧️ Прогнозирование дождя в Австралии на завтра')
st.markdown('---')

@st.cache_data
def load_and_prepare_data():
    """Загрузка и предобработка данных (кэшируется)"""
    
    # Загружаем данные
    data = pd.read_csv('weatherAUS.csv')
    data = data.drop_duplicates()
    
    # Удаляем строки с пропущенным RainTomorrow
    data = data.dropna(subset=['RainTomorrow'])
    
    # Заполняем категориальные признаки модой
    cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Заполняем числовые признаки медианой
    numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Извлекаем месяц из даты
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    
    # Создаём новые признаки
    data['TempRange'] = data['MaxTemp'] - data['MinTemp']
    data['PressureChange'] = data['Pressure9am'] - data['Pressure3pm']
    data['HumidityChange'] = data['Humidity3pm'] - data['Humidity9am']
    
    # Кодируем целевую переменную
    le = LabelEncoder()
    data['RainTomorrow_encoded'] = le.fit_transform(data['RainTomorrow'])
    
    # Кодируем RainToday
    data['RainToday_encoded'] = (data['RainToday'] == 'Yes').astype(int)
    
    # One-Hot Encoding для категориальных признаков
    cat_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    data = pd.get_dummies(data, columns=cat_features, drop_first=True)
    
    # Финальные числовые признаки
    final_numeric = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                    'Temp9am', 'Temp3pm', 'Month',
                    'TempRange', 'PressureChange', 'HumidityChange']
    
    # Признаки для модели
    onehot_cols = [col for col in data.columns if any(cat in col for cat in cat_features)]
    feature_cols = final_numeric + onehot_cols + ['RainToday_encoded']
    
    X = data[feature_cols].copy()
    y = data['RainTomorrow_encoded'].copy()
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    X[final_numeric] = scaler.fit_transform(X[final_numeric])
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, le, feature_cols, final_numeric, data

# Загружаем данные
with st.spinner('Загрузка данных...'):
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols, final_numeric, full_data = load_and_prepare_data()

st.success(f'Данные загружены! Обучающая выборка: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}')
st.info(f'Признаков: {len(feature_cols)}, Классов: {len(label_encoder.classes_)}')

st.sidebar.header('⚙️ Настройка модели RandomForest')

st.sidebar.markdown("""
**Random Forest** — ансамблевый метод из множества деревьев решений.

**Гиперпараметры:**
- **n_estimators** — количество деревьев (больше → точнее, но медленнее)
- **max_depth** — максимальная глубина деревьев (ограничивает переобучение)
- **min_samples_split** — минимальное число образцов для разделения узла
""")

# Слайдеры для гиперпараметров
n_estimators = st.sidebar.slider(
    'Количество деревьев (n_estimators)',
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help='Больше деревьев — выше точность, но дольше обучение'
)

max_depth = st.sidebar.slider(
    'Максимальная глубина (max_depth)',
    min_value=3,
    max_value=30,
    value=10,
    step=1,
    help='Ограничение глубины предотвращает переобучение'
)

min_samples_split = st.sidebar.slider(
    'Min samples split',
    min_value=2,
    max_value=20,
    value=2,
    step=1,
    help='Минимальное число образцов для разделения узла'
)

# Кнопка переобучения
retrain_button = st.sidebar.button('🔄 Переобучить модель', type='primary', use_container_width=True)

@st.cache_resource
def train_model(n_est, m_depth, m_split):
    """Обучает RandomForest с заданными параметрами"""
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=m_depth,
        min_samples_split=m_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Обучаем или загружаем модель
if retrain_button or 'current_model' not in st.session_state:
    with st.spinner('Обучение модели...'):
        st.session_state.current_model = train_model(n_estimators, max_depth, min_samples_split)
        st.session_state.current_params = (n_estimators, max_depth, min_samples_split)

model = st.session_state.current_model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

st.markdown(f'### 📊 Результаты модели')
st.markdown(f'**Параметры:** n_estimators={st.session_state.current_params[0]}, max_depth={st.session_state.current_params[1]}, min_samples_split={st.session_state.current_params[2]}')

# Метрики в ряд
col1, col2, col3, col4, col5 = st.columns(5)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

with col1:
    st.metric('Accuracy', f'{accuracy:.3f}')
with col2:
    st.metric('Precision', f'{precision:.3f}')
with col3:
    st.metric('Recall', f'{recall:.3f}')
with col4:
    st.metric('F1-score', f'{f1:.3f}')
with col5:
    st.metric('ROC-AUC', f'{roc_auc:.3f}')

st.markdown('---')
col_left, col_right = st.columns(2)

with col_left:
    st.subheader('Матрица ошибок')
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=label_encoder.classes_
    )
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f'Матрица ошибок')
    st.pyplot(fig)

with col_right:
    st.subheader('Важность признаков (топ-15)')
    feature_importance = pd.DataFrame({
        'Признак': feature_cols,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(feature_importance['Признак'], feature_importance['Важность'], color='steelblue')
    ax.set_title('Топ-15 важных признаков')
    ax.set_xlabel('Важность')
    st.pyplot(fig)

st.markdown('---')
st.subheader('📈 Влияние гиперпараметров на качество модели')

param_to_test = st.selectbox(
    'Выберите гиперпараметр для анализа:',
    ['n_estimators', 'max_depth', 'min_samples_split']
)

if st.button('Построить график зависимости (может занять время)'):
    if param_to_test == 'n_estimators':
        param_values = [10, 50, 100, 200, 300, 400, 500]
        fixed_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split}
        x_label = 'Количество деревьев'
    elif param_to_test == 'max_depth':
        param_values = [3, 5, 10, 15, 20, 25, 30]
        fixed_params = {'n_estimators': n_estimators, 'min_samples_split': min_samples_split}
        x_label = 'Максимальная глубина'
    else:
        param_values = [2, 5, 10, 15, 20]
        fixed_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        x_label = 'Min samples split'
    
    metrics_data = {'param': [], 'Accuracy': [], 'F1': [], 'ROC_AUC': []}
    progress_bar = st.progress(0)
    
    for i, p_val in enumerate(param_values):
        kwargs = {param_to_test: p_val, **fixed_params}
        temp_model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        temp_proba = temp_model.predict_proba(X_test)[:, 1]
        
        metrics_data['param'].append(p_val)
        metrics_data['Accuracy'].append(accuracy_score(y_test, temp_pred))
        metrics_data['F1'].append(f1_score(y_test, temp_pred, average='binary'))
        metrics_data['ROC_AUC'].append(roc_auc_score(y_test, temp_proba))
        
        progress_bar.progress((i + 1) / len(param_values))
    
    progress_bar.empty()
    
    # График
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_data['param'], metrics_data['Accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(metrics_data['param'], metrics_data['F1'], 's-', label='F1-score', linewidth=2)
    ax.plot(metrics_data['param'], metrics_data['ROC_AUC'], '^-', label='ROC-AUC', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Значение метрики')
    ax.set_title(f'Зависимость метрик от {param_to_test}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown('---')
st.subheader('🌤️ Предсказание дождя на завтра')

st.markdown('Введите погодные показатели на сегодня:')

col1, col2, col3 = st.columns(3)

with col1:
    min_temp = st.slider('Мин. температура (°C)', -10.0, 40.0, 15.0, 0.5)
    max_temp = st.slider('Макс. температура (°C)', 0.0, 50.0, 25.0, 0.5)
    rainfall = st.slider('Осадки сегодня (мм)', 0.0, 100.0, 0.0, 0.5)
    humidity9am = st.slider('Влажность 9:00 (%)', 0, 100, 60)
    humidity3pm = st.slider('Влажность 15:00 (%)', 0, 100, 50)

with col2:
    pressure9am = st.slider('Давление 9:00 (гПа)', 980.0, 1040.0, 1015.0, 0.5)
    pressure3pm = st.slider('Давление 15:00 (гПа)', 980.0, 1040.0, 1013.0, 0.5)
    cloud9am = st.slider('Облачность 9:00 (октанты)', 0, 8, 4)
    cloud3pm = st.slider('Облачность 15:00 (октанты)', 0, 8, 4)
    temp9am = st.slider('Температура 9:00 (°C)', -5.0, 45.0, 18.0, 0.5)

with col3:
    temp3pm = st.slider('Температура 15:00 (°C)', 0.0, 50.0, 24.0, 0.5)
    wind_speed9am = st.slider('Скорость ветра 9:00 (км/ч)', 0, 80, 15)
    wind_speed3pm = st.slider('Скорость ветра 15:00 (км/ч)', 0, 80, 20)
    wind_gust_speed = st.slider('Порывы ветра (км/ч)', 0, 120, 30)
    rain_today = st.selectbox('Был ли дождь сегодня?', ['No', 'Yes'])

# Упрощённый выбор направления ветра
wind_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
wind_gust_dir = st.selectbox('Направление порывов ветра', wind_dirs)
wind_dir9am = st.selectbox('Направление ветра 9:00', wind_dirs)
wind_dir3pm = st.selectbox('Направление ветра 15:00', wind_dirs)

# Кнопка предсказания
if st.button('🔮 Предсказать дождь на завтра', type='primary', use_container_width=True):
    # Создаём признаки
    temp_range = max_temp - min_temp
    pressure_change = pressure9am - pressure3pm
    humidity_change = humidity3pm - humidity9am
    month = 6  # Июнь по умолчанию
    
    # Формируем словарь со всеми признаками
    input_dict = {
        'MinTemp': min_temp, 'MaxTemp': max_temp, 'Rainfall': rainfall,
        'WindGustSpeed': wind_gust_speed, 'WindSpeed9am': wind_speed9am,
        'WindSpeed3pm': wind_speed3pm, 'Humidity9am': humidity9am,
        'Humidity3pm': humidity3pm, 'Pressure9am': pressure9am,
        'Pressure3pm': pressure3pm, 'Cloud9am': cloud9am, 'Cloud3pm': cloud3pm,
        'Temp9am': temp9am, 'Temp3pm': temp3pm, 'Month': month,
        'TempRange': temp_range, 'PressureChange': pressure_change,
        'HumidityChange': humidity_change, 'RainToday_encoded': 1 if rain_today == 'Yes' else 0
    }
    
    # Добавляем one-hot колонки
    for col in feature_cols:
        if col not in input_dict and col != 'RainToday_encoded':
            input_dict[col] = 0
    
    # One-hot для Location (используем средний город)
    if 'Location_Albury' in input_dict:
        input_dict['Location_Albury'] = 1
    
    # One-hot для направлений ветра
    for prefix, value in [('WindGustDir', wind_gust_dir), 
                          ('WindDir9am', wind_dir9am), 
                          ('WindDir3pm', wind_dir3pm)]:
        for d in wind_dirs:
            col_name = f'{prefix}_{d}'
            if col_name in input_dict:
                input_dict[col_name] = 1 if d == value else 0
    
    # Создаём DataFrame
    input_df = pd.DataFrame([input_dict])[feature_cols]
    
    # Масштабируем числовые признаки
    input_df[final_numeric] = scaler.transform(input_df[final_numeric])
    
    # Предсказываем
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    
    # Выводим результат
    st.markdown('---')
    
    if predicted_class == 'No':
        st.success(f'### ☀️ Прогноз: **Дождя завтра НЕ будет**')
        st.metric('Вероятность дождя', f'{prediction_proba[1]:.1%}')
    else:
        st.error(f'### 🌧️ Прогноз: **Завтра будет ДОЖДЬ**')
        st.metric('Вероятность дождя', f'{prediction_proba[1]:.1%}')
    
    # Факторы, повлиявшие на прогноз
    st.markdown('**Факторы, указывающие на дождь:**')
    factors = []
    if humidity3pm > 70:
        factors.append('🌡️ Высокая влажность днём (>70%)')
    if pressure_change > 5:
        factors.append('📉 Падение давления (>5 гПа)')
    if cloud3pm >= 6:
        factors.append('☁️ Сильная облачность днём (≥6 октантов)')
    if rainfall > 0:
        factors.append('🌧️ Сегодня были осадки')
    
    if factors:
        for f in factors:
            st.write(f)
    else:
        st.write('✅ Признаки дождя отсутствуют')

st.sidebar.markdown('---')
st.sidebar.markdown("""
### ℹ️ О модели
- **Тип**: RandomForestClassifier
- **Признаков**: числовые + one-hot
- **Классы**: No / Yes (дождь завтра)
- **Данные**: погода в Австралии за 10 лет
""")

st.sidebar.markdown("""
### 📝 Как использовать
1. Измените гиперпараметры в слайдерах
2. Нажмите **«Переобучить модель»**
3. Анализируйте изменение метрик
4. Введите погодные данные и получите прогноз
""")