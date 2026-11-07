import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(BASE_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((BASE_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
df_test = load_test()
config = load_config()

X_test = df_test.drop(columns='target')
y_test = df_test['target']
features = list(X_test.columns)
proba = test_proba(model, X_test)

st.title('Отбор хороших вин')

threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.01, key='threshold')
st.sidebar.write('Вино попадает в подборку, если вероятность не меньше порога')

tab1, tab2, tab3, tab4 = st.tabs(['Одно вино', 'Модельные метрики', 'Загрузка файла', 'Что выучила модель'])

with tab1:
    st.header('Оценка одного вина')

    wine_type = st.radio('Тип вина', ['белое', 'красное'], horizontal=True)
    values = {'is_red': 1 if wine_type == 'красное' else 0}

    cols = st.columns(2)
    names = [name for name in features if name != 'is_red']
    for i, name in enumerate(names):
        low = float(X_test[name].min())
        high = float(X_test[name].max())
        values[name] = cols[i % 2].number_input(name, low, high, float(X_test[name].median()),
                                                step=(high - low) / 100, format='%.4f')

    wine = pd.DataFrame([values])[features]
    p = model.predict_proba(wine)[0, 1]

    st.metric('Вероятность что вино хорошее', f'{p:.3f}')
    if p >= threshold:
        st.success(f'Решение модели: берем в подборку, порог {threshold:.2f}')
    else:
        st.warning(f'Решение модели: не берем в подборку, порог {threshold:.2f}')

with tab2:
    st.header('Модельные метрики на тестовой выборке')

    pred = (proba >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred)
    TN, FP, FN, TP = cm.ravel()

    col1, col2, col3 = st.columns(3)
    col1.metric('Precision', f'{metrics.precision_score(y_test, pred, zero_division=0):.3f}')
    col2.metric('Recall', f'{metrics.recall_score(y_test, pred):.3f}')
    col3.metric('F1', f'{metrics.f1_score(y_test, pred):.3f}')

    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    prec, rec, _ = metrics.precision_recall_curve(y_test, proba)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, label='модель')
    ax[0].plot([0, 1], [0, 1], 'k--', label='случайная')
    ax[0].scatter(FP / (FP + TN), TP / (TP + FN), color='red', s=70, zorder=3, label='текущий порог')
    ax[0].set_xlabel('FPR')
    ax[0].set_ylabel('TPR')
    ax[0].set_title('ROC')
    ax[0].legend()

    ax[1].plot(rec, prec, label='модель')
    ax[1].axhline(y_test.mean(), color='k', linestyle='--', label='случайная')
    if TP + FP > 0:
        ax[1].scatter(TP / (TP + FN), TP / (TP + FP), color='red', s=70, zorder=3, label='текущий порог')
    ax[1].set_xlabel('recall')
    ax[1].set_ylabel('precision')
    ax[1].set_title('PR')
    ax[1].legend()
    st.pyplot(fig)
    plt.close(fig)

    st.write(f'Сейчас в подборку попадает {pred.sum()} вин из {len(pred)}')
    st.info('Если увеличить порог, вин в подборке станет меньше, но посредственных среди них тоже будет меньше. '
            'Если уменьшить порог, подборка будет шире и мы пропустим меньше хороших вин, '
            'но зато в нее чаще будут попадать посредственные')

with tab3:
    st.header('Предсказание для файла')
    st.write(f'Порог берется из слайдера слева, сейчас {threshold:.2f}')

    uploaded = st.file_uploader('CSV с признаками вин', type='csv')

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            data = None
            st.error('Не получилось прочитать файл, нужен обычный CSV')

        if data is not None:
            missing = [name for name in features if name not in data.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                try:
                    file_proba = model.predict_proba(data[features])[:, 1]
                except ValueError:
                    st.error('В столбцах с признаками должны быть только числа и без пропусков')
                else:
                    result = data.copy()
                    result['probability'] = file_proba
                    result['decision'] = ['хорошее' if x >= threshold else 'не хорошее' for x in file_proba]
                    st.dataframe(result)
                    st.download_button('Скачать таблицу', result.to_csv(index=False).encode('utf-8-sig'),
                                       file_name='predictions.csv', mime='text/csv')

with tab4:
    st.header('Что выучила модель')

    coefs = pd.Series(model[-1].coef_[0], index=features).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:red' if x < 0 else 'tab:green' for x in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.set_xlabel('коэффициент после масштабирования')
    st.pyplot(fig)
    plt.close(fig)

    st.write('Сильнее всего за хорошее вино голосуют: ' + ', '.join(coefs.index[-3:][::-1]))
    st.write('Сильнее всего против: ' + ', '.join(coefs.index[:3]))
