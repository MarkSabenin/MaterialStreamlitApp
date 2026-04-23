import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go 
import plotly.express as px
from core_logic import load_hybrid_system, predict_hybrid, solve_inverse_problem, get_plot_data


if 'comparison_list' not in st.session_state:
    st.session_state['comparison_list'] = []
if 'cnt_value' not in st.session_state:
    st.session_state['cnt_value'] = 2.0

st.set_page_config(page_title="Composite Designer", layout="wide")

@st.cache_resource
def get_model_bundle():
    return load_hybrid_system()

bundle = get_model_bundle()
meta = bundle['meta']

st.title("🧪 Лаборатория композитов")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["🔮 Прогноз", "🎯 Оптимизация", "📈 Аналитика"])

# --- ВКЛАДКА 1: ПРЯМАЯ ЗАДАЧА ---
with tab1:
    st.sidebar.subheader("Параметры состава")
    
    # 1. Поле ввода СНАРУЖИ формы (для мгновенной работы кнопок + и -)
    cnt = st.sidebar.number_input(
        "Концентрация УНТ (%)", 
        min_value=1.0, 
        max_value=5.0, 
        value=st.session_state['cnt_value'], 
        step=0.01,
        format="%.2f",
        key="cnt_input_field" 
    )
    # Сразу обновляем значение в стейте
    st.session_state['cnt_value'] = cnt

    # 2. Форма для выбора метода и запуска
    with st.sidebar.form("input_form"):
        method_choice = st.selectbox("Метод смешивания", options=meta['method_labels'])
        submitted = st.form_submit_button("Рассчитать свойства", use_container_width=True)
    
    # 3. ЛОГИКА РАСЧЕТА
    if submitted:
        method_idx = meta['method_labels'].index(method_choice)
        # Делаем расчет, используя актуальный cnt
        preds, stds = predict_hybrid(cnt, method_idx, bundle)
        
        # Сохраняем результаты
        st.session_state['last_preds'] = preds
        st.session_state['last_stds'] = stds
        st.session_state['last_params'] = (cnt, method_choice)
        st.session_state['calculated'] = True

    # 4. ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    if st.session_state.get('calculated'):
        c, m = st.session_state['last_params']
        p = st.session_state['last_preds']
        s = st.session_state['last_stds']

        st.subheader(f"Анализ состава: {c}% УНТ | {m}")

        view_mode = st.radio(
            "Формат отображения:",
            ["Визуальные карточки", "Подробная таблица"],
            horizontal=True,
            key="view_mode_selector"
        )
        st.markdown("---")

        if view_mode == "Визуальные карточки":
            cols = st.columns(3)
            for i, prop in enumerate(meta['prop_cols']):
                with cols[i % 3]:
                    st.metric(
                        label=prop, 
                        value=f"{p[i]:.3f}", 
                        delta=f"±{s[i]:.3f}", 
                        delta_color="off"
                    )
        else:
            # Таблица с правильным расчетом доверительных интервалов
            df_res = pd.DataFrame({
                "Характеристика": meta['prop_cols'],
                "Прогноз (Среднее)": [f"{val:.4f}" for val in p],
                "Погрешность (±σ)": [f"{val:.4f}" for val in s],
                "Метод расчета": ["Sklearn (GPR)" if i in meta['sk_indices'] else "GPflow (Multi-GPR)" for i in range(5)]
            })
            # Добавляем интервалы отдельно, чтобы не запутаться в логах
            intervals = []
            for i in range(5):
                p_val, s_val = p[i], s[i]
                if i in meta['log_indices']:
                    # Логарифмический масштаб (Упругость, Износ)
                    log_mu = np.log1p(p_val)
                    low = np.expm1(log_mu - 2*s_val)
                    high = np.expm1(log_mu + 2*s_val)
                else:
                    low, high = p_val - 2*s_val, p_val + 2*s_val
                intervals.append(f"{low:.3f} — {high:.3f}")
            
            df_res["Доверительный интервал (95%)"] = intervals
            st.dataframe(df_res, use_container_width=True)
        
        st.markdown("---")
    
        if st.button("📥 Добавить этот состав в сравнение"):
            # Берем данные из последнего успешного расчета
            c_save, m_save = st.session_state['last_params']
            p_save = st.session_state['last_preds']
            
            import time
            row_id = f"{c_save}_{m_save}_{time.time()}"
            
            new_entry = {
                "ID": row_id,
                "Концентрация": f"{c_save}%",
                "Метод": m_save
            }
            for i, prop in enumerate(meta['prop_cols']):
                new_entry[prop] = round(p_save[i], 4)
            
            st.session_state['comparison_list'].append(new_entry)
            st.toast(f"Состав {c_save}% добавлен в сравнение!")  
    else:
        st.info("Измените параметры в меню слева и нажмите кнопку 'Рассчитать свойства'.")
        
        
with tab2:
    st.header("Настройка целевых показателей")
    st.write("Введите желаемые значения свойств и их важность (веса) для поиска оптимального состава.")

    with st.form("optimization_form"):
        # Создаем колонки для ввода Целей и Весов
        col_target, col_weight = st.columns(2)
        
        target_dict = {}
        weights = []

        with col_target:
            st.subheader("Желаемые значения")
            for i, prop in enumerate(meta['prop_cols']):
                # Значение по умолчанию берем среднее из метаданных или логичное
                # default_val = 20.0 if i == 0 else (0.2 if i == 3 else 100.0) 
                if i == 0:
                    default_val = 20.0
                elif i == 1:
                    default_val = 300.0
                elif i == 2:
                    default_val = 200.0
                elif i == 3:
                    default_val = 0.3
                elif i == 4:
                    default_val = 0.5
                    
                target_dict[prop] = st.number_input(f"Цель: {prop}", value=float(default_val), format="%.3f")

        with col_weight:
            st.subheader("Важность (веса)")
            for i, prop in enumerate(meta['prop_cols']):
                # Слайдер веса от 0.1 до 5.0
                w = st.slider(f"Вес для {prop}", 0.1, 5.0, 1.0, step=0.1, key=f"w_{i}")
                weights.append(w)

        submit_opt = st.form_submit_button("🚀 Найти идеальный рецепт")

    if submit_opt:
        with st.spinner("Математика работает... Оптимизируем состав"):
            # Вызываем функцию из core_logic
            best_cnt, best_m_idx, final_preds = solve_inverse_problem(
                target_dict, 
                np.array(weights), 
                bundle
            )

        # ВЫВОД РЕЗУЛЬТАТОВ
        st.success(f"### Оптимальное решение найдено!")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Рекомендуемый % УНТ", f"{best_cnt:.3f} %")
        with res_col2:
            st.metric("Метод смешивания", meta['method_labels'][best_m_idx])

        st.markdown("---")
        st.subheader("Сравнение результата с целью")

        # Формируем таблицу сравнения
        comparison_data = []
        for i, prop in enumerate(meta['prop_cols']):
            target = target_dict[prop]
            predicted = final_preds[i]
            diff_pct = ((predicted - target) / target) * 100
            
            comparison_data.append({
                "Свойство": prop,
                "Ваша цель": f"{target:.3f}",
                "Прогноз модели": f"{predicted:.3f}",
                "Отклонение (%)": f"{diff_pct:+.2f}%"
            })

        st.table(pd.DataFrame(comparison_data))

# --- ВКЛАДКА 3: ГРАФИКИ ---
with tab3:
    st.header("Анализ зависимостей")
    
    selected_prop_name = st.selectbox(
        "Выберите свойство для визуализации:",
        options=meta['prop_cols'],
        key="prop_selector_tab3"
    )
    prop_idx = meta['prop_cols'].index(selected_prop_name)

    # Кнопка теперь нужна только для ПЕРВОГО запуска или ОБНОВЛЕНИЯ
    if st.button("📈 Построить / Обновить график"):
        with st.spinner("Математика в процессе..."):
            from core_logic import get_plot_data
            plot_data = get_plot_data(prop_idx, bundle)

            fig = go.Figure()
            colors = ['#e74c3c', '#3498db', '#2ecc71']

            for i, (m_label, data) in enumerate(plot_data.items()):
                fig.add_trace(go.Scatter(
                    x=data['x'], y=data['y'],
                    mode='lines', name=m_label,
                    line=dict(color=colors[i], width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([data['x'], data['x'][::-1]]),
                    y=np.concatenate([data['upper'], data['lower'][::-1]]), # Больше никаких +2*std!
                    fill='toself', 
                    fillcolor=colors[i], 
                    opacity=0.2, 
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", 
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Зависимость: {selected_prop_name}",
                xaxis_title="Концентрация УНТ (%)",
                yaxis_title=selected_prop_name,
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Сохраняем готовую фигуру в "память"
            st.session_state['last_fig'] = fig
            st.session_state['last_fig_prop'] = selected_prop_name

    # ПРОВЕРКА: Если в памяти есть график — показываем его всегда
    if 'last_fig' in st.session_state:
        # Если пользователь выбрал другое свойство, но еще не нажал кнопку обновления
        if st.session_state.get('last_fig_prop') != selected_prop_name:
            st.warning(f"На графике ниже показано свойство '{st.session_state['last_fig_prop']}'. Нажмите кнопку выше, чтобы обновить данные для '{selected_prop_name}'.")
        
        st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
    else:
        st.info("Нажмите кнопку выше, чтобы построить график в первый раз.")


# --- ОБЩАЯ ТАБЛИЦА СРАВНЕНИЯ ---
if st.session_state['comparison_list']:
    st.markdown("---")
    st.subheader("📊 Управление сохраненными составами")
    
    df_comp = pd.DataFrame(st.session_state['comparison_list'])
    
    st.info("Выделите нужные строки галочками и нажмите 🗑️ в углу таблицы для удаления.")

    # Настраиваем редактор так, чтобы нельзя было менять текст в ячейках
    edited_df = st.data_editor(
        df_comp,
        key="main_comparison_editor",
        use_container_width=True,
        num_rows="dynamic",
        # Отключаем редактирование для всех колонок
        disabled=df_comp.columns.tolist(), 
        column_config={
            "ID": None, # Скрываем технический ID
        },
    )

    col_clear, col_export = st.columns(2)
    
    # with col_save:
    #     # Эта кнопка нужна, чтобы зафиксировать удаление в session_state
    #     if st.button("✅ Зафиксировать изменения", use_container_width=True):
    #         st.session_state['comparison_list'] = edited_df.to_dict('records')
    #         st.success("Список обновлен!")
    #         st.rerun()

    with col_clear:
        if st.button("🧹 Очистить всё", use_container_width=True):
            st.session_state['comparison_list'] = []
            st.rerun()
            
    with col_export:
        csv = edited_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "💾 Экспорт в CSV", 
            csv, 
            "composite_comparison.csv", 
            "text/csv",
            use_container_width=True
        )
