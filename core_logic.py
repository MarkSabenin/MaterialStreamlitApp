import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import gpflow
import os
from scipy.optimize import minimize_scalar
# import openpy

# Скрываем мусор от TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def apply_fe(pct_arr, is_stat_arr, is_uv_arr, version):
    """
    Генерация признаков. 
    ВНИМАНИЕ: Формулы строго синхронизированы с кодом обучения.
    """
    base = np.column_stack([pct_arr, is_stat_arr, is_uv_arr])
    pct_col = pct_arr.reshape(-1, 1)
    # Добавляем крошечный эпсилон, чтобы не было деления на 0 при 0% УНТ
    eps = 0.001 

    if version == 'base': 
        return base
    if version == 'log_only': 
        return np.hstack([base, np.log(pct_col + 0.1)])
    if version == 'inv_only': 
        return np.hstack([base, 1.0 / (pct_col + 0.1)**2])
    if version == 'full': 
        return np.hstack([
            base, 
            np.log(pct_col + 0.1), 
            1.0 / (pct_col + eps)**2, 
            np.sin(pct_col)
        ])

def load_hybrid_system():
    """Загрузка моделей и РЕКОНСТРУКЦИЯ данных для GPflow"""
    meta = joblib.load('model_package/metadata.pkl')
    sc_y = joblib.load('model_package/scaler_y.pkl')
    scaler_x_gp = joblib.load('model_package/scaler_x_gp.pkl')
    
    # 1. Загрузка Sklearn моделей
    models_sk = {}
    scalers_x_sk = {}
    for k in meta['sk_indices']:
        models_sk[k] = joblib.load(f'model_package/sk_models/gpr_model_{k}.pkl')
        scalers_x_sk[k] = joblib.load(f'model_package/sk_models/scaler_x_{k}.pkl')
        
    # 2. РЕКОНСТРУКЦИЯ ДАННЫХ ДЛЯ GPFLOW (Критически важно!)
    # Модель GPR не работает без обучающих точек
    df = pd.read_excel("raw_data_van.xlsx")
    method_ohe = ['Статическое', 'УЗ+100', 'Статическое с смешением в УВ']
    df['method_code'] = df[method_ohe].values.argmax(axis=1)
    grp = df.groupby(['% УНТ/ 99% ПТФЭ', 'method_code'])
    
    # Повторяем логику подготовки Y_sc из скрипта обучения
    Y_raw = grp[meta['prop_cols']].mean().values
    Y_trans = Y_raw.copy()
    for idx in meta['log_indices']:
        Y_trans[:, idx] = np.log1p(Y_trans[:, idx])
    Y_sc_train = sc_y.transform(Y_trans)
    
    # Повторяем логику подготовки X_sc из скрипта обучения
    pct_vals = grp.mean().index.get_level_values(0).values
    m_vals = grp.mean().index.get_level_values(1).values
    is_stat = (m_vals == 0).astype(float)
    is_uv = (m_vals == 2).astype(float)
    
    X_e_gp = apply_fe(pct_vals, is_stat, is_uv, 'full')
    X_sc_gp_train = scaler_x_gp.transform(X_e_gp)
    
    # Создаем аугментированные данные (индекс свойства как признак)
    X_aug = np.vstack([np.append(X_sc_gp_train, np.full((X_sc_gp_train.shape[0], 1), float(j)), axis=1) for j in range(5)])
    Y_aug = np.vstack([Y_sc_train[:, j:j+1] for j in range(5)])
    
    # Инициализация модели GPflow
    num_feat_gp = X_e_gp.shape[1]
    k_smooth = gpflow.kernels.Matern52(lengthscales=[1.0]*num_feat_gp, active_dims=list(range(num_feat_gp)))
    coreg = gpflow.kernels.Coregion(output_dim=5, rank=2, active_dims=[num_feat_gp])
    
    model_gp = gpflow.models.GPR(data=(X_aug, Y_aug), kernel=k_smooth * coreg)
    
    # Восстановление весов из чекпоинта
    ckpt = tf.train.Checkpoint(model=model_gp)
    latest = tf.train.latest_checkpoint('model_package/gpflow_weights/')
    if latest:
        ckpt.restore(latest).expect_partial()
    
    return {
        'models_sk': models_sk,
        'scalers_x_sk': scalers_x_sk,
        'model_gp': model_gp,
        'scaler_x_gp': scaler_x_gp,
        'sc_y': sc_y,
        'meta': meta
    }

def predict_hybrid(cnt, method_idx, bundle):
    """Прямая задача с учетом логарифмирования и правильного OHE"""
    meta = bundle['meta']
    sc_y = bundle['sc_y']
    
    # Правильный вектор метода (как в обучении)
    # 0 (Стат) -> [1,0], 2 (УВ) -> [0,1], 1 (УЗ) -> [0,0]
    m_vec = [1.0, 0.0] if method_idx == 0 else ([0.0, 1.0] if method_idx == 2 else [0.0, 0.0])
    
    pct_arr = np.array([float(cnt)])
    is_stat_arr, is_uv_arr = np.array([m_vec[0]]), np.array([m_vec[1]])
    
    preds, stds = np.zeros(5), np.zeros(5)

    # 1. Расчет через Sklearn (Прочность, Удлинение, Упругость)
    for k in meta['sk_indices']:
        fe_version = meta['sk_fe_versions'].get(k, 'base')
        x_e = apply_fe(pct_arr, is_stat_arr, is_uv_arr, fe_version)
        x_sc = bundle['scalers_x_sk'][k].transform(x_e)
        
        mu_sc, std_sc = bundle['models_sk'][k].predict(x_sc, return_std=True)
        
        # Де-скейлинг (мультипликативный для std)
        mu_orig = mu_sc[0] * sc_y.scale_[k] + sc_y.mean_[k]
        std_orig = std_sc[0] * sc_y.scale_[k]
        
        # Обратный логарифм если индекс в логах
        if k in meta['log_indices']:
            preds[k] = np.expm1(mu_orig)
            # Для интервала на графике: аппроксимируем std через экспоненту
            stds[k] = std_orig 
        else:
            preds[k] = mu_orig
            stds[k] = std_orig

    # 2. Расчет через GPflow (Трение, Износ)
    x_e_gp = apply_fe(pct_arr, is_stat_arr, is_uv_arr, 'full')
    x_sc_gp = bundle['scaler_x_gp'].transform(x_e_gp)
    
    for k in meta['gp_indices']:
        x_aug_gp = np.append(x_sc_gp, [[float(k)]], axis=1)
        mu_sc_gp, var_sc_gp = bundle['model_gp'].predict_y(x_aug_gp)
        
        mu_orig = mu_sc_gp.numpy()[0, 0] * sc_y.scale_[k] + sc_y.mean_[k]
        std_orig = np.sqrt(var_sc_gp.numpy()[0, 0]) * sc_y.scale_[k]
        
        if k in meta['log_indices']:
            preds[k] = np.expm1(mu_orig)
            stds[k] = std_orig
        else:
            preds[k] = mu_orig
            stds[k] = std_orig

    return preds, stds

# Функции инверсии и отрисовки остаются прежними, так как они используют predict_hybrid
def solve_inverse_problem(target_dict, weights, bundle):
    # (Код без изменений, он опирается на исправленный predict_hybrid)
    meta = bundle['meta']
    sc_y = bundle['sc_y']
    y_target_vec = np.zeros(5)
    for i, prop in enumerate(meta['prop_cols']):
        val = target_dict.get(prop, 0.0)
        y_target_vec[i] = np.log1p(val) if i in meta['log_indices'] else val
    y_target_sc = sc_y.transform(y_target_vec.reshape(1, -1)).flatten()
    best_loss, best_cnt, best_m_idx = float('inf'), 1.0, 0
    for m_idx in [0, 1, 2]:
        def objective(cnt):
            preds, _ = predict_hybrid(cnt, m_idx, bundle)
            p_trans = np.zeros(5)
            for i in range(5):
                p_trans[i] = np.log1p(preds[i]) if i in meta['log_indices'] else preds[i]
            p_sc = sc_y.transform(p_trans.reshape(1, -1)).flatten()
            return np.sum(weights * (p_sc - y_target_sc)**2)
        res = minimize_scalar(objective, bounds=(1.0, 5.0), method='bounded')
        if res.fun < best_loss:
            best_loss, best_cnt, best_m_idx = res.fun, res.x, m_idx
    final_preds, _ = predict_hybrid(best_cnt, best_m_idx, bundle)
    return best_cnt, best_m_idx, final_preds

def get_plot_data(prop_idx, bundle):
    pct_grid = np.linspace(1.0, 5.0, 100)
    plot_results = {}
    meta = bundle['meta']
    
    for m_idx, m_label in enumerate(meta['method_labels']):
        preds_l, stds_l = [], []
        for p in pct_grid:
            mu, std = predict_hybrid(p, m_idx, bundle)
            preds_l.append(mu[prop_idx])
            stds_l.append(std[prop_idx])
        
        mu_arr = np.array(preds_l)
        std_arr = np.array(stds_l)
        
        # КРИТИЧЕСКИЙ МОМЕНТ: Расчет интервалов
        if prop_idx in meta['log_indices']:
            # Если свойство было логарифмировано, std — это ошибка в лог-пространстве
            log_mu = np.log1p(mu_arr)
            lower = np.expm1(log_mu - 2 * std_arr)
            upper = np.expm1(log_mu + 2 * std_arr)
        else:
            # Обычный линейный случай
            lower = mu_arr - 2 * std_arr
            upper = mu_arr + 2 * std_arr

        plot_results[m_label] = {
            'x': pct_grid, 
            'y': mu_arr, 
            'lower': lower, 
            'upper': upper
        }
    return plot_results