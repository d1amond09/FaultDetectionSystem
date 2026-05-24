import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import traceback

from src.infrastructure.config import MODEL_DIR, DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEQUENCE_LENGTH
from src.infrastructure.pytorch_models import (
    LSTMAutoencoder, GRUAutoencoder, Conv1DAutoencoder,
    VAEAutoencoder, TransformerAutoencoder, TCNAutoencoder
)


def create_sequences(data, seq_length):
    return np.array([data[i:(i + seq_length)] for i in range(len(data) - seq_length)])


def remove_outlier_sequences_iqr(sequences, factor=2.5):
    n_samples, seq_len, n_features = sequences.shape
    flat = sequences.reshape(-1, n_features)
    Q1, Q3 = np.percentile(flat, 25, axis=0), np.percentile(flat, 75, axis=0)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    mask = np.ones(n_samples, dtype=bool)
    for i in range(n_features):
        mask &= ~((sequences[:, :, i] < lower[i]) | (sequences[:, :, i] > upper[i])).any(axis=1)
    return sequences[mask]


def train_model(model, loader, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for _ in range(1, EPOCHS + 1):
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            if hasattr(model, 'is_vae'):
                recon, mu, logvar = model(batch_x)
                loss = criterion(recon, batch_x) + 0.001 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
            else:
                loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model


def compute_errors(model, loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            out = model(batch_x)
            if hasattr(model, 'is_vae'):
                out = out[0]
            loss = torch.mean(torch.abs(out - batch_x), dim=(1, 2)).cpu().numpy()
            errors.extend(loss)
    return np.array(errors)


def iterative_cleaning(model_class, sequences, n_iter=2, keep_top=97):
    cleaned = sequences.copy()
    for i in range(n_iter):
        num_feats = cleaned.shape[2]
        model = model_class(num_feats).to(DEVICE)
        loader = torch.utils.data.DataLoader(torch.tensor(cleaned, dtype=torch.float32),
                                             batch_size=BATCH_SIZE, shuffle=True)
        criterion = nn.HuberLoss(reduction='sum', delta=1.0).to(DEVICE)
        train_model(model, loader, criterion)
        eval_loader = torch.utils.data.DataLoader(torch.tensor(cleaned, dtype=torch.float32),
                                                  batch_size=BATCH_SIZE, shuffle=False)
        errors = compute_errors(model, eval_loader)
        threshold = np.percentile(errors, keep_top)
        cleaned = cleaned[errors <= threshold]
    return cleaned


def run_full_training(files, model_choice='all', log_callback=None):
    if log_callback is None:
        log_callback = print

    log_callback("Начало обучения...")
    try:
        dfs = []
        for path in files:
            df = pd.read_csv(path)
            df.drop(columns=['Столбец_0', 'Столбец_1'], inplace=True, errors='ignore')
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True).fillna(0)
        valid_cols_new = [c for c in combined.columns if combined[c].nunique() > 1]
        joblib.dump(valid_cols_new, os.path.join(MODEL_DIR, 'valid_cols.pkl'))
        log_callback(f"Загружено окон: {len(combined)}, признаков: {len(valid_cols_new)}")

        df_train = combined[valid_cols_new].astype(np.float32)
        scaler_new = MinMaxScaler()
        scaled_train = scaler_new.fit_transform(df_train)
        joblib.dump(scaler_new, os.path.join(MODEL_DIR, 'scaler.pkl'))
        log_callback("Масштабирование выполнено")

        X_train = create_sequences(scaled_train, SEQUENCE_LENGTH)
        log_callback(f"Последовательностей до очистки: {len(X_train)}")
        X_train_iqr = remove_outlier_sequences_iqr(X_train, factor=3.0)
        log_callback(f"После IQR: {len(X_train_iqr)}")
        X_train_clean = iterative_cleaning(LSTMAutoencoder, X_train_iqr, n_iter=2, keep_top=98)
        log_callback(f"После итеративной очистки: {len(X_train_clean)}")

        num_features = X_train_clean.shape[2]
        clean_tensor = torch.tensor(X_train_clean, dtype=torch.float32)
        clean_loader = torch.utils.data.DataLoader(clean_tensor, batch_size=BATCH_SIZE, shuffle=True)
        criterion = nn.HuberLoss(reduction='sum', delta=1.0).to(DEVICE)

        try:
            with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'r') as f:
                new_thresholds = json.load(f)
        except:
            new_thresholds = {}

        if model_choice in ('all', 'nn'):
            models_dict = {
                'lstm': LSTMAutoencoder,
                'gru': GRUAutoencoder,
                'cnn': Conv1DAutoencoder,
                'vae': VAEAutoencoder,
                'transformer': TransformerAutoencoder,
                'tcn': TCNAutoencoder
            }
            for name, cls in models_dict.items():
                model = cls(num_features).to(DEVICE)
                train_model(model, clean_loader, criterion)
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}.pth'))
                eval_loader = torch.utils.data.DataLoader(clean_tensor, batch_size=BATCH_SIZE, shuffle=False)
                errs = compute_errors(model, eval_loader)
                smooth = pd.Series(errs).rolling(10, min_periods=1).mean().values
                new_thresholds[name] = float(np.percentile(smooth, 95))
                log_callback(f"{name}: порог {new_thresholds[name]:.4f}")

        X_flat = X_train_clean.reshape(len(X_train_clean), -1)
        X_flat += np.random.normal(0, 1e-6, X_flat.shape)

        if model_choice in ('all', 'ml'):
            iforest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1).fit(X_flat)
            joblib.dump(iforest, os.path.join(MODEL_DIR, 'iforest.pkl'))
            new_thresholds['iforest'] = float(np.percentile(-iforest.score_samples(X_flat), 95))
            log_callback(f"iforest: порог {new_thresholds['iforest']:.4f}")

            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.02).fit(X_flat)
            joblib.dump(ocsvm, os.path.join(MODEL_DIR, 'ocsvm.pkl'))
            new_thresholds['ocsvm'] = float(np.percentile(-ocsvm.decision_function(X_flat), 95))
            log_callback(f"ocsvm: порог {new_thresholds['ocsvm']:.4f}")

            lof = LocalOutlierFactor(novelty=True, contamination=0.02).fit(X_flat)
            joblib.dump(lof, os.path.join(MODEL_DIR, 'lof.pkl'))
            new_thresholds['lof'] = float(np.percentile(-lof.score_samples(X_flat), 95))
            log_callback(f"lof: порог {new_thresholds['lof']:.4f}")

            pca_ee = PCA(n_components=0.95, random_state=42).fit(X_flat)
            joblib.dump(pca_ee, os.path.join(MODEL_DIR, 'pca_ee_transformer.pkl'))
            X_pca_ee = pca_ee.transform(X_flat)
            ee = EllipticEnvelope(contamination=0.02, random_state=42, support_fraction=0.9).fit(X_pca_ee)
            joblib.dump(ee, os.path.join(MODEL_DIR, 'elliptic.pkl'))
            new_thresholds['elliptic'] = float(np.percentile(ee.mahalanobis(X_pca_ee), 95))
            log_callback(f"elliptic: порог {new_thresholds['elliptic']:.4f}")

            pca = PCA(n_components=0.95, random_state=42).fit(X_flat)
            joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))
            X_recon = pca.inverse_transform(pca.transform(X_flat))
            pca_errs = np.mean(np.square(X_flat - X_recon), axis=1)
            new_thresholds['pca'] = float(np.percentile(pca_errs, 95))
            log_callback(f"pca: порог {new_thresholds['pca']:.4f}")

        with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f:
            json.dump(new_thresholds, f, indent=2)
        log_callback("Обучение завершено успешно. Пороги обновлены.")
    except Exception as e:
        log_callback(f"ОШИБКА: {str(e)}")
        log_callback(traceback.format_exc())