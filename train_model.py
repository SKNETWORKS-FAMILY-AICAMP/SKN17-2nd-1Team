# model 폴더(joblib)가 없을 경우 실행하는 파일
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import os  # os 모듈 추가

# --- 상수 정의 ---
FILE_PATH = './data/'
MODEL_PATH = './model/'

# --- 폴더 생성 로직 추가 ---
# 모델을 저장할 폴더가 없으면 자동으로 생성합니다.
os.makedirs(MODEL_PATH, exist_ok=True)

# --- 데이터 로드 ---
print("데이터를 로드합니다...")
df_fin = pd.read_csv(f'{FILE_PATH}survey_results_cleaned_scaler.csv')

X = df_fin.drop('is_churned', axis=1)
y = df_fin['is_churned']

# --- 스케일러 학습 및 저장 ---
print("MinMaxScaler를 학습하고 저장합니다...")
scaler = MinMaxScaler()
scaler.fit(X)
dump(scaler, f"{MODEL_PATH}scaler.joblib")
print(f"✅ 스케일러가 '{MODEL_PATH}scaler.joblib'에 저장되었습니다.")

# --- 모델 학습 및 저장 ---
print("CatBoost 모델을 학습하고 저장합니다...")
X_scaled = scaler.transform(X) # 스케일링 적용
model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, 
                           loss_function='Logloss', eval_metric='AUC',
                           random_seed=42, verbose=100)
model.fit(X_scaled, y)
dump(model, f"{MODEL_PATH}catboost_model.joblib")
print(f"✅ 모델이 '{MODEL_PATH}catboost_model.joblib'에 저장되었습니다.")