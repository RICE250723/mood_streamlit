# -*- coding: utf-8 -*-
# app.py

# -----------------------
# 📦 必要なライブラリのインポート
# -----------------------
import os
from datetime import datetime

import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

# -----------------------
# 📁 モデル・トークナイザー・ラベルエンコーダーの読み込み
# -----------------------
@st.cache_resource(show_spinner="モデルを読み込んでいます...")
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("RICE250727/mood-classifier")
    tokenizer = BertTokenizer.from_pretrained("RICE250727/mood-classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# ラベルエンコーダーもオンラインから
label_encoder_path = hf_hub_download(repo_id="RICE250727/mood-classifier", filename="label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path)

# -----------------------
# 🔍 感情分析の推論関数
# -----------------------
def analyze_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    emotion = label_encoder.inverse_transform([pred_label])[0]
    return emotion, confidence

# -----------------------
# 📚 ログ保存関数
# -----------------------
def save_log(entry, emotion, confidence):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = "logs/user_log.csv"
    with open(log_path, "a", encoding="utf-8") as f:
        safe_entry = entry.replace(",", " ")  # CSV衝突回避
        f.write(f'"{safe_entry}",{emotion},{confidence:.2f},{timestamp}\n')

# -----------------------
# 🖼️ Streamlit UI
# -----------------------
st.title("🧠 HabitMood - 日記感情分析アプリ")
st.write("日記から感情を分類し、振り返りに役立てましょう。")

# --- セッションステートの初期化 ---
if "diary_input" not in st.session_state:
    st.session_state.diary_input = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if st.session_state.clear_input:
    st.session_state.diary_input = ""
    st.session_state.clear_input = False

# --- 入力欄 ---
user_input = st.text_area(
    "📝 今日の出来事や感情を自由に記録してください：",
    height=200,
    key="diary_input"
)

# --- Analyze ボタン ---
if st.button("Analyze"):
    text = st.session_state.diary_input.strip()
    if not text:
        st.warning("テキストを入力してください。")
    else:
        emotion, confidence = analyze_text(text)
        st.success(f"**感情:** {emotion}（信頼度: {confidence:.2f}）")

        if emotion == "Negative":
            st.info("そう感じた理由を少し振り返ってみましょう。")

        save_log(text, emotion, confidence)
        st.session_state.clear_input = True
        st.rerun()

# -----------------------
# 📈 ログ表示
# -----------------------
st.subheader("📜 過去の記録（感情ごとに色分け）")
log_path = "logs/user_log.csv"

if os.path.exists(log_path):
    df_logs = pd.read_csv(log_path, names=["テキスト", "感情", "信頼度", "日時"])
    df_logs["日時"] = pd.to_datetime(df_logs["日時"], errors="coerce")
    df_logs = df_logs.dropna(subset=["日時"]).sort_values("日時", ascending=False).reset_index(drop=True)

    # 色付き表示用のスタイル関数
    def color_emotion(val):
        return {
            "Positive": "color: #dc2626; font-weight: bold;",
            "Negative": "color: #2563eb; font-weight: bold;",
            "Neutral":  "color: #16a34a; font-weight: bold;",
        }.get(val, "")

    styled_df = df_logs.style.applymap(color_emotion, subset=["感情"]).set_table_styles([
        {"selector": "td", "props": [("white-space", "pre-wrap"), ("max-width", "400px")]},
        {"selector": "th", "props": [("text-align", "center")]}
    ])

    st.dataframe(df_logs.style.applymap(color_emotion, subset=["感情"]))
else:
    st.info("まだ記録がありません。")