import string
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Emotion Detector", page_icon="🙂")

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}
COLORS = {"joy": "#F5B83D", "sadness": "#5B8DEF", "anger": "#E5484D",
          "fear": "#8E6BD9", "love": "#EC6FA6", "surprise": "#2BB8A8"}

@st.cache_resource
def load_model():
    return joblib.load("emotion_model.joblib")

@st.cache_resource
def load_stop_words():
    with open("stop_words.txt") as f:
        return set(f.read().split("\n"))

model = load_model()
stop_words = load_stop_words()

def clean(txt):
    txt = txt.lower()
    txt = " ".join(w for w in txt.split() if w not in stop_words)
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    return txt

def predict(line):
    cleaned = clean(line)
    if not cleaned.strip():
        return None, cleaned
    probs = pd.Series(model.predict_proba([cleaned])[0], index=model.classes_)
    return probs.sort_values(ascending=False), cleaned

def card(emoji, title, subtitle, color):
    return (f'<div style="border-left:6px solid {color};background:{color}22;padding:14px 18px;'
            f'border-radius:10px;margin:6px 0 10px 0;">'
            f'<span style="font-size:1.6rem">{emoji}</span>'
            f'<span style="font-size:1.3rem;font-weight:700;margin-left:8px">{title}</span>'
            f'<span style="opacity:.75;margin-left:10px">{subtitle}</span></div>')

def bars(probs):
    rows = ""
    for emo, p in probs.items():
        rows += (f'<div style="display:flex;align-items:center;margin:4px 0;font-size:0.9rem">'
                 f'<div style="width:100px">{EMOJI[emo]} {emo}</div>'
                 f'<div style="flex:1;background:#ffffff14;border-radius:6px;height:10px">'
                 f'<div style="width:{p*100:.1f}%;background:{COLORS[emo]};height:10px;border-radius:6px"></div></div>'
                 f'<div style="width:52px;text-align:right">{p:.0%}</div></div>')
    return rows

st.title("Emotion Detector")
st.write("Write one or more sentences, one per line.")
st.caption('Works best on short first-person text like "i feel ...". Trained on tweets, so other styles may be less accurate.')

text = st.text_area("Your text", placeholder="i feel so happy today", height=120)

if st.button("Detect emotion", type="primary"):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        st.warning("Please write something first.")
    for line in lines:
        st.markdown(f"**“{line}”**")
        probs, cleaned = predict(line)
        if probs is None:
            st.warning("Too short for the model, please write a little more.")
            continue
        top, conf = probs.index[0], probs.iloc[0]
        if conf < 0.5:
            second, p2 = probs.index[1], probs.iloc[1]
            st.markdown(card("🤔", "Not sure", f"closest: {top} {conf:.0%}, then {second} {p2:.0%}", "#8A8F98"),
                        unsafe_allow_html=True)
        else:
            st.markdown(card(EMOJI[top], top.capitalize(), f"{conf:.0%} confident", COLORS[top]),
                        unsafe_allow_html=True)
        st.markdown(bars(probs), unsafe_allow_html=True)
        st.caption(f"Model saw: {cleaned}")
        st.divider()

with st.expander("About this model"):
    st.write("Logistic Regression on TF-IDF features, trained on 12,800 English tweets and tested on 3,200 "
             "(accuracy about 88%, macro F1 0.85). Weakest spots: **surprise** (precision about 0.66) and "
             "negation such as 'not happy', because the model counts words and does not understand their order.")