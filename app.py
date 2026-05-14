import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import re
import math
import io
import base64
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as _np
import wave
import struct
import requests
try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    canvas = None
    letter = None
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title='DeepCrack – AI Password Analyzer', layout='wide')

# -----------------------------
# Styles (Cyberpunk theme)
# -----------------------------
PRIMARY = '#00FF9C'
SECONDARY = '#00C2FF'
BG = '#0D1117'
WEAK = '#FF4C4C'
MED = '#FFD93D'
STRONG = PRIMARY

def local_css():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG}; color: #E6EDF3 }}
    .card {{ background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
             border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); transition: transform .18s;}}
    .card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 22px rgba(0,0,0,0.7); }}
    .neon-btn button, .stButton>button {{ background: linear-gradient(90deg,{PRIMARY}, {SECONDARY}); color: #001; padding:8px 14px; border-radius:10px; font-weight:700; border: none }}
    .badge {{ font-size:18px; font-weight:800; padding:8px 12px; border-radius:10px; color:#001; }}
    .roast-box {{ padding:12px; border-radius:10px; color:#071018; }}

    /* Animated progress */
    .progress-track {{ background:#071019; border-radius:12px; padding:6px; }}
    .progress-fill {{ height:18px; border-radius:8px; background: linear-gradient(90deg,{SECONDARY},{PRIMARY}); width:0%; animation:grow 900ms ease-out forwards; }}
    @keyframes grow {{ from {{ width:0%; }} to {{ width: var(--w); }} }}

    /* Roast gradient subtle animation */
    @keyframes roastGlow {{ 0% {{ filter:brightness(0.98) }} 50% {{ filter:brightness(1.02) }} 100% {{ filter:brightness(0.98) }} }}
    .roast-box.weak {{ background: linear-gradient(90deg, #ff8a8a, #ff6b6b); }}
    .roast-box.medium {{ background: linear-gradient(90deg, #ffe58a, #ffd93d); }}
    .roast-box.strong {{ background: linear-gradient(90deg, #bfffd4, #8cffb0); }}

    /* subtle floating */
    @keyframes floaty {{ 0% {{ transform: translateY(0px) }} 50% {{ transform: translateY(-6px) }} 100% {{ transform: translateY(0px) }} }}
    .floaty {{ animation: floaty 6s ease-in-out infinite; }}
    </style>
    """, unsafe_allow_html=True)

local_css()

# -----------------------------
# Feature & generator functions
# -----------------------------
common_words = [
    'password','admin','letmein','welcome','master','login','hello','sunshine','monkey','dragon',
    'football','baseball','superman','iloveyou','princess','rockyou','shadow','ninja','qwerty','abc123'
]

def is_keyboard_pattern(pwd):
    patterns = ['qwerty','asdfgh','zxcvbn','1qaz','zaq1','12345','qwertyuiop','asdfghjkl']
    low = str(pwd).lower()
    return any(p in low for p in patterns)

def has_repeat_chars(pwd):
    return bool(re.search(r'(.)\1{2,}', str(pwd)))

def shannon_entropy(pwd):
    pwd = str(pwd)
    if len(pwd) == 0:
        return 0.0
    probs = [float(pwd.count(c)) / len(pwd) for c in set(pwd)]
    return -sum([p * math.log(p, 2) for p in probs])

def has_dict_word(pwd):
    s = str(pwd).lower()
    return any(w in s for w in common_words)

def generate_strong_passphrase():
    word_list = ['correct','horse','battery','staple','coffee','tree','house','yellow','phone','cloud','tiger']
    words = random.sample(word_list, 4)
    sep = random.choice(['-','_','.', ''])
    return sep.join(words)

def generate_human_strong_password():
    words1 = ['Moon','Shadow','Cyber','Dragon','Pixel','Storm']
    words2 = ['Falcon','Tiger','Wizard','Hunter','Knight','Phoenix']
    symbols = ['@','#','$','!']
    return random.choice(words1)+random.choice(words2)+random.choice(symbols)+str(random.randint(10,9999))

def extract_features_advanced(pwd):
    pwd = str(pwd)
    length = len(pwd)
    digit_count = sum(c.isdigit() for c in pwd)
    upper_count = sum(c.isupper() for c in pwd)
    lower_count = sum(c.islower() for c in pwd)
    special_count = length - digit_count - upper_count - lower_count
    entropy = shannon_entropy(pwd)
    unique_ratio = len(set(pwd)) / length if length>0 else 0
    return {
        'length': length,
        'digit_count': digit_count,
        'upper_count': upper_count,
        'lower_count': lower_count,
        'special_count': special_count,
        'has_number': int(digit_count>0),
        'has_upper': int(upper_count>0),
        'has_lower': int(lower_count>0),
        'has_special': int(special_count>0),
        'is_keyboard': int(is_keyboard_pattern(pwd)),
        'has_dict_word': int(has_dict_word(pwd)),
        'has_repeat_chars': int(has_repeat_chars(pwd)),
        'entropy': entropy,
        'unique_ratio': unique_ratio
    }

# -----------------------------
# Model load / train utilities
# -----------------------------
MODEL_PATH = Path('model.pkl')

@st.cache_resource
def get_model_and_testdata(train_if_missing=True, train_n=15000):
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        # No testdata saved; return model only
        return model, None
    if not train_if_missing:
        return None, None

    # Train a RandomForest on a synthetic realistic dataset
    def generate_password_with_strength(strength):
        if strength == 'weak':
            return random.choice(['password','123456','qwerty','letmein','admin','111111','abc123'])
        if strength == 'medium':
            base = ''.join(random.choices(string.ascii_lowercase, k=random.randint(6,9)))
            return base + str(random.randint(1,99))
        return generate_human_strong_password()

    def generate_dataset(n=15000):
        pwds=[]; labels=[]
        dist={'weak':0.4,'medium':0.35,'strong':0.25}
        for _ in range(n):
            r=random.random()
            if r<dist['weak']:
                s='weak'
            elif r<dist['weak']+dist['medium']:
                s='medium'
            else:
                s='strong'
            pwds.append(generate_password_with_strength(s)); labels.append(s)
        return pd.DataFrame({'password':pwds,'strength_label':labels}).sample(frac=1).reset_index(drop=True)

    df = generate_dataset(train_n)
    feats = df['password'].apply(lambda p: extract_features_advanced(p))
    X = pd.DataFrame(feats.tolist())
    y = df['strength_label']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=120, max_depth=18, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    # compute small test items for confusion matrix
    return model, (X_test, y_test)

# -----------------------------
# Roast messages
# -----------------------------
weak_roasts = [
    "This password is the digital equivalent of a wet paper bag.",
    "That's so weak, even spaghetti has more backbone.",
    "Your password just got bullied off the playground.",
]
medium_roasts = [
    "Medium?! Like a half-empty glass of flat soda.",
    "Decent, but a hacker could still guess this while sleepwalking.",
    "Balanced, but risky.",
]
strong_messages = [
    "Now THAT's a fortress.",
    "Built like a cybersecurity gym rat.",
    "Gandalf approves.",
]

used_roasts = set()
def get_unique_roast(roast_list):
    global used_roasts
    available = [r for r in roast_list if r not in used_roasts]
    if not available:
        used_roasts.clear(); available = roast_list
    roast = random.choice(available); used_roasts.add(roast); return roast

# -----------------------------
# UI sections
# -----------------------------
def header():
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px'>
            <h1 class='floaty' style='color:{PRIMARY};text-shadow:0 0 18px rgba(0,255,156,0.8)'>🔐 DeepCrack – AI Password Analyzer</h1>
            <div style='color:{SECONDARY};margin-left:12px;opacity:0.9'>Cyberpunk Hacker Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

def password_input_section():
    st.sidebar.markdown('## Input')
    pw = st.sidebar.text_input('Enter your password...', type='password', key='pw')
    show = st.sidebar.checkbox('Show password', key='show_pw')
    if show and st.session_state.get('pw'):
        st.sidebar.text_input('Visible', value=st.session_state.get('pw'), key='visible_pw')
    # simplified: no generator buttons in sidebar
    analyze = st.sidebar.button('Analyze Password')
    return st.session_state.get('pw',''), analyze

def strength_badge_and_progress(score):
        if score<30:
                color=WEAK; label='WEAK 💀'; cls='weak'
        elif score<60:
                color=MED; label='MEDIUM 🤷'; cls='medium'
        else:
                color=STRONG; label='STRONG 🏆'; cls='strong'
        # progress bar via HTML using animated CSS --w variable
        st.markdown(f"""
        <div class='card floaty'>
            <div style='display:flex;align-items:center;justify-content:space-between'>
                <div class='badge { 'strong' if cls=='strong' else '' }' style='background:{color}'>{label}</div>
                <div style='flex:1;margin-left:16px'>
                    <div class='progress-track'>
                        <div class='progress-fill' style='--w:{score}%; background: {color}; box-shadow: 0 0 18px {color};'></div>
                    </div>
                </div>
                <div style='margin-left:12px;font-weight:800'>{score}/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return label

def crack_time(score):
    if score<30: return '2 seconds 💀'
    if score<60: return '3 days 🤷'
    return 'Centuries 🏆'

def roast_and_feedback(label):
    if 'WEAK' in label:
        roast = get_unique_roast(weak_roasts)
        st.markdown(f"<div class='roast-box weak'>{roast}</div>", unsafe_allow_html=True)
    elif 'MEDIUM' in label:
        roast = get_unique_roast(medium_roasts)
        st.markdown(f"<div class='roast-box medium'>{roast}</div>", unsafe_allow_html=True)
    else:
        roast = get_unique_roast(strong_messages)
        st.markdown(f"<div class='roast-box strong'>{roast}</div>", unsafe_allow_html=True)
    return roast

def feature_breakdown_card(features):
    df = pd.DataFrame([features]).T.reset_index()
    df.columns = ['feature','value']
    st.markdown('<div class="card"><h3 style="color:#9eeac9">Feature Breakdown</h3></div>', unsafe_allow_html=True)
    st.table(df)

def visualizations(model, vis_testdata=None, last_features=None):
    cols = st.columns([1,1])
    with cols[0]:
        st.subheader('Feature Importance')
        if model is not None:
            fi = model.feature_importances_
            feat_names = list(last_features.keys()) if last_features else []
            if not feat_names:
                feat_names = ['length','digit_count','upper_count','lower_count','special_count','entropy','unique_ratio','has_number','has_upper','has_special','is_keyboard','has_dict_word','has_repeat_chars']
            imp = pd.Series(fi, index=feat_names).sort_values(ascending=True)
            imp_top = imp.tail(10)
            fig = px.bar(imp_top, orientation='h', color=imp_top.values, color_continuous_scale=[SECONDARY, PRIMARY])
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color='#E6EDF3', transition={'duration':700,'easing':'cubic-in-out'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Model not available for feature importance')

    with cols[1]:
        st.subheader('Password Composition')
        if last_features is not None:
            parts = dict(digits=last_features['digit_count'], uppercase=last_features['upper_count'], lowercase=last_features['lower_count'], special=last_features['special_count'])
            fig = px.pie(names=list(parts.keys()), values=list(parts.values()), color_discrete_sequence=[SECONDARY, PRIMARY, MED, WEAK])
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color='#E6EDF3', transition={'duration':700,'easing':'cubic-in-out'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Analyze a password to see composition')

    st.subheader('Confusion Matrix')
    if vis_testdata is not None:
        X_test, y_test = vis_testdata
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=['weak','medium','strong'])
        fig = go.Figure(data=go.Heatmap(z=cm, x=['weak','medium','strong'], y=['weak','medium','strong'], colorscale='RdYlGn'))
        fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color='#E6EDF3', transition={'duration':700,'easing':'cubic-in-out'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Confusion matrix will appear after model training')

def generate_report(password, label, score, features, roast):
    lines = []
    lines.append('DeepCrack – Analysis Report')
    lines.append('===========================')
    lines.append(f'Password: {password}')
    lines.append(f'Strength: {label}')
    lines.append(f'Score: {score}/100')
    lines.append(f'Crack Time Estimate: {crack_time(score)}')
    lines.append('\nFeatures:')
    for k,v in features.items(): lines.append(f' - {k}: {v}')
    lines.append('\nRoast/Feedback:')
    lines.append(roast)
    return '\n'.join(lines).encode('utf-8')

def generate_tone_wav(frequency=880, duration=0.5, volume=0.3, samplerate=44100):
    # generate simple sine wave and return WAV bytes
    t = _np.linspace(0, duration, int(samplerate * duration), False)
    tone = _np.sin(frequency * t * 2 * _np.pi) * volume
    # convert to 16-bit PCM
    audio = (tone * 32767).astype(_np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()

def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def generate_pdf_report(password, label, score, features, roast):
    if not REPORTLAB_AVAILABLE:
        text = generate_report(password, label, score, features, roast)
        return text
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x = 40; y = height - 40
    c.setFont('Helvetica-Bold', 16)
    c.drawString(x, y, 'DeepCrack – Analysis Report')
    y -= 24
    c.setFont('Helvetica', 10)
    c.drawString(x, y, f'Password: {password}')
    y -= 14
    c.drawString(x, y, f'Strength: {label}    Score: {score}/100')
    y -= 18
    c.drawString(x, y, f'Crack Time Estimate: {crack_time(score)}')
    y -= 20
    c.drawString(x, y, 'Features:')
    y -= 14
    for k,v in features.items():
        c.drawString(x+10, y, f'- {k}: {v}')
        y -= 12
        if y < 60:
            c.showPage(); y = height - 40; c.setFont('Helvetica', 10)
    y -= 8
    c.drawString(x, y, 'Roast/Feedback:')
    y -= 14
    for line in roast.split('\n'):
        c.drawString(x+6, y, line)
        y -= 12
        if y < 60:
            c.showPage(); y = height - 40; c.setFont('Helvetica', 10)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

def render_score_indicator(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number={'suffix':'/100','font':{'color':'#E6EDF3','size':28}},
        gauge = {
            'axis': {'range': [0,100], 'visible': False},
            'bar': {'color': PRIMARY},
            'bgcolor': BG
        }
    ))
    fig.update_layout(height=200, margin={'t':10,'b':10,'l':10,'r':10}, paper_bgcolor=BG, plot_bgcolor=BG, transition={'duration':700})
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Main app flow
# -----------------------------
def main():
    header()
    model, vis_testdata = get_model_and_testdata()
    left, right = st.columns([2,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Analyze a Password')
        pw = st.text_input('Enter your password...', placeholder='Enter your password...', type='password', key='main_pw')
        col1, col2 = st.columns([1,1])
        analyze = col1.button('Analyze Password', key='analyze')
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze or pw:
            features = extract_features_advanced(pw)
            input_df = pd.DataFrame([features])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            # get strong probability if exists
            try:
                strong_idx = list(classes).index('strong')
                score = int(proba[strong_idx]*100)
            except ValueError:
                # fallback: map weak=20, medium=50, strong=85
                mapping = {'weak':20,'medium':50,'strong':85}
                score = mapping.get(pred,50)

            label = strength_badge_and_progress(score)
            st.write('Crack Time Estimate:', crack_time(score))
            # animated numeric indicator
            render_score_indicator(score)
            roast = roast_and_feedback(label)

            # Feature breakdown
            st.markdown('<div class="card">', unsafe_allow_html=True)
            feature_breakdown_card(features)
            st.markdown('</div>', unsafe_allow_html=True)

            # Visualizations
            visualizations(model, vis_testdata, last_features=features)

            # History
            h = st.session_state.get('history', [])
            h.insert(0, {'password':pw,'label':label,'score':score})
            st.session_state['history'] = h[:5]

            # Download report
            if REPORTLAB_AVAILABLE:
                pdf_bytes = generate_pdf_report(pw,label,score,features,roast)
                st.download_button('Download Report (PDF)', data=pdf_bytes, file_name='deepcrack_report.pdf', mime='application/pdf')
            else:
                st.warning('`reportlab` is not installed, so a text report fallback is being provided.')
                txt_bytes = generate_report(pw,label,score,features,roast)
                st.download_button('Download Report (TXT fallback)', data=txt_bytes, file_name='deepcrack_report.txt', mime='text/plain')

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('History (last 5)')
        for item in st.session_state.get('history', []):
            pwd = item.get('password','')
            masked = '(' + ('*' * len(pwd)) + ')'
            st.write(f"{masked} — {item['label']} — {item['score']}/100")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
    st.markdown('<footer style="color:#8aa">Built for demo & university projects.</footer>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
