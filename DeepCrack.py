"""
DeepCrack - Password Strength Analyzer
Random Forest based password strength classification with interactive feedback.
"""
import pandas as pd
import numpy as np
import random
import string
import re
import math
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


def is_keyboard_pattern(pwd):
    patterns = [
        'qwerty', 'asdfgh', 'zxcvbn', '1qaz', 'zaq1', '12345',
        'qwertyuiop', 'asdfghjkl', 'zxcvbnm', '0987654321', '!@#$%'
    ]
    low_pwd = pwd.lower()
    return any(pattern in low_pwd for pattern in patterns)


def has_repeat_chars(pwd):
    return bool(re.search(r'(.)\1{2,}', pwd))


def shannon_entropy(pwd):
    probs = [float(pwd.count(c)) / len(pwd) for c in set(pwd)]
    return -sum([p * math.log(p, 2) for p in probs])


common_words = [
    'password', 'admin', 'letmein', 'welcome', 'master', 'login', 'hello',
    'sunshine', 'monkey', 'dragon', 'football', 'baseball', 'superman',
    'iloveyou', 'princess', 'rockyou', 'shadow', 'ninja', 'whatever',
    'qwerty', 'abc123', 'trustno1', 'freedom', 'pokemon', 'charlie',
    'donald', 'michael', 'jessica', 'hunter', 'cookie', 'pepper',
    'matrix', 'internet', 'computer', 'secret', 'killer', 'soccer',
    'batman', 'thomas', 'tigger', 'summer', 'winter', 'harley',
    'mustang', 'access', 'starwars', 'cheese', 'banana', 'orange',
    'pokemon123', 'love123', 'babygirl', 'flower', 'purple',
    'ashley', 'jordan', 'daniel', 'buster', 'george', 'assassin',
    'naruto', 'goku', 'anime', 'gaming', 'minecraft', 'fortnite',
    'iphone', 'samsung', 'linkedin', 'google', 'facebook',
    'instagram', 'youtube', 'tiktok', 'pass123', 'passw0rd',
    'welcome123', 'manager', 'administrator', 'security',
    'pakistan', 'karachi', 'lahore', 'islamabad'
]


def has_dict_word(pwd):
    return any(word in pwd.lower() for word in common_words)


def generate_strong_passphrase():
    word_list = [
        'correct', 'horse', 'battery', 'staple', 'coffee', 'tree', 'house', 'yellow',
        'phone', 'cloud', 'tiger', 'eagle', 'ocean', 'mountain', 'puzzle', 'wizard',
        'hammer', 'falcon', 'spider', 'galaxy'
    ]
    words = random.sample(word_list, 4)
    sep = random.choice(['-', '_', '.', ''])
    return sep.join(words)


def generate_human_strong_password():
    words1 = ['Moon', 'Shadow', 'Cyber', 'Dragon', 'Pixel', 'Storm']
    words2 = ['Falcon', 'Tiger', 'Wizard', 'Hunter', 'Knight', 'Phoenix']
    symbols = ['@', '#', '$', '!']
    return random.choice(words1) + random.choice(words2) + random.choice(symbols) + str(random.randint(10, 9999))


def generate_password_with_strength(strength):
    if strength == 'weak':
        if random.random() < 0.75:
            patterns = [
                lambda: random.choice(['password', '123456', 'qwerty', 'admin', 'letmein', 'welcome', 'iloveyou', 'princess']),
                lambda: random.choice(['111111', 'abc123', 'passw0rd', 'football', 'monkey', 'dragon', 'master']),
                lambda: random.choice(['qwertyuiop', 'asdfghjkl', 'zxcvbnm', '1qaz2wsx', 'zaq1xsw2']),
                lambda: random.choice(['Password@123', 'P@ssword1', 'Karachi786', 'Ali12345', 'Ahmed786', 'Pakistan1', 'Qwerty@123', 'Admin@123']),
                lambda: f"{random.randint(1900, 2025)}",
                lambda: f"{random.choice(['john', 'mike', 'sarah', 'jessica'])}{random.randint(1, 99)}",
                lambda: random.choice(['aaa', 'bbb', '111', '!!!', 'ddd']) * 2
            ]
            return random.choice(patterns)()
        else:
            length = random.randint(4, 7)
            return ''.join(random.choices(string.ascii_lowercase, k=length))

    elif strength == 'medium':
        if random.random() < 0.6:
            word1 = random.choice(['dog', 'cat', 'sun', 'sky', 'blue', 'red', 'run', 'fast', 'love', 'hate'])
            word2 = random.choice(['123', '2025', '!', '??', 'x', 'one', 'two'])
            return word1 + word2
        else:
            length = random.randint(8, 10)
            chars = string.ascii_letters + string.digits
            pwd = ''.join(random.choices(chars, k=length))
            if not any(c.isupper() for c in pwd):
                pwd += random.choice(string.ascii_uppercase)
            if not any(c.isdigit() for c in pwd):
                pwd += random.choice(string.digits)
            return pwd

    else:  # strong
        if random.random() < 0.3:
            return generate_human_strong_password()
        elif random.random() < 0.6:
            return generate_strong_passphrase()
        else:
            length = random.randint(12, 18)
            chars = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?"
            return ''.join(random.choices(chars, k=length))


def generate_dataset(n=100000):
    passwords = []
    strengths = []
    dist = {'weak': 0.4, 'medium': 0.35, 'strong': 0.25}
    for _ in range(n):
        r = random.random()
        if r < dist['weak']:
            s = 'weak'
        elif r < dist['weak'] + dist['medium']:
            s = 'medium'
        else:
            s = 'strong'
        pwd = generate_password_with_strength(s)
        passwords.append(pwd)
        strengths.append(s)
    df = pd.DataFrame({'password': passwords, 'strength_label': strengths})
    return df.sample(frac=1).reset_index(drop=True)


def extract_features_advanced(pwd):
    pwd = str(pwd)
    length = len(pwd)
    digit_count = sum(c.isdigit() for c in pwd)
    upper_count = sum(c.isupper() for c in pwd)
    lower_count = sum(c.islower() for c in pwd)
    special_count = length - digit_count - upper_count - lower_count
    entropy = shannon_entropy(pwd)
    unique_ratio = len(set(pwd)) / length if length > 0 else 0
    return {
        'length': length,
        'digit_count': digit_count,
        'upper_count': upper_count,
        'lower_count': lower_count,
        'special_count': special_count,
        'has_number': int(digit_count > 0),
        'has_upper': int(upper_count > 0),
        'has_lower': int(lower_count > 0),
        'has_special': int(special_count > 0),
        'is_keyboard': int(is_keyboard_pattern(pwd)),
        'has_dict_word': int(has_dict_word(pwd)),
        'has_repeat_chars': int(has_repeat_chars(pwd)),
        'entropy': entropy,
        'unique_ratio': unique_ratio
    }


print("Generating 100,000 realistic passwords...")
df = generate_dataset(100000)
print("\nDataset distribution:")
print(df['strength_label'].value_counts())

features_df = df['password'].apply(extract_features_advanced)
X = pd.DataFrame(features_df.tolist())
y = df['strength_label']

print("\nTraining Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.2%}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['weak', 'medium', 'strong'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['weak', 'medium', 'strong'], yticklabels=['weak', 'medium', 'strong'])
plt.title('Confusion Matrix')
plt.show()
print(df)


weak_roasts = [
    "This password is the digital equivalent of a wet paper bag.",
    "That's so weak, even spaghetti has more backbone.",
    "I've seen more security in a screen door on a submarine.",
    "Your password just got bullied off the playground.",
    "This password is what you'd set if you wanted to lose a hacking minigame.",
    "Weak! It took longer to type than to crack.",
    "'password123' called – it wants its loser crown back.",
    "This password is basically leaving your front door open with a 'steal me' sign.",
    "0 creativity points.",
    "Weak sauce.",
    "This password died before it was born.",
    "A toddler with sticky fingers could hack this.",
    "Your cybersecurity stock just crashed.",
    "This password folds under pressure instantly.",
    "Even NPCs use stronger passwords.",
    "This belongs in the digital toilet.",
    "Hackers cracked this accidentally while sneezing.",
    "Amazon package security is stronger than this."
]

medium_roasts = [
    "Medium?! Like a half-empty glass of flat soda.",
    "Decent, but a hacker could still guess this while sleepwalking.",
    "Medium energy.",
    "Could be worse.",
    "Rusty shield energy.",
    "Better than 'password'. Barely.",
    "One dictionary attack away from tears.",
    "Balanced, but risky.",
    "Functional. Barely.",
    "This password rides without training wheels but still crashes.",
    "Your password studied 10 minutes before the exam.",
    "Mid-tier anime side character energy.",
    "Secure enough for toast. Not banking."
]

strong_messages = [
    "Now THAT's a fortress.",
    "Built like a cybersecurity gym rat.",
    "Gandalf approves.",
    "Encrypted with hacker tears.",
    "Password bingo winner.",
    "Hackers hate this one trick.",
    "I'd recommend this to my own servers.",
    "NATO wants to borrow this password.",
    "This password entered the cybersecurity hall of fame.",
    "Built different.",
    "This password pays taxes and respects its mother.",
    "Hackers saw this and chose another victim."
]

used_roasts = set()

def get_unique_roast(roast_list):
    global used_roasts
    available = [r for r in roast_list if r not in used_roasts]
    if not available:
        used_roasts.clear()
        available = roast_list
    roast = random.choice(available)
    used_roasts.add(roast)
    return roast


print("\nDeepCrack loaded successfully.")

while True:
    pwd = input("\nEnter password (or type exit): ")

    if pwd.lower() == "exit":
        break

    feats = extract_features_advanced(pwd)
    input_df = pd.DataFrame([feats])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_

    strength_percent = int(proba[np.where(classes == 'strong')[0][0]] * 100)

    print("\nResult:", pred.upper())
    print("Strength Score:", strength_percent, "/100")

    if strength_percent < 30:
        print("Crack Time: 2 seconds 💀")
    elif strength_percent < 60:
        print("Crack Time: 3 days 🤷")
    else:
        print("Crack Time: centuries 🏆")