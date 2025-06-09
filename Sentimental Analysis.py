import os
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import time

# Define directories
DEFAULT_POSITIVE_DIR = r"C:\Users\Lenovo\Downloads\New folder\movie_reviews\movie_reviews\pos"
DEFAULT_NEGATIVE_DIR = r"C:\Users\Lenovo\Downloads\New folder\movie_reviews\movie_reviews\neg"

# Sentiment lexicons (full lists as requested)
POSITIVE_WORDS = [
    'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'superb', 'brilliant',
    'awesome', 'outstanding', 'perfect', 'terrific', 'fabulous', 'marvelous', 'stellar',
    'enjoyable', 'engaging', 'incredible', 'delightful', 'impressive', 'inspiring',
    'touching', 'heartwarming', 'fun', 'entertaining', 'captivating', 'masterpiece',
    'beautiful', 'charming', 'uplifting', 'memorable', 'well-done', 'well-made',
    'well-acted', 'well-written', 'clever', 'smart', 'thoughtful', 'moving', 'hilarious',
    'funny', 'laugh', 'loved', 'love', 'favorite', 'recommend', 'must-see', 'enjoyed',
    'spectacular', 'breathtaking', 'refreshing', 'unique', 'creative', 'original',
    'strong', 'powerful', 'emotional', 'satisfying', 'rewarding', 'solid', 'top-notch',
    'phenomenal', 'riveting', 'absorbing', 'gripping', 'remarkable', 'exceptional',
    'flawless', 'genius', 'well-crafted', 'well-executed', 'well-directed', 'well-cast',
    'well-paced', 'well-shot', 'well-produced', 'well-developed', 'well-performed',
    'enchanting', 'magical', 'bravo', 'enriching', 'enlightening', 'enjoy', 'pleasure',
    'pleasing', 'pleased', 'sublime', 'exquisite', 'glorious', 'outstanding', 'superior',
    'commendable', 'noteworthy', 'notable', 'admirable', 'commend', 'praise', 'applaud',
    'acclaimed', 'acclaim', 'favorite', 'best', 'top', 'winner', 'winning', 'award-winning',
    'blockbuster', 'hit', 'crowd-pleaser', 'must', 'mustwatch', 'mustsee', 'worthwhile',
    'worth', 'enjoyment', 'joy', 'cheerful', 'positive', 'optimistic', 'hopeful',
    'heartfelt', 'satisfy', 'satisfying', 'satisfied', 'impressed', 'impress', 'impressive',
    'commend', 'commendable', 'commendation', 'applause', 'applaud', 'applauded',
    'recommendation', 'recommended', 'recommend', 'favorite', 'favorites', 'fav', 'fave',
    'gem', 'hidden gem', 'classic', 'timeless', 'iconic', 'legendary', 'epic', 'must-own',
    'must-have', 'must-see', 'must-watch', 'must experience', 'must try', 'must buy',
    'must read', 'must listen', 'must play', 'must visit', 'must go', 'must do',
    'must attend', 'must eat', 'must drink', 'must taste', 'must feel', 'must love',
    'must enjoy', 'must appreciate', 'must admire', 'must respect', 'must cherish',
    'must treasure', 'must value', 'must honor', 'must celebrate', 'must embrace',
    'must support', 'must encourage', 'must inspire', 'must motivate', 'must uplift',
    'must empower', 'must enlighten', 'must educate', 'must inform', 'must entertain',
    'must amuse', 'must delight', 'must please', 'must satisfy', 'must gratify',
    'must fulfill', 'must enrich', 'must enhance', 'must improve', 'must better',
    'must advance', 'must progress', 'must develop', 'must grow', 'must evolve',
    'must transform', 'must change', 'must innovate', 'must create', 'must build',
    'must make', 'must produce', 'must generate', 'must invent', 'must discover',
    'must explore', 'must learn', 'must teach', 'must share', 'must give', 'must help',
    'must serve', 'must care', 'must love', 'must like', 'must prefer', 'must choose',
    'must select', 'must pick', 'must opt', 'must decide', 'must determine', 'must resolve',
    'must solve', 'must fix', 'must repair', 'must mend', 'must heal', 'must cure',
    'must treat', 'must prevent', 'must protect', 'must defend', 'must guard', 'must shield',
    'must save', 'must rescue', 'must recover', 'must restore', 'must revive', 'must renew',
    'must refresh', 'must rejuvenate', 'must revitalize', 'must energize', 'must invigorate',
    'must stimulate', 'must excite', 'must thrill', 'must exhilarate', 'must inspire',
    'must motivate', 'must encourage', 'must support', 'must help', 'must assist',
    'must aid', 'must benefit', 'must profit', 'must gain', 'must win', 'must succeed',
    'must achieve', 'must accomplish', 'must attain', 'must reach', 'must realize',
    'must fulfill', 'must complete', 'must finish', 'must end', 'must conclude',
    'must close', 'must wrap', 'must finalize', 'must settle', 'must resolve',
    'must solve', 'must fix', 'must repair', 'must mend', 'must heal', 'must cure',
    'must treat', 'must prevent', 'must protect', 'must defend', 'must guard', 'must shield',
]

NEGATIVE_WORDS = [
    'bad', 'worst', 'awful', 'terrible', 'horrible', 'atrocious', 'dreadful',
    'abysmal', 'lousy', 'poor', 'subpar', 'mediocre', 'unacceptable', 'disappointing',
    'boring', 'predictable', 'uninteresting', 'forgettable', 'tedious', 'slow',
    'annoying', 'unbearable', 'painful', 'mess', 'flawed', 'weak', 'unconvincing',
    'overrated', 'underwhelming', 'cliché', 'cliche', 'ridiculous', 'nonsense',
    'unrealistic', 'waste', 'pointless', 'dull', 'unimpressive', 'cringe', 'cringeworthy',
    'unoriginal', 'incoherent', 'confusing', 'unfunny', 'forced', 'flat', 'shallow',
    'forgettable', 'unnecessary', 'unpleasant', 'unwatchable', 'cheesy', 'corny',
    'painstaking', 'tedium', 'drag', 'lackluster', 'insipid', 'overlong', 'bloated',
    'contrived', 'derivative', 'sloppy', 'amateurish', 'unfocused', 'awkward',
    'unresolved', 'unfulfilled', 'unremarkable', 'unengaging', 'unappealing',
    'unbelievable', 'unbalanced', 'unpolished', 'unrefined', 'unprofessional',
    'unmemorable', 'unrelatable', 'unreal', 'unconvincing', 'unbearable', 'unimaginative',
    'unintelligent', 'unnecessary', 'unpleasant', 'unrewarding', 'unsettling',
    'unsubtle', 'unworthy', 'unwelcome', 'unwise', 'unwieldy', 'unworthy', 'unjustified',
    'unjust', 'unforgivable', 'unforgiving', 'unfortunate', 'unfriendly', 'unfulfilled',
    'unimpressed', 'uninspired', 'uninspiring', 'uninteresting', 'unlikable', 'unlucky',
    'unmotivated', 'unoriginal', 'unpleasant', 'unrealistic', 'unsatisfying', 'unsuccessful',
    'unsuitable', 'unsurprising', 'untalented', 'unwatchable', 'upsetting', 'useless',
    'vapid', 'weak', 'worthless', 'yawn', 'disaster', 'disastrous', 'disgusting',
    'distasteful', 'disturbing', 'dreary', 'embarrassing', 'excruciating', 'fail',
    'failure', 'flop', 'garbage', 'hackneyed', 'hated', 'hate', 'illogical', 'inferior',
    'irritating', 'lame', 'lacking', 'letdown', 'messy', 'monotonous', 'nonsensical',
    'offensive', 'painful', 'pathetic', 'poorly', 'regret', 'regrettable', 'repetitive',
    'shame', 'shameful', 'stupid', 'tiresome', 'trash', 'trite', 'unbearable', 'unconvincing',
    'unimpressive', 'uninteresting', 'unlikable', 'unoriginal', 'unpleasant', 'unwatchable',
    'waste', 'worthless'
]

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.positive_words = set(POSITIVE_WORDS)
        self.negative_words = set(NEGATIVE_WORDS)
        
    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        processed_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens
            if token not in self.punctuation and token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(processed_tokens)

def load_reviews(directory, label):
    reviews = []
    labels = []
    preprocessor = EnhancedSentimentAnalyzer()
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                if text:
                    processed_text = preprocessor.preprocess(text)
                    reviews.append(processed_text)
                    labels.append(label)
    return reviews, labels

def train_model(positive_dir, negative_dir):
    print("Loading data...")
    positive_reviews, positive_labels = load_reviews(positive_dir, 1)
    negative_reviews, negative_labels = load_reviews(negative_dir, 0)
    print(f"Loaded {len(positive_reviews)} positive and {len(negative_reviews)} negative reviews")

    all_reviews = positive_reviews + negative_reviews
    all_labels = positive_labels + negative_labels

    X_train, X_test, y_train, y_test = train_test_split(
        all_reviews, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print("Training model...")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.6, sublinear_tf=True, max_features=10000),
        LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, C=0.85, solver='liblinear')
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    print("\nEvaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return model

def analyze_sentiment(text, model):
    if not text.strip():
        return {"error": "Empty text input"}
    
    preprocessor = EnhancedSentimentAnalyzer()
    processed_text = preprocessor.preprocess(text)
    
    proba = model.predict_proba([processed_text])[0]
    prediction = model.predict([processed_text])[0]
    
    tokens = processed_text.split()
    pos_words = [w for w in tokens if w in POSITIVE_WORDS]
    neg_words = [w for w in tokens if w in NEGATIVE_WORDS]
    
    return {
        "prediction": "Positive" if prediction == 1 else "Negative",
        "confidence": float(max(proba)),
        "positive_words": list(set(pos_words)),
        "negative_words": list(set(neg_words))
    }

def analyze_directory(directory, model):
    results = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'confidence_sum': 0,
        'positive_words': defaultdict(int),
        'negative_words': defaultdict(int)
    }
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                if text:
                    analysis = analyze_sentiment(text, model)
                    results['total'] += 1
                    if analysis['prediction'] == 'Positive':
                        results['positive'] += 1
                    else:
                        results['negative'] += 1
                    results['confidence_sum'] += analysis['confidence']
                    for word in analysis['positive_words']:
                        results['positive_words'][word] += 1
                    for word in analysis['negative_words']:
                        results['negative_words'][word] += 1
    
    results['average_confidence'] = results['confidence_sum'] / results['total'] if results['total'] > 0 else 0
    results['top_positive_words'] = sorted(results['positive_words'].items(), key=lambda x: x[1], reverse=True)[:5]
    results['top_negative_words'] = sorted(results['negative_words'].items(), key=lambda x: x[1], reverse=True)[:5]
    
    return results

def print_directory_analysis(results, dir_name):
    print(f"\nAnalysis for {dir_name}:")
    print(f"Total reviews: {results['total']}")
    print(f"Positive: {results['positive']} ({results['positive']/results['total']*100:.1f}%)")
    print(f"Negative: {results['negative']} ({results['negative']/results['total']*100:.1f}%)")
    print(f"Average confidence: {results['average_confidence']:.2f}")
    
    print("\nTop 5 positive words:")
    for word, count in results['top_positive_words']:
        print(f"  {word}: {count}")
    
    print("\nTop 5 negative words:")
    for word, count in results['top_negative_words']:
        print(f"  {word}: {count}")

    # Matplotlib bar plot for sentiment distribution
    plt.figure(figsize=(6, 4))
    labels = ['Positive', 'Negative']
    counts = [results['positive'], results['negative']]
    colors = ['#4CAF50', '#F44336']
    plt.bar(labels, counts, color=colors)
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.title('Sentiment Distribution')
    plt.show()

def interactive_mode(model):
    print("\nInteractive Mode (type 'quit' to exit):")
    while True:
        text = input("\nEnter review: ").strip()
        if text.lower() == 'quit':
            break
        if not text:
            print("Please enter text")
            continue
        analysis = analyze_sentiment(text, model)
        print(f"\nPrediction: {analysis['prediction']} (Confidence: {analysis['confidence']:.2f})")
        if analysis['positive_words']:
            print(f"Positive words: {', '.join(analysis['positive_words'])}")
        if analysis['negative_words']:
            print(f"Negative words: {', '.join(analysis['negative_words'])}")

def main():
    print("Movie Review Sentiment Analyzer\n")
    
    pos_dir = input(f"Positive reviews directory [default: {DEFAULT_POSITIVE_DIR}]: ").strip() or DEFAULT_POSITIVE_DIR
    neg_dir = input(f"Negative reviews directory [default: {DEFAULT_NEGATIVE_DIR}]: ").strip() or DEFAULT_NEGATIVE_DIR
    
    model = train_model(pos_dir, neg_dir)
    
    while True:
        print("\n1. Analyze directory\n2. Test review\n3. Exit")
        choice = input("Choose (1-3): ").strip()
        
        if choice == '1':
            dir_path = input("Directory path: ").strip()
            if not os.path.isdir(dir_path):
                print("Invalid directory")
                continue
            start_time = time.time()
            results = analyze_directory(dir_path, model)
            print_directory_analysis(results, dir_path)
            print(f"Analysis done in {time.time() - start_time:.2f} seconds")
        
        elif choice == '2':
            interactive_mode(model)
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Choose 1, 2, or 3")

if __name__ == "__main__":
    main()
    