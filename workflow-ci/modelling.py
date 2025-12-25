import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_estimators', type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*50)
    print("🤖 HEART DISEASE CLASSIFICATION TRAINING")
    print("="*50)
    print(f"Parameters: test_size={args.test_size}, "
          f"n_estimators={args.n_estimators}")
    
    # Load data
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    main()