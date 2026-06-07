import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data():
    print("Mulai proses otomatisasi data preprocessing...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.abspath(os.path.join(base_dir, '..', 'breast_cancer_raw.csv'))
    output_folder = os.path.abspath(os.path.join(base_dir, 'breast_cancer_preprocessing'))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    try:
        df = pd.read_csv(input_file)
        print(f"Data berhasil dimuat. Dimensi awal: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan.")
        return

    columns_to_drop = ['Unnamed: 32', 'id']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    print("Kolom tidak berguna telah dihapus.")
    
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    print("Encoding label selesai.")
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['diagnosis'] = y_train.reset_index(drop=True)
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['diagnosis'] = y_test.reset_index(drop=True)
    print("Scaling data selesai.")
    
    scaler_path = os.path.join(output_folder, 'scaler.joblib')
    label_encoder_path = os.path.join(output_folder, 'label_encoder.joblib')
    train_output_path = os.path.join(output_folder, 'train_data_clean.csv')
    test_output_path = os.path.join(output_folder, 'test_data_clean.csv')
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, label_encoder_path)
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"\nSukses! Data bersih, scaler, & label encoder tersimpan di:\n - {train_output_path}\n - {test_output_path}\n - {scaler_path}\n - {label_encoder_path}")

if __name__ == "__main__":
    preprocess_data()