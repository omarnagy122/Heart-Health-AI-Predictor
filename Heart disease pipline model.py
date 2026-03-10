import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from category_encoders import BinaryEncoder


class HeartDiseasePipeline:
    def __init__(self, data_path='HeartDiseaseTrain-Test.csv'):
        self.data_path = data_path
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=3)
        }
        self.best_model = None
        self.best_model_name = ""
        self.preprocessors = {}

    def load_and_split(self):
        df = pd.read_csv(self.data_path)
        Y = df['target']
        X = df.drop(columns=['target'])

        # Split: 70% Train, 15% Val, 15% Test
        x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=24)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=24)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def preprocess(self, x_train, x_val, x_test, y_train):
        # 1. Binary Encoding
        bi_cols = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']
        bi_enc = BinaryEncoder()
        x_train_bi = bi_enc.fit_transform(x_train[bi_cols])
        x_val_bi = bi_enc.transform(x_val[bi_cols])
        x_test_bi = bi_enc.transform(x_test[bi_cols])

        x_train = pd.concat([x_train.drop(columns=bi_cols), x_train_bi], axis=1)
        x_val = pd.concat([x_val.drop(columns=bi_cols), x_val_bi], axis=1)
        x_test = pd.concat([x_test.drop(columns=bi_cols), x_test_bi], axis=1)

        # 2. Label Encoding
        le_cols = ['slope', 'vessels_colored_by_flourosopy']
        le_dict = {}
        for col in le_cols:
            le = LabelEncoder()
            x_train[col] = le.fit_transform(x_train[col])
            x_val[col] = le.transform(x_val[col])
            x_test[col] = le.transform(x_test[col])
            le_dict[col] = le

        # 3. One-Hot Encoding
        oh_cols = ['chest_pain_type', 'rest_ecg', 'thalassemia']
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        x_train_oh = ohe.fit_transform(x_train[oh_cols])
        oh_features = ohe.get_feature_names_out(oh_cols)
        x_train_oh = pd.DataFrame(x_train_oh, columns=oh_features, index=x_train.index)
        x_train = pd.concat([x_train.drop(columns=oh_cols), x_train_oh], axis=1)

        x_val_oh = ohe.transform(x_val[oh_cols])
        x_val_oh = pd.DataFrame(x_val_oh, columns=oh_features, index=x_val.index)
        x_val = pd.concat([x_val.drop(columns=oh_cols), x_val_oh], axis=1)

        x_test_oh = ohe.transform(x_test[oh_cols])
        x_test_oh = pd.DataFrame(x_test_oh, columns=oh_features, index=x_test.index)
        x_test = pd.concat([x_test.drop(columns=oh_cols), x_test_oh], axis=1)

        # 4. Scaling
        num_cols = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
        scaler = StandardScaler()
        x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
        x_val[num_cols] = scaler.transform(x_val[num_cols])
        x_test[num_cols] = scaler.transform(x_test[num_cols])

        # 5. Feature Selection
        temp_df = pd.concat([x_train, y_train], axis=1)
        target_corr = temp_df.corr()['target']
        low_corr_features = target_corr[abs(target_corr) < 0.05].index.tolist()
        if 'target' in low_corr_features: low_corr_features.remove('target')

        x_train.drop(columns=low_corr_features, inplace=True, errors='ignore')
        x_val.drop(columns=low_corr_features, inplace=True, errors='ignore')
        x_test.drop(columns=low_corr_features, inplace=True, errors='ignore')

        # 6. PCA (10 components)
        pca = PCA(n_components=10)
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = pca.transform(x_val)
        x_test_pca = pca.transform(x_test)

        self.preprocessors = {
            'binary': bi_enc, 'labels': le_dict, 'onehot': ohe,
            'scaler': scaler, 'pca': pca, 'final_columns': x_train.columns.tolist()
        }

        return x_train_pca, x_val_pca, x_test_pca

    def train_and_evaluate(self, x_train, x_val, x_test, y_train, y_val, y_test):
        best_score = 0  # ممكن نغيره لـ AUC أو Precision حسب الأولوية

        for name, model in self.models.items():
            model.fit(x_train, y_train)

            # حساب التوقعات (Predictions)
            y_pred = model.predict(x_val)
            # حساب الاحتمالات (Probabilities) عشان الـ AUC
            y_prob = model.predict_proba(x_val)[:, 1]

            # حساب المقاييس المطلوبة
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)

            print(f"--- {name} ---")
            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")  # دي بتهتم بدقة الموديل لما يقول إن الشخص مريض
            print(f"ROC-AUC  : {auc:.4f}")  # دي بتهتم بقدرة الموديل على التمييز بين المريض والسليم

            # اختيار الموديل الأفضل بناءً على الـ AUC مثلاً
            if auc > best_score:
                best_score = auc
                self.best_model = model
                self.best_model_name = name

        # حفظ الموديل الأفضل والـ Preprocessors
        artifact = {'model': self.best_model, 'preprocessors': self.preprocessors}
        joblib.dump(artifact, 'best_heart_disease_model.joblib')

    def run(self):
        x_train, x_val, x_test, y_train, y_val, y_test = self.load_and_split()
        x_train_p, x_val_p, x_test_p = self.preprocess(x_train, x_val, x_test, y_train)
        self.train_and_evaluate(x_train_p, x_val_p, x_test_p, y_train, y_val, y_test)


if __name__ == "__main__":
    pipeline = HeartDiseasePipeline()
    pipeline.run()