import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, target_col):
        self.output_dir = '../output/final_features/'
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.tfidf_vectorizers = {}
        self.num_cols = None
        self.drop_cols = [
            "pr_number", "title", "created_at", "closed_at", "directories", "file_types", "language_types", "author",
            "mergeable", "mergeable_state", "rebaseable", "milestone" , # 一堆空值和垃圾信息
            "updated_at", "first_review_at", "ahead_by", "behind_by"
        ]

    def delete_useless_columns(self, df : pd.DataFrame):
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        # 删除目标列缺失的行
        df = df.dropna(subset=[self.target_col])
        return df

    def encode_columns(self, df: pd.DataFrame):
        # bool
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            df[col] = df[col].astype(int)

        # 时间
        time_cols = ['updated_at', 'first_review_at']
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df = df.dropna(subset=[c for c in time_cols if c in df.columns]).copy()

        if 'created_at' in df.columns and 'closed_at' in df.columns:
            df['duration_days'] = (df['closed_at'] - df['created_at']).dt.days

        for col in time_cols:
            if col in df.columns:
                df[col + '_weekday'] = df[col].dt.weekday.astype(int)
        df = df.drop(columns=[c for c in time_cols if c in df.columns])

        # 文本特征
        text_cols = ['title']
        tfidf_features = []
        for col in text_cols:
            if col in df.columns:
                tfidf = TfidfVectorizer(max_features=50)
                tfidf_matrix = tfidf.fit_transform(df[col].fillna(''))
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
                df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
                df.drop(columns=[col], inplace=True)
                tfidf_features.extend(tfidf_df.columns.tolist())
                self.tfidf_vectorizers[col] = tfidf

        # 暂时先不用author,这个只在分类的时候用
        categorical_cols = ['state']
        # 分类列 0/1 映射
        for col in categorical_cols:
            if col in df.columns and col != self.target_col:
                df[col] = df[col].map({'open': 1, 'closed': 0}).fillna(0).astype(int)

        return df

    def preprocess_features(self, df, filename):
        df = self.delete_useless_columns(df)
        df = self.encode_columns(df)

        y = df[self.target_col].values
        X = df.drop(columns=[self.target_col], errors="ignore")

        X = X.fillna(0)

        non_numeric_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        if non_numeric_cols:
            print("非数值列:", non_numeric_cols)

        self.num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        X_num = X[self.num_cols]

        X_scaled = self.scaler.fit_transform(X_num)

        final_features_df = pd.DataFrame(X_scaled, columns=self.num_cols)
        final_features_df[self.target_col] = y
        save_path = os.path.join(self.output_dir, f"{filename}_final_features.csv")
        final_features_df.to_csv(save_path, index=False)
        print(f"feature saved in: {save_path}")

        return X_scaled, y, self.num_cols
