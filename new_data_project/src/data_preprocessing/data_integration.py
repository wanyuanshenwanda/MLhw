def load_all_data(data_path):
    dataframes = {}
    files = {
        'pr_info': 'PR_info_add_conversation.xlsx',
        'pr_features': 'PR_features.xlsx',
        'author_features': 'author_features.xlsx',
        'project_features': 'project_features.xlsx',
        'pr_commit_info': 'PR_commit_info.xlsx',
        'pr_comment_info': 'PR_comment_info.xlsx',
        'reviewer_features': 'reviewer_features.xlsx'
    }

    for name, filename in files.items():
        file_path = data_path + filename
        dataframes[name] = pd.read_excel(file_path)
        print(f"load: {filename} - shape: {dataframes[name].shape}")

    return dataframes


def extract_basic_features(pr_info_df):
    df = pr_info_df.copy()

    basic_features = [
        'number', 'state', 'title', 'author', 'body',
        'created_at', 'updated_at', 'merged_at', 'merged', 'comments',
        'review_comments', 'commits', 'additions', 'deletions', 'changed_files',
        'conversation', 'closed_at'
    ]

    df_basic = df[basic_features].copy()

    df_basic['title_length'] = df_basic['title'].fillna('').str.len()
    df_basic['body_length'] = df_basic['body'].fillna('').str.len()

    df_basic['created_at'] = pd.to_datetime(df_basic['created_at'])
    df_basic['closed_at'] = pd.to_datetime(df_basic['closed_at'])

    return df_basic

def merge_pr_technical_features(main_df, pr_features_df):
    main_prs = set(main_df['number'])
    feature_prs = set(pr_features_df['number'])
    common_prs = main_prs.intersection(feature_prs)

    tech_feature_columns = [
        'number',
        'has_test', 'has_bug', 'has_feature', 'has_document', 'has_improve', 'has_refactor',
        'files_updated','files_deleted'
        'is_reviewed', 'comment_num', 'comment_length'
    ]

    available_columns = [col for col in tech_feature_columns if col in pr_features_df.columns]

    df_merged = main_df.merge(
        pr_features_df[available_columns],
        on='number',
        how='left',
        suffixes=('', '_tech')
    )

    return df_merged

def merge_author_features(main_df, author_features_df):
    main_numbers = set(main_df['number'].dropna())
    feature_numbers = set(author_features_df['number'].dropna())
    common_numbers = main_numbers.intersection(feature_numbers)

    author_feature_columns = [
        'number',
        'experience', 'is_reviewer', 'change_num', 'participation',
        'changes_per_week', 'merge_proportion',
        'degree_centrality', 'closeness_centrality'
    ]

    available_columns = [col for col in author_feature_columns if col in author_features_df.columns]

    df_merged = main_df.merge(
        author_features_df[available_columns],
        on='number',
        how='left',
        suffixes=('', '_author')
    )

    return df_merged


def merge_project_features(main_df, project_features_df):
    project_feature_columns = [
        'number',
        'project_age', 'language_num', 'change_num', 'author_num',
        'reviewer_num', 'team_size', 'changes_per_week', 'merge_proportion'
    ]

    available_columns = [col for col in project_feature_columns if col in project_features_df.columns]

    df_merged = main_df.merge(
        project_features_df[available_columns],
        on='number',
        how='left',
        suffixes=('', '_project')
    )

    return df_merged



def aggregate_commit_features(main_df, commit_info_df):
    commit_agg = commit_info_df.groupby('belong_to_PR').agg({
        'changes': 'sum',
        'additions': 'sum',
        'deletions': 'sum',
        'author': 'nunique',
        'file_name_list': 'count'
    }).reset_index()

    commit_agg = commit_agg.rename(columns={
        'belong_to_PR': 'number',
        'changes': 'total_changes',
        'additions': 'total_additions_commit',
        'deletions': 'total_deletions_commit',
        'author': 'commit_authors_count',
        'file_name_list': 'commit_count'
    })

    df_merged = main_df.merge(
        commit_agg,
        on='number',
        how='left'
    )
    return df_merged


def aggregate_comment_features(main_df, comment_info_df):
    comment_agg = comment_info_df.groupby('belong_to_PR').agg({
        'reviewer': 'nunique',
        'body': 'count',
        'created_at': 'min'
    }).reset_index()

    comment_agg = comment_agg.rename(columns={
        'belong_to_PR': 'number',
        'reviewer': 'unique_reviewers',
        'body': 'total_comments',
        'created_at': 'first_comment_time'
    })

    df_merged = main_df.merge(
        comment_agg,
        on='number',
        how='left'
    )

    return df_merged

def finalize_features(df):
    print(f"\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_features = missing_stats[missing_stats > 0]
    if len(missing_features) > 0:
        for feature, missing_count in missing_features.items():
            missing_pct = missing_count / len(df) * 100
            print(f"  {feature}: {missing_count} ({missing_pct:.1f}%)")
    else:
        print("  无缺失值")

    return df



import pandas as pd

def data_pre(data_path: str):
    dataframes = load_all_data(data_path)
    # 提取基础特征
    df_main = extract_basic_features(dataframes['pr_info'])
    print(f"基础特征列({data_path}):", df_main.columns.tolist())

    df_with_tech = merge_pr_technical_features(df_main, dataframes['pr_features'])
    # df_with_author = merge_author_features(df_with_tech, dataframes['author_features'])
    # df_with_project = merge_project_features(df_with_tech, dataframes['project_features'])
    df_with_commit = aggregate_commit_features(df_with_tech, dataframes['pr_commit_info'])
    df_final = aggregate_comment_features(df_with_commit, dataframes['pr_comment_info'])

    df_final = finalize_features(df_final)

    return df_final


if __name__ == '__main__':
    files = ['yii2', 'pytorch', 'vscode']
    all_dfs = []

    for file in files:
        data_path = f'../../datas/{file}/'
        output_path = f'../../datas/{file}/final_features.csv'

        df = data_pre(data_path)
        df["project"] = file
        df.to_csv(output_path, index=False)
        print(f"{file} table saved in: {output_path}")

        all_dfs.append(df)

    # 合并
    df_total = pd.concat(all_dfs, ignore_index=True)

    total_output_path = '../../datas/all_projects_final_features.csv'
    df_total.to_csv(total_output_path, index=False)
    print(f"\nfinal table saved in: {total_output_path}")
    print("row:", df_total.shape[0])
    print("col:", df_total.shape[1])


