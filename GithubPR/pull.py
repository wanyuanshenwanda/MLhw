import requests
import pandas as pd
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# 定义特征描述
FEATURE_DESCRIPTIONS = {
    "pr_number": "PR编号",
    "title": "PR标题",
    "state": "PR状态(open/closed)",
    "merged": "是否已合并",
    "created_at": "创建时间",
    "closed_at": "关闭时间",
    "updated_at": "最后更新时间",
    "author": "作者用户名",
    "author_is_collaborator": "作者是否是协作者",
    "additions": "添加的代码行数",
    "deletions": "删除的代码行数",
    "changed_files": "修改的文件数量",
    "commits": "提交数量",
    "comments": "评论数量",
    "review_comments": "代码审查评论数量",
    "file_types_count": "修改的文件类型数量",
    "directories_count": "修改的目录数量",
    "total_changes": "总改动行数(添加+删除)",
    "first_review_at": "首次代码审查时间",
    "time_to_close_days": "从创建到关闭的天数",
    "response_time_days": "从创建到首次审查的天数",
    "review_count": "代码审查次数",
    "approval_count": "通过审查的次数",
    "change_request_count": "请求更改的次数",
    "reviewer_count": "参与审查的人数",
    "test_files_count": "测试文件数量",
    "test_changes": "测试文件改动行数",
    "doc_files_count": "文档文件数量",
    "config_files_count": "配置文件数量",
    "title_length": "标题长度",
    "description_length": "描述长度",
    "has_description": "是否有描述",
    "label_count": "标签数量",
    "milestone": "里程碑名称",
    "is_fork": "是否来自fork仓库",
    "ahead_by": "领先提交数",
    "behind_by": "落后提交数",
    "mergeable": "是否可以合并",
    "mergeable_state": "合并状态",
    "rebaseable": "是否可以变基",
    "assignees_count": "指派人数量",
    "has_test": "是否包含测试文件",
    "has_bug": "是否包含bug修复",
    "has_feature": "是否包含新功能",
    "has_improve": "是否包含改进",
    "has_document": "是否包含文档更新",
    "has_refactor": "是否包含重构",
    "directories": "修改的目录数量",
    "language_types": "编程语言类型数量",
    "file_types": "文件类型数量",
    "lines_added": "添加的代码行数",
    "lines_deleted": "删除的代码行数",
    "segs_added": "添加的代码段数量",
    "segs_deleted": "删除的代码段数量",
    "segs_changed": "更改的代码段数量",
    "files_added": "添加的文件数量",
    "files_deleted": "删除的文件数量",
    "files_changed": "更改的文件数量",
    "change_num": "变更数量",
    "files_modified": "修改的文件数量",
    "is_core_member": "是否是核心成员",
    "prev_PRs": "作者之前的PR数量",
    "title_words": "标题单词数量",
    "body_words": "正文单词数量"
}


def print_feature_table():
    """打印所有特征及其描述的表格"""
    print("特征列表:")
    print("-" * 80)
    print(f"{'序号':<4} {'特征名称':<25} {'描述':<50}")
    print("-" * 80)

    features = list(FEATURE_DESCRIPTIONS.keys())
    for i, feature in enumerate(features, 1):
        print(f"{i:<4} {feature:<25} {FEATURE_DESCRIPTIONS.get(feature, '无描述'):<50}")

    print("-" * 80)
    print(f"总共 {len(features)} 个特征")


def get_paginated_data(url, params=None):
    """处理分页请求，返回所有页面的结果列表"""
    all_data = []
    page = 1
    per_page = 100  # 每页最多100条
    if params is None:
        params = {}
    params['per_page'] = per_page

    while True:
        params['page'] = page
        response = requests.get(url, headers=HEADERS, params=params)

        # 简单的错误处理
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}, 信息: {response.text}")
            break

        page_data = response.json()
        if not page_data:
            break

        all_data.extend(page_data)

        if len(page_data) < per_page:
            break

        page += 1
        time.sleep(0.5)  # 添加短暂延迟，避免过快请求

    return all_data


def get_repo_pulls(owner, repo):
    """获取指定仓库的所有PR（包括open和closed）"""
    print(f"正在获取 {owner}/{repo} 的PR列表...")
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {'state': 'all'}
    pulls = get_paginated_data(url, params)
    print(f"从 {owner}/{repo} 获取到 {len(pulls)} 个PR")
    return pulls


def get_pr_details(owner, repo, pr_number):
    """获取单个PR的详细信息"""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取PR #{pr_number} 详情失败: {response.status_code}")
        return None


def get_pr_files(owner, repo, pr_number):
    """获取单个PR修改的文件列表"""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取PR #{pr_number} 文件列表失败: {response.status_code}")
        return []


def get_pr_reviews(owner, repo, pr_number):
    """获取单个PR的评审信息"""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取PR #{pr_number} 评审信息失败: {response.status_code}")
        return []


def get_contributors(owner, repo):
    """获取仓库的贡献者列表，用于判断用户身份"""
    print(f"正在获取 {owner}/{repo} 的贡献者信息...")
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    contributors = get_paginated_data(url)
    contributor_logins = [contributor['login'] for contributor in contributors]
    print(f"从 {owner}/{repo} 获取到 {len(contributor_logins)} 名贡献者")
    return contributor_logins


def get_author_previous_prs(owner, repo, author):
    """获取作者在此仓库中之前的PR数量"""
    url = f"https://api.github.com/search/issues"
    params = {
        'q': f'repo:{owner}/{repo} author:{author} type:pr',
        'per_page': 1
    }

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('total_count', 0)
    else:
        print(f"获取作者 {author} 的PR历史失败: {response.status_code}")
        return 0


def count_words(text):
    """计算文本中的单词数量"""
    if not text:
        return 0
    return len(re.findall(r'\w+', text))


def extract_features_from_pr(pr_data, pr_files, pr_reviews, contributors_list, owner, repo):
    """从原始API数据中提取和计算我们关心的特征"""
    features = {}

    # 基础信息
    features['pr_number'] = pr_data['number']
    features['title'] = pr_data['title']
    features['state'] = pr_data['state']  # open, closed
    features['merged'] = pr_data.get('merged', False)  # 是否已合并
    features['created_at'] = pr_data['created_at']
    features['closed_at'] = pr_data['closed_at']
    features['updated_at'] = pr_data['updated_at']

    # 用户信息
    user = pr_data['user']
    features['author'] = user['login']
    features['author_is_collaborator'] = features['author'] in contributors_list  # 粗略判断是否为协作者

    # 获取作者之前的PR数量
    features['prev_PRs'] = get_author_previous_prs(owner, repo, features['author'])

    # PR基本指标
    features['additions'] = pr_data['additions']  # 添加行数
    features['deletions'] = pr_data['deletions']  # 删除行数
    features['changed_files'] = pr_data['changed_files']  # 修改文件数
    features['commits'] = pr_data['commits']  # 提交数
    features['comments'] = pr_data['comments']  # 评论数（Issue评论？）
    features['review_comments'] = pr_data['review_comments']  # 评审评论数

    # 文本分析特征
    features['title_length'] = len(pr_data['title'])
    features['description_length'] = len(pr_data['body']) if pr_data['body'] else 0
    features['has_description'] = 1 if pr_data['body'] and len(pr_data['body']) > 0 else 0
    features['title_words'] = count_words(pr_data['title'])
    features['body_words'] = count_words(pr_data['body'] if pr_data['body'] else "")

    # 检查关键词
    title_body = (pr_data['title'] + " " + (pr_data['body'] if pr_data['body'] else "")).lower()
    features['has_test'] = 1 if any(
        keyword in title_body for keyword in ['test', 'testing', 'unit test', 'integration test']) else 0
    features['has_bug'] = 1 if any(keyword in title_body for keyword in ['bug', 'fix', 'issue', 'error']) else 0
    features['has_feature'] = 1 if any(
        keyword in title_body for keyword in ['feature', 'new feature', 'implement']) else 0
    features['has_improve'] = 1 if any(keyword in title_body for keyword in ['improve', 'enhance', 'optimize']) else 0
    features['has_document'] = 1 if any(keyword in title_body for keyword in ['doc', 'document', 'readme']) else 0
    features['has_refactor'] = 1 if any(keyword in title_body for keyword in ['refactor', 'restructure']) else 0

    # 指派人信息
    features['assignees_count'] = len(pr_data['assignees'])

    # 从files中计算更多关于改动的特征
    file_extensions = set()
    programming_languages = set()
    directories_touched = set()
    total_changes = 0
    test_files = []
    doc_files = []
    config_files = []
    added_files = 0
    deleted_files = 0
    modified_files = 0

    # 编程语言映射
    lang_ext_map = {
        'py': 'Python', 'js': 'JavaScript', 'java': 'Java', 'cpp': 'C++', 'c': 'C',
        'php': 'PHP', 'rb': 'Ruby', 'go': 'Go', 'rs': 'Rust', 'ts': 'TypeScript',
        'html': 'HTML', 'css': 'CSS', 'scss': 'SCSS', 'sql': 'SQL', 'swift': 'Swift',
        'kt': 'Kotlin', 'sh': 'Shell', 'r': 'R', 'm': 'Objective-C', 'pl': 'Perl'
    }

    for file in pr_files:
        filename = file['filename']
        status = file.get('status', 'modified')

        # 统计文件状态
        if status == 'added':
            added_files += 1
        elif status == 'removed':
            deleted_files += 1
        else:  # modified, renamed, etc.
            modified_files += 1

        # 文件扩展名和语言
        ext = filename.split('.')[-1].lower() if '.' in filename else 'none'
        file_extensions.add(ext)

        if ext in lang_ext_map:
            programming_languages.add(lang_ext_map[ext])
        elif ext in ['md', 'txt', 'rst']:
            programming_languages.add('Documentation')
        elif ext in ['yml', 'yaml', 'json', 'xml', 'config', 'ini']:
            programming_languages.add('Configuration')

        # 目录信息
        directories_touched.add(filename.rsplit('/', 1)[0] if '/' in filename else '/')
        total_changes += file['additions'] + file['deletions']

        # 分类文件类型
        if 'test' in filename.lower() or 'spec' in filename.lower():
            test_files.append(file)
        if any(ext in filename.lower() for ext in ['.md', '.rst', '.txt', '.doc', '.docx']):
            doc_files.append(file)
        if any(ext in filename.lower() for ext in ['.yml', '.yaml', '.json', '.xml', '.config', '.ini']):
            config_files.append(file)

    features['file_types_count'] = len(file_extensions)
    features['file_types'] = len(file_extensions)
    features['language_types'] = len(programming_languages)
    features['directories_count'] = len(directories_touched)
    features['directories'] = len(directories_touched)
    features['total_changes'] = total_changes
    features['test_files_count'] = len(test_files)
    features['test_changes'] = sum(f['additions'] + f['deletions'] for f in test_files)
    features['doc_files_count'] = len(doc_files)
    features['config_files_count'] = len(config_files)

    # 文件变更统计
    features['files_added'] = added_files
    features['files_deleted'] = deleted_files
    features['files_modified'] = modified_files
    features['files_changed'] = added_files + deleted_files + modified_files
    features['change_num'] = features['commits']  # 使用提交数作为变更数量

    # 代码段统计（简化处理，使用文件数作为代理）
    features['segs_added'] = added_files
    features['segs_deleted'] = deleted_files
    features['segs_changed'] = modified_files

    # 核心成员判断
    features['is_core_member'] = 1 if features['author'] in contributors_list[:10] else 0  # 前10名贡献者视为核心成员

    # 行数统计
    features['lines_added'] = features['additions']
    features['lines_deleted'] = features['deletions']

    # 从reviews中获取评审信息
    if pr_reviews:
        first_review = min(pr_reviews, key=lambda x: x['submitted_at'])
        features['first_review_at'] = first_review['submitted_at']
        features['review_count'] = len(pr_reviews)
        features['approval_count'] = sum(1 for review in pr_reviews if review['state'] == 'APPROVED')
        features['change_request_count'] = sum(1 for review in pr_reviews if review['state'] == 'CHANGES_REQUESTED')
        # features['reviewer_count'] = len(set(review['user']['login'] for review in pr_reviews))
        features['reviewer_count'] = len(set(
            review['user']['login'] for review in pr_reviews
            if review['user'] is not None and 'login' in review['user']
        ))
    else:
        features['first_review_at'] = None
        features['review_count'] = 0
        features['approval_count'] = 0
        features['change_request_count'] = 0
        features['reviewer_count'] = 0

    # 时间相关特征
    from dateutil.parser import parse
    created = parse(pr_data['created_at'])
    if pr_data['closed_at']:
        closed = parse(pr_data['closed_at'])
        features['time_to_close_days'] = (closed - created).total_seconds() / 86400

    if features.get('first_review_at'):
        first_review = parse(features['first_review_at'])
        features['response_time_days'] = (first_review - created).total_seconds() / 86400

    # 分支和合并信息
    features['is_fork'] = pr_data['head']['repo']['fork'] if pr_data['head']['repo'] else False
    features['ahead_by'] = pr_data.get('ahead_by', 0)
    features['behind_by'] = pr_data.get('behind_by', 0)
    features['mergeable'] = pr_data.get('mergeable', None)
    features['mergeable_state'] = pr_data.get('mergeable_state', None)
    features['rebaseable'] = pr_data.get('rebaseable', None)

    # 标签和里程碑
    features['label_count'] = len(pr_data['labels'])
    features['milestone'] = pr_data['milestone']['title'] if pr_data['milestone'] else None

    return features


def collect_pr_data_for_repo(owner, repo, max_prs=None):
    """主函数：收集一个仓库的所有PR数据"""

    # 获取贡献者列表
    contributors = get_contributors(owner, repo)

    # 获取所有PR列表
    all_pulls = get_repo_pulls(owner, repo)
    if max_prs:
        all_pulls = all_pulls[:max_prs]  # 用于测试，限制PR数量

    all_pr_features = []

    print(f"开始处理 {len(all_pulls)} 个PR的详细信息...")
    for idx, pull in enumerate(all_pulls):
        pr_number = pull['number']
        print(f"正在处理 PR #{pr_number} ({idx + 1}/{len(all_pulls)})")

        # 获取详细信息、文件、评论
        pr_details = get_pr_details(owner, repo, pr_number)
        if pr_details is None:
            continue

        pr_files = get_pr_files(owner, repo, pr_number)
        pr_reviews = get_pr_reviews(owner, repo, pr_number)

        # 提取特征
        pr_features = extract_features_from_pr(pr_details, pr_files, pr_reviews, contributors, owner, repo)
        all_pr_features.append(pr_features)

        time.sleep(0.5)  # 添加延迟，避免触发API速率限制

    return all_pr_features


def save_to_csv(pr_features_list, filename):
    """将PR特征列表保存为CSV文件"""
    df = pd.DataFrame(pr_features_list)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"数据已保存至 {filename}")
    return df


def split_train_test_by_time(df, created_at_col='created_at', test_size=0.2):
    """按PR创建时间划分训练集和测试集，确保测试集时间晚于训练集"""
    df_sorted = df.sort_values(by=created_at_col).reset_index(drop=True)
    split_index = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    print(f"训练集时间范围: {train_df[created_at_col].min()} 至 {train_df[created_at_col].max()}")
    print(f"测试集时间范围: {test_df[created_at_col].min()} 至 {test_df[created_at_col].max()}")

    return train_df, test_df


# 示例用法
if __name__ == "__main__":
    # 首先打印特征表格
    print_feature_table()

    owner = "PaddlePaddle"
    repo = "PaddleOCR"
    max_prs_to_fetch = 600

    pr_features = collect_pr_data_for_repo(owner, repo, max_prs=max_prs_to_fetch)

    if pr_features:
        df = save_to_csv(pr_features, f"data/paddle/github_pr_data_{owner}_{repo}.csv")

        # 按时间切分数据
        train_df, test_df = split_train_test_by_time(df)
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")

        # 保存切分后的数据
        train_df.to_csv(f"data/paddle/github_pr_train_{owner}_{repo}.csv", index=False)
        test_df.to_csv(f"data/paddle/github_pr_test_{owner}_{repo}.csv", index=False)
    else:
        print("未获取到PR数据。")