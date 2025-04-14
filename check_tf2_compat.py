import os
import re
import glob

def check_file_for_issues(file_path):
    """检查文件中可能的TF 2.x兼容性问题"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            print(f"Warning: Couldn't read {file_path} due to encoding issues")
            return []
    
    issues = []
    
    # 检查已知的TF 1.x API
    deprecated_apis = {
        'tf.contrib': 'tf.contrib命名空间在TF 2.x中已移除',
        'tf.placeholder': '在TF 2.x中，建议使用tf.keras输入层或函数参数而不是占位符',
        'tf.Session': '会话API在TF 2.x中已被Eager模式替代',
        'tf.train.exponential_decay': '在TF 2.x中，建议使用学习率调度器',
        'tf.global_variables_initializer': '在TF 2.x中，初始化方式不同'
    }
    
    for api, message in deprecated_apis.items():
        if api in content:
            issues.append(f"Found {api}: {message}")
    
    return issues

def scan_all_files():
    """扫描所有Python文件查找潜在问题"""
    python_files = glob.glob('**/*.py', recursive=True)
    all_issues = {}
    
    for file_path in python_files:
        issues = check_file_for_issues(file_path)
        if issues:
            all_issues[file_path] = issues
    
    return all_issues

if __name__ == "__main__":
    print("Scanning for TensorFlow 2.x compatibility issues...")
    issues = scan_all_files()
    
    if not issues:
        print("No compatibility issues found!")
    else:
        print(f"Found issues in {len(issues)} files:")
        for file, file_issues in issues.items():
            print(f"\n{file}:")
            for issue in file_issues:
                print(f"  - {issue}")
