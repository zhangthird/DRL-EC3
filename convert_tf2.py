import os
import re
import glob

def fix_tf_imports(file_path):
    """更新导入语句并添加兼容性代码"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 添加TF 2.x兼容性代码
    if 'import tensorflow as tf' in content and '# 确保兼容性' not in content:
        compat_code = """
# 确保兼容性
if hasattr(tf, 'function'):
    # TF 2.x
    tf1 = tf.compat.v1
    tf1.disable_eager_execution()
else:
    # TF 1.x
    tf1 = tf
"""
        content = content.replace('import tensorflow as tf', 'import tensorflow as tf' + compat_code)
    
    # 替换tensorflow.contrib.layers
    if 'tensorflow.contrib.layers' in content:
        content = content.replace('import tensorflow.contrib.layers as layers', 
                                 '# 使用keras layers替代tensorflow.contrib.layers\nlayers = tf.keras.layers')
    
    # 保存更改
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Updated imports in {file_path}")

def fix_common_api_calls(file_path):
    """修复常见的API更改"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 替换Session相关API
    content = re.sub(r'tf\.Session', r'tf1.Session', content)
    content = re.sub(r'tf\.placeholder', r'tf1.placeholder', content)
    content = re.sub(r'tf\.global_variables', r'tf1.global_variables', content)
    content = re.sub(r'tf\.train\.Saver', r'tf1.train.Saver', content)
    content = re.sub(r'tf\.summary\.FileWriter', r'tf1.summary.FileWriter', content)
    content = re.sub(r'tf\.summary\.scalar', r'tf1.summary.scalar', content)
    
    # 替换优化器
    content = re.sub(r'tf\.train\.exponential_decay', r'tf1.train.exponential_decay', content)
    
    # 更新keep_dims为keepdims
    content = re.sub(r'keep_dims=', r'keepdims=', content)
    
    # 更新softmax的dim参数为axis
    content = re.sub(r'tf\.nn\.softmax\((.*), dim=', r'tf.nn.softmax(\1, axis=', content)
    
    # 保存更改
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Fixed common API calls in {file_path}")

def process_all_files():
    """处理所有Python文件"""
    python_files = glob.glob('**/*.py', recursive=True)
    
    for file_path in python_files:
        print(f"Processing {file_path}")
        fix_tf_imports(file_path)
        fix_common_api_calls(file_path)

if __name__ == "__main__":
    print("Starting TensorFlow 1.x to 2.x conversion...")
    process_all_files()
    print("Conversion complete. Please manually review the changes.")
