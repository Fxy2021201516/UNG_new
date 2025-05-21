import os
import pandas as pd
from itertools import product
import re

# 定义文件夹路径
folder_path = '/data/fxy/UNG_data/words/results/words_query_1'

# 获取所有CSV文件
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 用于存储需要配对的前缀（比如：words_query_details_L50）
base_names = set()

# 提取基础名（去掉 _ung 或正常的后缀）
for file in all_files:
    base_name = re.sub(r'_ung\.csv$|\.csv$', '', file)
    base_names.add(base_name)

# 创建输出目录（如果不存在）
output_dir = '/data/fxy/UNG_data/words/results/words_query_1/merged_results'
os.makedirs(output_dir, exist_ok=True)

# 遍历每个基础名进行合并
for base in base_names:
    file_ung = os.path.join(folder_path, f"{base}_ung.csv")
    file_new = os.path.join(folder_path, f"{base}.csv")

    # 检查两个文件是否存在
    if not (os.path.exists(file_ung) and os.path.exists(file_new)):
        print(f"缺少对应文件对：{file_ung} 或 {file_new}")
        continue

    # 读取两个CSV文件
    df_ung = pd.read_csv(file_ung)
    df_new = pd.read_csv(file_new)

    # 合并列
    merge_columns = ['QueryID', 'EntryPoints', 'LNGDescendants', 'entry_group_total_coverage']
    compare_columns = ['Time(ms)', 'DistanceCalcs', 'QPS', 'Recall', 'is_global_search']

    # 使用 merge 进行内连接（确保 QueryID 等字段一致）
    merged_df = pd.merge(df_ung, df_new, on=merge_columns, suffixes=('_ung', '_new'))

    # 重命名列方便识别
    merged_df.rename(columns={col: col + '_ung' for col in compare_columns}, inplace=True)
    merged_df.rename(columns={col: col + '_new' for col in compare_columns}, inplace=True)

    # 输出文件路径
    output_file = os.path.join(output_dir, f"merged_{base}.csv")

    # 保存结果
    merged_df.to_csv(output_file, index=False)
    print(f"已合并并保存：{output_file}")