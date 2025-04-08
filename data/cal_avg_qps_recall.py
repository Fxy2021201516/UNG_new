# 将跑出的多次测试结果求平均值放到一个csv文件中

import os
import csv
from collections import defaultdict

def calculate_averages(results_dir):
    # 初始化数据结构来存储所有数据
    data = defaultdict(list)
    
    # 遍历results目录下的所有words_query_*文件夹
    for folder in os.listdir(results_dir):
        if folder.startswith('words_query_'):
            file_path = os.path.join(results_dir, folder, 'words_result.csv')
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        L = int(row['L'])
                        # 将数据转换为浮点数并存储
                        data[L].append({
                            'Cmps': float(row['Cmps']),
                            'QPS': float(row['QPS']),
                            'Recall': float(row['Recall'])
                        })
    
    # 计算每个L的平均值
    averages = {}
    for L in sorted(data.keys()):
        entries = data[L]
        count = len(entries)
        
        avg_Cmps = sum(e['Cmps'] for e in entries) / count
        avg_QPS = sum(e['QPS'] for e in entries) / count
        avg_Recall = sum(e['Recall'] for e in entries) / count
        
        averages[L] = {
            'Cmps': avg_Cmps,
            'QPS': avg_QPS,
            'Recall': avg_Recall,
            'Count': count  # 包含有多少次测试包含这个L值
        }
    
    return averages

def save_averages_to_csv(averages, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['L', 'Avg_Cmps', 'Avg_QPS', 'Avg_Recall', 'Test_Count'])
        
        for L in sorted(averages.keys()):
            avg = averages[L]
            writer.writerow([
                L,
                round(avg['Cmps'], 2),
                round(avg['QPS'], 2),
                round(avg['Recall'], 2),
                avg['Count']
            ])

if __name__ == '__main__':
    results_directory = '/home/fengxiaoyao/UNG_data/words/results_thread=1'
    output_csv = '/home/fengxiaoyao/UNG_data/words/results_thread=1/average_results.csv'
    
    averages = calculate_averages(results_directory)
    save_averages_to_csv(averages, output_csv)
    
    print(f"平均值已计算并保存到 {output_csv}")
    print("\n结果预览:")
    for L in sorted(averages.keys()):
        print(f"L={L}: Cmps={averages[L]['Cmps']:.2f}, QPS={averages[L]['QPS']:.2f}, Recall={averages[L]['Recall']:.2f} (来自{averages[L]['Count']}次测试)")