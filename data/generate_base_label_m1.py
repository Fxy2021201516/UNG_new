import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

def generate_unique_attr_vectors(n, m, k, required_attrs, min_coverage):
    """
    属性生成方法，确保所有必需属性都达到最小覆盖率（属性值范围：1~k）
    参数：
        n: 总向量数
        m: 每个向量的属性数
        k: 属性最大值（属性值范围为 1~k）
        required_attrs: 需要保证覆盖率的属性列表
        min_coverage: 每个属性需要的最小覆盖率(0-1)
    """
    # 参数校验
    if len(required_attrs) > m:
        raise ValueError(f"无法满足要求：每个向量只有{m}个属性，但需要保证{len(required_attrs)}个属性的覆盖率")
    
    if any(attr < 1 or attr > k for attr in required_attrs):
        raise ValueError("required_attrs 中的属性必须在 1 到 k 范围内")

    data = []
    attr_counts = {a: 0 for a in required_attrs}
    target_count = int(n * min_coverage)
    
    # 第一阶段：为每个必需属性创建专用向量
    for a in required_attrs:
        needed = max(0, target_count - attr_counts[a])
        
        # 创建包含该属性的向量，尽可能包含其他必需属性
        for _ in range(needed):
            other_req_attrs = [ra for ra in required_attrs 
                             if ra != a and attr_counts[ra] < target_count]
            num_other_req = min(len(other_req_attrs), m-1)
            
            # 选择其他属性
            other_attrs = []
            if num_other_req > 0:
                other_attrs = list(np.random.choice(other_req_attrs, size=num_other_req, replace=False))
            
            # 补充随机属性
            remaining = m - 1 - len(other_attrs)
            if remaining > 0:
                available = [x for x in range(1, k+1) if x not in [a] + other_attrs]
                other_attrs.extend(np.random.choice(available, size=remaining, replace=False))
            
            vec = np.array([a] + other_attrs)
            np.random.shuffle(vec)
            data.append(vec)
            
            # 更新计数
            for attr in vec:
                if attr in attr_counts:
                    attr_counts[attr] += 1
    
    # 第二阶段：填充剩余向量
    while len(data) < n:
        under_covered = [a for a in required_attrs if attr_counts[a] < target_count]
        
        if under_covered:
            a = np.random.choice(under_covered)
            available = [x for x in range(1, k+1) if x != a]
            other_attrs = np.random.choice(available, size=m-1, replace=False)
            vec = np.append(a, other_attrs)
        else:
            vec = np.random.choice(range(1, k+1), size=m, replace=False)
        
        np.random.shuffle(vec)
        data.append(vec)
        
        # 更新计数
        for a in required_attrs:
            if a in vec:
                attr_counts[a] += 1
    
    # 最终强制修正
    for a in required_attrs:
        if attr_counts[a] < target_count:
            deficit = target_count - attr_counts[a]
            candidates = [i for i, vec in enumerate(data) if a not in vec]
            
            for i in np.random.choice(candidates, size=deficit, replace=False):
                replaceable = [x for x in data[i] if x not in required_attrs or 
                             (x in required_attrs and attr_counts[x] > target_count)]
                if not replaceable:
                    continue
                
                to_replace = np.random.choice(replaceable)
                data[i] = np.where(data[i] == to_replace, a, data[i])
                attr_counts[a] += 1
                if to_replace in attr_counts:
                    attr_counts[to_replace] -= 1
    
    # 验证
    attr_counts = {a: 0 for a in required_attrs}
    for vec in data:
        for a in required_attrs:
            if a in vec:
                attr_counts[a] += 1
    
    print("\n最终属性覆盖率验证:")
    for a in required_attrs:
        coverage = attr_counts[a] / n
        print(f"属性 {a}: {coverage:.2%} (目标: {min_coverage:.0%})")
        if coverage < min_coverage:
            raise RuntimeError(f"无法满足属性 {a} 的覆盖率要求")
    
    return data

def map_to_groups_with_ids(data):
    """
    将原始向量按属性集合分组，并记录每个 group 来自哪些原始向量的索引（ID）
    返回：
        groups: list of frozenset，每个是一个唯一的 group
        group_id_map: dict，group -> group ID
        group_members: list of list，每个元素是属于该 group 的原始向量 ID 列表
    """
    from collections import defaultdict

    group_members = defaultdict(list)  # 每个 group 对应的原始向量 ID 列表

    for idx, vec in enumerate(data):
        group = frozenset(vec)
        group_members[group].append(idx)

    groups = list(group_members.keys())
    group_id_map = {group: gid for gid, group in enumerate(groups)}
    members_list = [group_members[group] for group in groups]

    return groups, group_id_map, members_list


def count_group_with_attributes(groups, attrs=[0, 1, 2]):
    """
    统计有多少 group 包含指定属性
    """
    counts = {a: 0 for a in attrs}
    total = len(groups)

    for group in groups:
        for a in attrs:
            if a in group:
                counts[a] += 1

    print("Group 中包含指定属性的比例：")
    for a in attrs:
        print(f"属性 {a}出现次数: {counts[a]}")
        print(f"总组数: {total}")
        print(f"属性 {a}: {counts[a]/total:.2%}")
    return counts


def build_inverted_index_for_groups(groups):
    """
    构建倒排索引：属性 -> group ID 列表
    """
    index = defaultdict(list)
    for idx, group in enumerate(groups):
        for val in group:
            index[val].append(idx)
    return index


def analyze_query_coverage(index, total_groups, attrs=[0, 1, 2]):
    """
    分析查询覆盖率：查询某个属性能命中多少个 group
    """
    print("\n查询覆盖率分析：")
    for a in attrs:
        coverage = len(set(index[a])) / total_groups
        print(f"查询属性 {a} 能定位到 {coverage:.2%} 的 entry point")

def save_vectors_to_txt(data, filename="vectors.txt"):
    """
    将生成的向量数据保存为 txt 文件
    每行是一个向量，属性之间用空格分隔
    """
    with open(filename, 'w') as f:
        for vec in data:
            line = ','.join(map(str, sorted(vec)))  # 排序便于查看
            f.write(line + '\n')
    print(f"Vectors saved to {filename}")

def print_group_details(groups, members_list):
    print("\n📊 Group Details:")
    for gid, (group, members) in enumerate(zip(groups, members_list)):
        print(f"Group {gid}:")
        print(f"  属性集合: {sorted(group)}")
        print(f"  向量数量: {len(members)}")
        print(f"  原始向量索引: {members}")

def print_attribute_to_groups_map(groups, inverted_index):
    print("\n🏷️ Attribute to Groups Mapping:")
    for attr, group_ids in sorted(inverted_index.items()):
        print(f"属性 {attr}:,出现在 {len(group_ids)} 个 group")
        print(f"  出现在以下 group(s): {group_ids}")
        print(f"  group 内容: {[sorted(groups[gid]) for gid in group_ids]}")
# -----------------------------
# 示例运行函数
# -----------------------------

def run_example(dataset_name="word", n=20, m=4, k=8, required_attrs=[0, 1, 2], min_coverage=0.3):
    print(f"\n--- Running Example for Dataset: {dataset_name} ---")

    # 设置随机种子以保证可复现性
    np.random.seed(42)

    # Step 1: 生成数据 (确保满足覆盖率要求)
    print("Step 1: Generating data with coverage requirements...")
    data = generate_unique_attr_vectors(n, m, k, required_attrs, min_coverage)

    # 新增：保存数据到 txt 文件
    save_vectors_to_txt(data, filename=f"{dataset_name}_base_labels.txt")
    
    # Step 2: 分组
    print("Step 2: Mapping to groups (f-points)...")
    groups, group_id_map, members_list = map_to_groups_with_ids(data)
    total_groups = len(groups)
    print(f"Total number of f-points (groups): {total_groups}")

   #  # 新增：打印 group 详情
   #  print_group_details(groups, members_list)

    # Step 3: 属性统计
    print("Step 3: Counting attributes in groups...")
    counts = count_group_with_attributes(groups, required_attrs)
    
    # 验证group级别的覆盖率
    for a in required_attrs:
        coverage = counts[a] / total_groups
        assert coverage >= min_coverage, f"属性 {a} 在group级别的覆盖率不足"

    # Step 4: 建立倒排索引
    print("Step 4: Building inverted index for querying...")
    inverted_index = build_inverted_index_for_groups(groups)

    # Step 5: 查询覆盖率分析
    print("Step 5: Analyzing query coverage...")
    analyze_query_coverage(inverted_index, total_groups, required_attrs)

    print(f"\n--- End of Example for Dataset: {dataset_name} ---")
    return {
        'data': data,
        'groups': groups,
        'inverted_index': inverted_index,
        'total_groups': total_groups
    }
# -----------------------------
# 主程序入口
# -----------------------------

if __name__ == "__main__":
    # 生成n个向量，每个m个属性，属性值范围0-k,要求属性required_attrs在每个group中的覆盖率至少min_coverage
    results_word = run_example("words", n=8000, m=6, k=26, required_attrs=[1,2,3], min_coverage=0.3)