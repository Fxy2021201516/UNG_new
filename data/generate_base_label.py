from collections import defaultdict, deque
import random
from itertools import combinations

# def generate_labels(num_vectors, min_dim=1, max_dim=5):
#     """生成指定数量的标签，控制每个标签的维度范围"""
#     labels = []
#     dim = 0  # 当前维度
    
#     # 生成初始单维度标签
#     while len(labels) < num_vectors and dim <= max_dim:
#         labels.append(frozenset({dim}))
#         dim += 1
    
#     # 如果还需要更多标签，开始组合
#     next_dim = dim
#     while len(labels) < num_vectors and next_dim <= max_dim:
#         new_labels = []
#         # 尝试用现有标签组合新标签
#         for i in range(len(labels)):
#             if len(labels[i]) < max_dim:
#                 new_label = labels[i].union({next_dim})
#                 if new_label not in labels and new_label not in new_labels:
#                     new_labels.append(new_label)
#                     if len(labels) + len(new_labels) >= num_vectors:
#                         break
#         labels.extend(new_labels)
#         next_dim += 1
    
#     # 确保标签维度在指定范围内
#     labels = [label for label in labels if min_dim <= len(label) <= max_dim]
    
#     # 如果还不够，随机生成
#     while len(labels) < num_vectors:
#         import random
#         dims = random.randint(min_dim, max_dim)
#         new_label = frozenset(random.sample(range(max_dim+1), dims))
#         if new_label not in labels:
#             labels.append(new_label)
    
#     return labels[:num_vectors]  # 确保不超出数量


def generate_labels(num_vectors, min_dim=1, max_dim=5, single_prob=0.3, combo_prob=0.5):
    """
    生成 num_vectors 个标签（frozenset），用于表示向量所属的组。
    
    参数：
        num_vectors (int): 需要生成的总标签数量
        min_dim (int): 标签最小包含的维度数
        max_dim (int): 标签最大包含的维度数
        single_prob (float): 生成一个一维标签的概率
        combo_prob (float): 生成一个组合标签的概率（非一维）
        
    返回：
        list of frozenset: 包含 num_vectors 个标签的列表，可以重复
    """
    all_dims = list(range(max_dim + 1))  # 所有可用维度 [0, 1, ..., max_dim]
    used_labels = set()  # 已经生成的标签集合，防止重复定义时浪费资源
    result_labels = []   # 最终输出的标签列表，可以重复

    while len(result_labels) < num_vectors:
        # 决定本次生成哪种类型的标签
        p = random.random()

        if p < single_prob:
            # 生成一维标签：随机选一个维度
            dim = random.choice(all_dims/5)
            label = frozenset({dim})
        elif p < single_prob + combo_prob:
            # 生成多维组合标签：随机选择维度数量和具体维度
            dims_count = random.randint(min_dim, max_dim)
            label = frozenset(random.sample(all_dims, dims_count))
        else:
            # 如果没有合适标签或想增加多样性，也可以随机从已用标签中取一个
            if used_labels:
                label = random.choice(list(used_labels))
            else:
                # fallback：如果还没有任何标签，就随机生成一个
                label = frozenset({random.choice(all_dims)})
        
        # 可以直接添加（允许重复）
        result_labels.append(label)
        used_labels.add(label)  # 记录唯一标签（去重存储）

    return result_labels

def build_lng_graph(labels):
    """构建LNG图：B是A的最小超集当且仅当 A ⊂ B 且不存在中间节点 C (A ⊂ C ⊂ B)"""
    graph = {label: set() for label in labels}
    labels_sorted = sorted(labels, key=lambda x: len(x))  # 按维度升序处理
    
    for i, A in enumerate(labels_sorted):
        for B in labels_sorted[i+1:]:
            if A.issubset(B):
                # 检查是否存在中间节点 C
                is_minimal = True
                for C in labels_sorted[i+1:]:
                    if C != B and A.issubset(C) and C.issubset(B):
                        is_minimal = False
                        break
                if is_minimal:
                    graph[A].add(B)
    return graph

def calculate_max_depth(graph):
    """计算图的最大深度"""
    in_degree = {node: 0 for node in graph}
    depth = {node: 0 for node in graph}
    
    # 计算入度
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # 拓扑排序
    queue = deque([node for node in graph if in_degree[node] == 0])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if depth[neighbor] < depth[node] + 1:
                depth[neighbor] = depth[node] + 1
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return max(depth.values()) if depth else 0

def count_descendants(graph, node):
    """计算单个节点的所有后代节点数量"""
    visited = set()
    queue = deque([node])
    count = 0
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                count += 1
    return count

def calculate_avg_descendants(graph):
    """计算所有节点的平均后代数量"""
    total = 0
    for node in graph:
        total += count_descendants(graph, node)
    return total / len(graph) if graph else 0

def print_graph(graph):
    """打印图的边"""
    for A in sorted(graph.keys(), key=lambda x: (len(x), sorted(x))):
        for B in sorted(graph[A], key=lambda x: (len(x), sorted(x))):
            print(f"{set(A)} -> {set(B)}")

# 用户输入
num_vectors = int(input("请输入向量数量: "))
min_dim = int(input("请输入最小维度: "))
max_dim = int(input("请输入最大维度: "))

# 生成标签并构建LNG图
labels = generate_labels(num_vectors, min_dim, max_dim)
graph = build_lng_graph(labels)
max_depth = calculate_max_depth(graph)
avg_descendants = calculate_avg_descendants(graph)

print("\nGenerated Labels:")
for label in sorted(labels, key=lambda x: (len(x), sorted(x))):
    print(set(label))
label_sizes = [len(label) for label in labels]
print(f"Label Sizes: {label_sizes}")

print("\nLNG Graph Edges:")
print_graph(graph)

print(f"\nMaximum Depth of LNG Graph: {max_depth}")
print(f"Average Number of Descendants: {avg_descendants:.2f}")

# 打印每个节点的后代数量
print("\nDescendants Count for Each Node:")
for node in sorted(graph.keys(), key=lambda x: (len(x), sorted(x))):
    print(f"{set(node)}: {count_descendants(graph, node)} descendants")