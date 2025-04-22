#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <vector>
#include <queue>

#include <random>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <bitset>

#include "utils.h"
#include "vamana/vamana.h"
#include "include/uni_nav_graph.h"

namespace fs = boost::filesystem;

// 文件格式常量
const std::string FVEC_EXT = ".fvecs";
const std::string TXT_EXT = ".txt";
const size_t FVEC_HEADER_SIZE = sizeof(uint32_t);
struct QueryTask
{
   ANNS::IdxType vec_id;         // 向量ID
   std::vector<uint32_t> labels; // 查询标签集
};

namespace ANNS
{

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      prepare_group_storages_graphs();
      _label_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::high_resolution_clock::now() - start_time)
                                   .count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      build_global_vamana_graph();
      build_vector_and_attr_graph();

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();

         // calculate the coverage ratio
         cal_f_coverage_ratio();

         // build cross-group edges
         build_cross_group_edges();
      }

      // index time
      _index_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::build_trie_and_divide_groups()
   {

      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // obtain the candidates
      std::vector<std::shared_ptr<TrieNode>> candidates;
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment);

      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // obtain the minimum size
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // get the minimum super sets
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         // add to the minimum super sets
         if (is_min)
            min_super_set_ids.emplace_back(cur_group_id);
      }
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::build_graph_for_all_groups()
   {
      std::cout << "Building graph for each group ..." << std::endl;
      omp_set_num_threads(_num_threads);
      auto start_time = std::chrono::high_resolution_clock::now();

      // build vamana index
      if (_index_name == "Vamana")
      {
         _vamana_instances.resize(_num_groups + 1);
         _group_entry_points.resize(_num_groups + 1);

#pragma omp parallel for schedule(dynamic, 1)
         for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (group_id % 100 == 0)
               std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;

            // if there are less than _max_degree points in the group, just build a complete graph
            const auto &range = _group_id_to_range[group_id];
            if (range.second - range.first <= _max_degree)
            {
               build_complete_graph(_group_graphs[group_id], range.second - range.first);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);

               // build the vamana graph
            }
            else
            {
               _vamana_instances[group_id] = std::make_shared<Vamana>(false);
               _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                                  _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
            }

            // set entry point
            _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         }

         // if none of the above
      }
      else
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }

      _build_graph_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
   }

   // fxy_add：构建全局Vamana图
   void UniNavGraph::build_global_vamana_graph()
   {
      std::cout << "Building global Vamana graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _global_vamana = std::make_shared<Vamana>(false);

      _global_vamana->build(_base_storage, _distance_handler, _global_graph, _max_degree, _Lbuild, _alpha, 1);

      auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      _global_vamana_entry_point = _global_vamana->get_entry_point();

      std::cout << "- Global Vamana graph built in " << build_time << " ms" << std::endl;
   }
   //=====================================begin 数据预处理：构建向量-属性二分图=========================================
   // fxy_add: 构建向量-属性二分图
   void UniNavGraph::build_vector_and_attr_graph()
   {
      std::cout << "Building vector-attribute bipartite graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // 初始化属性到ID的映射和反向映射
      _attr_to_id.clear();
      _id_to_attr.clear();
      _vector_attr_graph.clear();

      // 第一遍：收集所有唯一属性并分配ID
      AtrType attr_id = 0;
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            if (_attr_to_id.find(label) == _attr_to_id.end())
            {
               _attr_to_id[label] = attr_id;
               _id_to_attr[attr_id] = label;
               attr_id++;
            }
         }
      }

      // 初始化图结构（向量节点 + 属性节点）
      _vector_attr_graph.resize(_num_points + static_cast<size_t>(attr_id));

      // 第二遍：构建图结构
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            AtrType a_id = _attr_to_id[label];

            // 添加双向边
            // 向量节点ID范围: [0, _num_points-1]
            // 属性节点ID范围: [_num_points, _num_points+attr_id-1]
            _vector_attr_graph[vec_id].push_back(_num_points + static_cast<IdxType>(a_id));
            _vector_attr_graph[_num_points + static_cast<IdxType>(a_id)].push_back(vec_id);
         }
      }

      // 统计信息
      _num_attributes = attr_id;
      auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- Built bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in " << build_time << " ms" << std::endl;
      std::cout << "- Total edges: " << count_graph_edges() << std::endl;

      // 可选：保存图结构供调试
      save_bipartite_graph_info();
   }

   // fxy_add: 计算向量-属性二分图的边数
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2; // 因为是双向边，实际边数是总数的一半
   }

   // fxy_add: 保存二分图信息到txt文件调试用
   void UniNavGraph::save_bipartite_graph_info() const
   {
      std::ofstream outfile("bipartite_graph_info.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Warning: Could not open file to save bipartite graph info" << std::endl;
         return;
      }

      outfile << "Bipartite Graph Information\n";
      outfile << "==========================\n";
      outfile << "Total vectors: " << _num_points << "\n";
      outfile << "Total attributes: " << _num_attributes << "\n";
      outfile << "Total edges: " << count_graph_edges() << "\n\n";

      // 输出属性映射
      outfile << "Attribute to ID Mapping:\n";
      for (const auto &pair : _attr_to_id)
      {
         outfile << pair.first << " -> " << static_cast<IdxType>(pair.second) << "\n";
      }
      outfile << "\n";

      // 输出部分图结构示例
      outfile << "Sample Graph Connections (first 10 vectors and attributes):\n";
      outfile << "Vector connections:\n";
      for (IdxType i = 0; i < std::min(_num_points, static_cast<IdxType>(10)); ++i)
      {
         outfile << "Vector " << i << " connects to attributes: ";
         for (auto a_node : _vector_attr_graph[i])
         {
            AtrType attr_id = static_cast<AtrType>(a_node - _num_points);
            outfile << _id_to_attr.at(attr_id) << " ";
         }
         outfile << "\n";
      }

      outfile << "\nAttribute connections:\n";
      for (AtrType i = 0; i < std::min(_num_attributes, static_cast<AtrType>(5)); ++i)
      {
         outfile << "Attribute " << _id_to_attr.at(i) << " connects to vectors: ";
         for (auto v_node : _vector_attr_graph[_num_points + static_cast<IdxType>(i)])
         {
            outfile << v_node << " ";
         }
         outfile << "\n";
      }

      outfile.close();
      std::cout << "- Bipartite graph info saved to bipartite_graph_info.txt" << std::endl;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      // 简单的校验和计算示例
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   // fxy_add: 保存二分图到文件
   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      // 1. 写入文件头标识和版本
      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      // 2. 写入基本元数据
      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      // 3. 写入属性映射表
      // 3.1 先写入条目数量
      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      // 3.2 写入每个映射条目（LabelType是uint16_t，直接存储）
      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      // 4. 写入邻接表数据
      // 4.1 先写入节点总数
      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      // 4.2 写入每个节点的邻居列表
      for (const auto &neighbors : _vector_attr_graph)
      {
         // 先写入邻居数量
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         // 写入邻居ID列表
         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      // 5. 写入文件尾校验和
      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   // fxy_add: 读取二分图
   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      // 1. 验证文件头
      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      // 2. 读取基本元数据
      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      // 3. 读取属性映射表
      _attr_to_id.clear();
      _id_to_attr.clear();

      // 3.1 读取条目数量
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      // 3.2 读取每个映射条目
      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      // 4. 读取邻接表数据
      _vector_attr_graph.clear();

      // 4.1 读取节点总数
      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      // 4.2 读取每个节点的邻居列表
      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      // 5. 验证校验和
      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }

   // fxy_add: 比较两个向量-属性二分图
   bool UniNavGraph::compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2)
   {
      // 1. 验证基本属性
      if (g1._num_points != g2._num_points)
      {
         std::cerr << "Mismatch in _num_points: "
                   << g1._num_points << " vs " << g2._num_points << std::endl;
         return false;
      }

      if (g1._num_attributes != g2._num_attributes)
      {
         std::cerr << "Mismatch in _num_attributes: "
                   << g1._num_attributes << " vs " << g2._num_attributes << std::endl;
         return false;
      }

      // 2. 验证属性映射
      if (g1._attr_to_id.size() != g2._attr_to_id.size())
      {
         std::cerr << "Mismatch in _attr_to_id size" << std::endl;
         return false;
      }

      for (const auto &[label, id] : g1._attr_to_id)
      {
         auto it = g2._attr_to_id.find(label);
         if (it == g2._attr_to_id.end())
         {
            std::cerr << "Label " << label << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != id)
         {
            std::cerr << "Mismatch ID for label " << label
                      << ": " << id << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 3. 验证反向属性映射
      for (const auto &[id, label] : g1._id_to_attr)
      {
         auto it = g2._id_to_attr.find(id);
         if (it == g2._id_to_attr.end())
         {
            std::cerr << "ID " << id << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != label)
         {
            std::cerr << "Mismatch label for ID " << id
                      << ": " << label << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 4. 验证邻接表
      if (g1._vector_attr_graph.size() != g2._vector_attr_graph.size())
      {
         std::cerr << "Mismatch in graph size" << std::endl;
         return false;
      }

      for (size_t i = 0; i < g1._vector_attr_graph.size(); ++i)
      {
         const auto &neighbors1 = g1._vector_attr_graph[i];
         const auto &neighbors2 = g2._vector_attr_graph[i];

         if (neighbors1.size() != neighbors2.size())
         {
            std::cerr << "Mismatch in neighbors count for node " << i << std::endl;
            return false;
         }

         for (size_t j = 0; j < neighbors1.size(); ++j)
         {
            if (neighbors1[j] != neighbors2[j])
            {
               std::cerr << "Mismatch in neighbor " << j << " for node " << i
                         << ": " << neighbors1[j] << " vs " << neighbors2[j] << std::endl;
               return false;
            }
         }
      }

      std::cout << "Graphs are identical!" << std::endl;

      return true;
   }
   //=====================================end 数据预处理：构建向量-属性二分图=========================================

   //=====================================begein 查询过程：计算bitmap=========================================

   // fxy_add: 构建bitmap
   std::vector<bool> UniNavGraph::compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const
   {
      // 1. 初始化全true的bitmap（表示开始时所有点都满足条件）
      std::vector<bool> bitmap(_num_points, true);

      // 2. 处理每个查询属性
      for (LabelType attr_label : query_attributes)
      {
         //  2.1 查找属性ID
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end())
         {
            // 属性不存在，没有任何点能满足所有条件
            return std::vector<bool>(_num_points, false);
         }

         // 2.2 获取属性节点ID
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);

         // 2.3 创建临时bitmap记录当前属性的满足情况
         std::vector<bool> temp_bitmap(_num_points, false);
         for (IdxType vec_id : _vector_attr_graph[attr_node_id])
         {
            if (vec_id < _num_points)
            {
               temp_bitmap[vec_id] = true;
            }
         }

         // 2.4 与主bitmap进行AND操作
         for (IdxType i = 0; i < _num_points; ++i)
         {
            bitmap[i] = bitmap[i] && temp_bitmap[i];
         }
      }

      return bitmap;
   }

   //====================================end 查询过程：计算bitmap=========================================
   void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points)
   {
      for (auto i = 0; i < num_points; ++i)
         for (auto j = 0; j < num_points; ++j)
            if (i != j)
               graph->neighbors[i].emplace_back(j);
   }

   //=====================================begin LNG中每个f覆盖率计算=========================================
   // fxy_add: 递归打印孩子节点及其覆盖率
   void print_children_recursive(const std::shared_ptr<ANNS::LabelNavGraph> graph, IdxType group_id, std::ofstream &outfile, int indent_level)
   {
      // 根据缩进级别生成前缀空格
      std::string indent(indent_level * 4, ' '); // 每一层缩进4个空格

      // 打印当前节点
      outfile << "\n"
              << indent << "[" << group_id << "] (" << graph->coverage_ratio[group_id] << ")";

      // 如果有子节点，递归打印子节点
      if (!graph->out_neighbors[group_id].empty())
      {
         for (size_t i = 0; i < graph->out_neighbors[group_id].size(); ++i)
         {
            auto child_group_id = graph->out_neighbors[group_id][i];
            print_children_recursive(graph, child_group_id, outfile, indent_level + 1); // 增加缩进级别
         }
      }
   }

   // fxy_add: 输出覆盖率及孩子节点,调用print_children_recursive
   void output_coverage_ratio(const std::shared_ptr<ANNS::LabelNavGraph> _label_nav_graph, IdxType _num_groups, std::ofstream &outfile)
   {
      outfile << "Coverage Ratio for each group\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] -> CoverageRatio \n";
      outfile << "-> [child_GroupID (CoverageRatio)]\n\n";

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 打印当前组的覆盖率
         outfile << "[" << group_id << "] -> " << _label_nav_graph->coverage_ratio[group_id];

         // 打印孩子节点
         if (!_label_nav_graph->out_neighbors[group_id].empty())
         {
            outfile << " ->";
            for (size_t i = 0; i < _label_nav_graph->out_neighbors[group_id].size(); ++i)
            {
               auto child_group_id = _label_nav_graph->out_neighbors[group_id][i];
               print_children_recursive(_label_nav_graph, child_group_id, outfile, 1); // 第一层子节点缩进为1
            }
         }
         outfile << "\n";
      }
   }

   // fxy_add：计算每个label set的向量覆盖比率
   void UniNavGraph::cal_f_coverage_ratio()
   {
      // step1：初始化，计算每个label set的向量覆盖比率(有且仅有)
      std::cout << "Calculating coverage ratio..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph->coverage_ratio.resize(_num_groups + 1, 0.0);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &label_set = _group_id_to_label_set[group_id];
         if (label_set.empty())
            continue;

         double coverage_ratio = static_cast<double>(_group_id_to_vec_ids[group_id].size()) / _num_points;
         _label_nav_graph->coverage_ratio[group_id] = coverage_ratio;
      }

      // step2：检查出度为0的group并压入队列
      std::queue<IdxType> q;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->out_neighbors[group_id].empty())
         {
            q.push(group_id);
         }
      }
      std::cout << "- Number of leaf nodes: " << q.size() << std::endl;

      // step3：从队列中取出group，更新其父节点的覆盖比率
      while (!q.empty())
      {
         IdxType current = q.front();
         q.pop();

         // 更新父节点的覆盖比率
         for (auto parent : _label_nav_graph->in_neighbors[current])
         {
            _label_nav_graph->coverage_ratio[parent] += _label_nav_graph->coverage_ratio[current];
            _label_nav_graph->out_degree[parent] -= 1;
            if (_label_nav_graph->out_degree[parent] == 0)
            {
               q.push(parent);
            }
         }
      }

      // 时间
      auto _coverage_ratio_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::high_resolution_clock::now() - start_time)
                                      .count();

      std::cout << "- Coverage ratio calculated in " << _coverage_ratio_time << " ms" << std::endl;

      // step4：递归存储所有孩子的覆盖比率
      std::ofstream outfile0("LNG_coverage_ratio.txt");
      output_coverage_ratio(_label_nav_graph, _num_groups, outfile0);
      outfile0.close();

      // step5：存储每个group的覆盖比率和标签集 Format: GroupID CoverageRatio LabelSet
      std::ofstream outfile("group_coverage_ratio.txt");
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         outfile << group_id << " " << _label_nav_graph->coverage_ratio[group_id] << " ";
         for (const auto &label : _group_id_to_label_set[group_id])
            outfile << label << " ";
         outfile << "\n";
      }

      // step6：存储每个group里面有几个向量 Format: GroupID NumVectors
      std::ofstream outfile1("group_num_vectors.txt");
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         outfile1 << group_id << " " << _group_id_to_vec_ids[group_id].size() << "\n";
   }

   // =====================================end LNG中每个f覆盖率计算=========================================

   /*void UniNavGraph::build_label_nav_graph() {
       std::cout << "Building label navigation graph... " << std::endl;
       auto start_time = std::chrono::high_resolution_clock::now();
       _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups+1);
       omp_set_num_threads(_num_threads);

       // obtain out-neighbors
       #pragma omp parallel for schedule(dynamic, 256)
       for (auto group_id=1; group_id<=_num_groups; ++group_id) {
           if (group_id % 100 == 0)
               std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
           std::vector<IdxType> min_super_set_ids;
           get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
           _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
       }

       // obtain in-neighbors
       for (auto group_id=1; group_id<=_num_groups; ++group_id)
           for (auto each : _label_nav_graph->out_neighbors[group_id])
               _label_nav_graph->in_neighbors[each].emplace_back(group_id);

       _build_LNG_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - start_time).count();
       std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   }*/

   // fxy_add : 打印信息的build_label_nav_graph
   void UniNavGraph::build_label_nav_graph()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

      std::ofstream outfile("lng_structure.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Error: Could not open lng_structure.txt for writing!" << std::endl;
         return;
      }
      outfile << "Label Navigation Graph (LNG) Structure\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] {LabelSet} -> [OutNeighbor1]{LabelSet}, [OutNeighbor2]{LabelSet}, ...\n\n";

// obtain out-neighbors
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (group_id % 100 == 0)
         {
#pragma omp critical
            std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
         }

         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
         _label_nav_graph->out_degree[group_id] = min_super_set_ids.size();

#pragma omp critical
         {
            outfile << "[" << group_id << "] {";
            // 打印标签集
            for (const auto &label : _group_id_to_label_set[group_id])
               outfile << label << ",";
            outfile << "} -> ";

            // 打印出边（包含目标节点的标签集）
            for (size_t i = 0; i < min_super_set_ids.size(); ++i)
            {
               auto target_id = min_super_set_ids[i];
               outfile << "[" << target_id << "] {";
               // 打印目标节点的标签集
               for (const auto &label : _group_id_to_label_set[target_id])
               {
                  outfile << label << ",";
               }
               outfile << "}";
               if (i != min_super_set_ids.size() - 1)
                  outfile << ", ";
            }
            outfile << "\n";
         }
      }

      // obtain in-neighbors (不需要打印入边，但保留原有逻辑)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (auto each : _label_nav_graph->out_neighbors[group_id])
         {
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);
            _label_nav_graph->in_degree[group_id] += 1;
         }
      }

      // outfile.close();
      _build_LNG_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
      std::cout << "- LNG structure saved to lng_structure.txt" << std::endl;
   }

   // 将分组内的局部索引转换为全局索引
   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

   void UniNavGraph::build_cross_group_edges()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      // allocate memory for search caches
      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      // for each group
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].size() > 0)
         {
            if (group_id % 100 == 0)
               std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            IdxType offset = _group_id_to_range[group_id].first;

            // query vamana index
            if (_index_name == "Vamana")
            {
               auto index = _vamana_instances[group_id];
               if (_num_cross_edges > _Lbuild)
               {
                  std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                  exit(-1);
               }

               // for each in-neighbor group
               for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
               {
                  const auto &range = _group_id_to_range[in_group_id];

// take each vector in the group as the query
#pragma omp parallel for schedule(dynamic, 1)
                  for (auto vec_id = range.first; vec_id < range.second; ++vec_id)
                  {
                     const char *query = _base_storage->get_vector(vec_id);
                     auto search_cache = search_cache_list.get_free_cache();
                     index->iterate_to_fixed_point(query, search_cache);

                     // update the cross-group edges for vec_id
                     for (auto k = 0; k < search_cache->search_queue.size(); ++k)
                        cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                             search_cache->search_queue[k].distance);
                     search_cache_list.release_cache(search_cache);
                  }
               }

               // if none of the above
            }
            else
            {
               std::cerr << "Error: invalid index name " << _index_name << std::endl;
               exit(-1);
            }
         }
      }

      // add additional edges
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &cur_range = _group_id_to_range[group_id];
         std::unordered_set<IdxType> connected_groups;

         // obtain connected groups
         for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

         // add additional cross-group edges for unconnected groups
         for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            if (connected_groups.find(out_group_id) == connected_groups.end())
            {
               IdxType cnt = 0;
               for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
               {
                  auto search_cache = search_cache_list.get_free_cache();
                  _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                  for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                  {
                     additional_edges[group_id].emplace_back(vec_id,
                                                             search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                     cnt += 1;
                  }
                  search_cache_list.release_cache(search_cache);
               }
            }
      }

      // add offset for uni-nav graph
      add_offset_for_uni_nav_graph();

// merge cross-group edges
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

// merge additional cross-group edges
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }

      _build_cross_edges_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
   }

   void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                            uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                            IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                            std::vector<std::vector<bool>> &bitmap)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // for overlap or nofilter scenario
         if (scenario == "overlap" || scenario == "nofilter")
         {
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            cur_result.reserve(K);

            // obtain entry group
            std::vector<IdxType> entry_group_ids;
            if (scenario == "overlap")
               get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
            else
               get_min_super_sets({}, entry_group_ids, true, true);

            // for each entry group
            for (const auto &group_id : entry_group_ids)
            {
               std::vector<IdxType> entry_points;
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

               // graph search and dump to current result
               num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false);
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
            }

            // for the other scenarios: containment, equality
         }
         else
         {

            // obtain entry points
            auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
            if (entry_points.empty())
            {
               num_cmps[id] = 0;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               continue;
            }

            // graph search
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
            cur_result = search_cache->search_queue;
         }

         // write results
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
               results[id * K + k].first = -1;
         }

         // clean
         search_cache_list.release_cache(search_cache);
      }
   }

   // fxy_add
   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> query_storage,
                                   std::shared_ptr<DistanceHandler> distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<std::vector<bool>> &bitmaps)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // 参数设置
      const float COVERAGE_THRESHOLD = 0.5f;   // 覆盖率阈值
      const int MIN_ENTRY_SETS_THRESHOLD = 10; // 最小入口集数量阈值

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         search_cache->search_queue.clear();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // 获取查询标签集
         const auto &query_labels = _query_storage->get_label_set(id);

         // 计算最小超集的覆盖率总和
         std::vector<IdxType> entry_group_ids;
         get_min_super_sets(query_labels, entry_group_ids, false, false);

         float total_coverage = 0.0f;
         for (auto group_id : entry_group_ids)
         {
            total_coverage += _label_nav_graph->coverage_ratio[group_id];
         }

         // 决定使用哪种搜索策略
         bool use_global_search = (total_coverage > COVERAGE_THRESHOLD) ||
                                  (entry_group_ids.size() > MIN_ENTRY_SETS_THRESHOLD);

         if (use_global_search)
         {
            // 使用全局图搜索 (DisKANN)
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            search_cache->search_queue.clear();
            cur_result.reserve(K);

            // 从全局入口点开始搜索
            std::vector<IdxType> global_entry_points;
            if (_global_vamana_entry_point != -1)
            {
               global_entry_points.push_back(_global_vamana_entry_point);
            }
            else
            {
               // 如果没有预定义的全局入口点，随机选择几个
               for (int i = 0; i < num_entry_points; ++i)
               {
                  global_entry_points.push_back(rand() % _num_points);
               }
            }

            // 全局图搜索
            num_cmps[id] = iterate_to_fixed_point_global(query, search_cache, id, global_entry_points);

            // 过滤结果 - 保留前K个满足条件的邻居
            int valid_count = 0;
            for (size_t k = 0; k < search_cache->search_queue.size() && valid_count < K; k++)
            {
               auto candidate = search_cache->search_queue[k];
               const auto &candidate_labels = _base_storage->get_label_set(candidate.id);

               // 检查候选是否满足查询条件
               bool is_valid = true;
               if (scenario == "equality")
               {
                  // 对于equality场景，检查标签是否完全匹配
                  is_valid = (candidate_labels == query_labels);
               }

               // 使用bitmaps进行额外的过滤
               if (bitmaps.size() > id && bitmaps[id].size() > candidate.id)
               {
                  if (scenario == "containment")
                  {
                     is_valid = bitmaps[id][candidate.id];
                  }
                  else
                  {
                     is_valid = is_valid && bitmaps[id][candidate.id];
                  }
               }

               if (is_valid)
               {
                  cur_result.insert(candidate.id, candidate.distance);
                  valid_count++;
               }
            }
         }
         else
         {
            if (scenario == "overlap" || scenario == "nofilter")
            {
               num_cmps[id] = 0;
               search_cache->visited_set.clear();
               cur_result.reserve(K);

               // 获取入口组
               std::vector<IdxType> entry_group_ids;
               if (scenario == "overlap")
               {
                  get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
               }
               else
               {
                  get_min_super_sets({}, entry_group_ids, true, true);
               }

               // 对每个入口组进行搜索
               for (const auto &group_id : entry_group_ids)
               {
                  std::vector<IdxType> entry_points;
                  get_entry_points_given_group_id(num_entry_points, search_cache->visited_set,
                                                  group_id, entry_points);

                  // 图搜索并合并结果
                  num_cmps[id] += iterate_to_fixed_point(query, search_cache, id,
                                                         entry_points, true, false);
                  for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  {
                     cur_result.insert(search_cache->search_queue[k].id,
                                       search_cache->search_queue[k].distance);
                  }
               }
            }
            else
            {
               // 获取入口点
               auto entry_points = get_entry_points(_query_storage->get_label_set(id),
                                                    num_entry_points, search_cache->visited_set);
               if (entry_points.empty())
               {
                  num_cmps[id] = 0;
                  for (auto k = 0; k < K; ++k)
                  {
                     results[id * K + k].first = -1;
                  }
                  continue;
               }

               // 图搜索
               num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
               cur_result = search_cache->search_queue;
            }
         }

         // 写入结果
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
            {
               results[id * K + k].first = -1;
            }
         }

         // 清理
         search_cache_list.release_cache(search_cache);
      }
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points, VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      // obtain entry points for label-equality scenario
      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);

         // obtain entry points for label-containment scenario
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                                     IdxType group_id, std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      // not enough entry points, use all of them
      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      // add the entry point of the group
      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      // randomly sample the other entry points
      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id, const std::vector<IdxType> &entry_points,
                                               bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
            neighbors = _graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   // fxy_add
   IdxType UniNavGraph::iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_global_graph->neighbor_locks[cur.id]);
            neighbors = _global_graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   void UniNavGraph::save(std::string index_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["index_time(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // save LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

      // save vector attr graph data
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      save_bipartite_graph(vector_attr_graph_filename);

      // print
      std::cout << "- Index saved in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::load(std::string index_path_prefix, const std::string &data_type)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      // fxy_add:load global graph data
      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      // fxy_add: load global vamana entry point
      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // fxy_add: load LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      // print
      std::cout << "- Index loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::statistics()
   {

      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
            _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      // return as MB
      _index_size /= 1024 * 1024;
   }

   // fxy_add: 生成查询向量和标签
   void UniNavGraph::query_generate(std::string &output_prefix, int n, float keep_prob, bool stratified_sampling, bool verify)
   {
      std::ofstream fvec_file(output_prefix + FVEC_EXT, std::ios::binary);
      std::ofstream txt_file(output_prefix + "_labels" + TXT_EXT);

      uint32_t dim = _base_storage->get_dim();
      std::cout << "Vector dimension: " << dim << std::endl;
      std::cout << "Number of points: " << _num_points << std::endl;

      // 随机数生成器
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

      std::vector<QueryTask> all_queries;
      size_t total_queries = 0;

      // min(7000个查询,_num_groups)
      total_queries = std::min(7000, int(_num_groups));
      for (ANNS::IdxType group_id = 1; group_id <= total_queries; ++group_id)
      {
         // std::cout << "group_id: " << group_id << std::endl;
         auto [start, end] = _group_id_to_range[group_id];
         size_t group_size = end - start;
         if (group_size == 0)
            continue;

         // 计算该组实际采样数量
         int sample_num = n;
         if (stratified_sampling)
         {
            sample_num = std::max(1, static_cast<int>(n * group_size / _num_points)); // 分层采样：按组大小比例调整采样数
         }
         sample_num = std::min(sample_num, static_cast<int>(group_size));

         // 非重复采样
         std::vector<ANNS::IdxType> vec_ids(group_size);
         std::iota(vec_ids.begin(), vec_ids.end(), start);
         std::shuffle(vec_ids.begin(), vec_ids.end(), gen);

         for (int i = 0; i < sample_num; ++i) // 每个组采样的个数
         {
            ANNS::IdxType vec_id = vec_ids[i];
            if (_base_storage->get_label_set(vec_id).empty())
            {
               continue; // 跳过无 base 属性的向量
            }
            QueryTask task;
            task.vec_id = vec_id;

            // 生成查询标签集
            for (auto label : _group_id_to_label_set[group_id])
            {
               if (prob_dist(gen) <= keep_prob)
               {
                  task.labels.push_back(label);
               }
            }

            // 确保至少保留一个标签
            if (task.labels.empty())
            {
               task.labels.push_back(*_group_id_to_label_set[group_id].begin());
            }

            // 验证标签是组的子集
            if (verify)
            {
               auto &group_labels = _group_id_to_label_set[group_id];
               for (auto label : task.labels)
               {
                  assert(std::find(group_labels.begin(), group_labels.end(), label) != group_labels.end());
               }
            }

            all_queries.push_back(task);
         }
      }

      // 写入fvecs文件前验证标签
      std::ofstream verify_file(output_prefix + "_verify.txt"); // 新增验证输出文件
      for (const auto &task : all_queries)
      {
         // 获取基础存储中的原始标签集
         const auto &original_labels = _base_storage->get_label_set(task.vec_id);

         // 写入验证文件（无论是否开启verify都记录）
         verify_file << task.vec_id << " base_labels:";
         for (auto l : original_labels)
            verify_file << " " << l;
         verify_file << " query_labels:";
         for (auto l : task.labels)
            verify_file << " " << l;
         verify_file << "\n";

         // 验证查询标签是原始标签的子集
         if (verify)
         {
            for (auto label : task.labels)
            {
               if (std::find(original_labels.begin(), original_labels.end(), label) == original_labels.end())
               {
                  std::cerr << "Error: Label " << label << " not found in original label set for vector " << task.vec_id << std::endl;
                  std::cerr << "Original labels: ";
                  for (auto l : original_labels)
                     std::cerr << l << " ";
                  std::cerr << "\nQuery labels: ";
                  for (auto l : task.labels)
                     std::cerr << l << " ";
                  std::cerr << std::endl;
                  throw std::runtime_error("Label verification failed");
               }
            }
         }
      }
      verify_file.close(); // 关闭验证文件

      // 写入fvecs文件
      for (const auto &task : all_queries)
      {
         const char *vec_data = _base_storage->get_vector(task.vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t)); // 每个向量前写维度
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      // 写入txt文件
      for (const auto &task : all_queries)
      {
         // txt_file << task.vec_id;
         for (auto label : task.labels)
         {
            txt_file << label << ",";
         }
         txt_file << "\n";
      }

      std::cout << "Generated " << all_queries.size() << " queries\n";
      std::cout << "FVECS file: " << output_prefix + FVEC_EXT << "\n";
      std::cout << "TXT file: " << output_prefix + TXT_EXT << "\n";
   }

   // fxy_add:生成多个查询任务
   void UniNavGraph::generate_multiple_queries(
       std::string dataset,
       UniNavGraph &index,
       const std::string &base_output_path,
       int num_sets,
       int n_per_set,
       float keep_prob,
       bool stratified_sampling,
       bool verify)
   {
      std::cout << "enter generate_multiple_queries" << std::endl;
      namespace fs = std::filesystem;

      // 确保基础目录存在
      fs::create_directories(base_output_path);

      for (int i = 1; i <= num_sets; ++i)
      {
         std::cout << "Generating query set " << i << "..." << std::endl;
         std::string folder_name = base_output_path + "/" + dataset + "_query_" + std::to_string(i);

         // 创建目录（包括所有必要的父目录）
         fs::create_directories(folder_name);

         std::string output_prefix = folder_name + "/" + dataset + "_query"; // 路径在文件夹内
         index.query_generate(output_prefix, n_per_set, keep_prob, stratified_sampling, verify);

         std::cout << "Generated query set " << i << " at " << folder_name << std::endl;
      }
   }

}