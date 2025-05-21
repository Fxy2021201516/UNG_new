#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include <unordered_set>
#include "config.h"

namespace ANNS
{

   class LabelNavGraph
   {

   public:
      LabelNavGraph(IdxType num_nodes)
      {
         in_neighbors.resize(num_nodes + 1);
         out_neighbors.resize(num_nodes + 1);
         coverage_ratio.resize(num_nodes + 1, 0.0); // 存储每个节点的覆盖比例
         covered_sets.resize(num_nodes + 1);        // 存储每个节点的覆盖集合
         in_degree.resize(num_nodes + 1, 0);
         out_degree.resize(num_nodes + 1, 0);

         _lng_descendants_num.resize(num_nodes + 1); // group_id, descendants_count
      };

      std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;
      std::vector<double> coverage_ratio;                    // 每个 label set 的覆盖比例
      std::vector<std::unordered_set<IdxType>> covered_sets; // 每个 group 的覆盖向量集合
      std::vector<int> in_degree, out_degree;                // 入度和出度

      std::vector<std::pair<IdxType, int>> _lng_descendants_num; // group_id, descendants_count
      double avg_descendants;                                    // 平均后代数量
      std::vector<std::unordered_set<IdxType>> _lng_descendants; // 每个 group 的覆盖的group的集合
      ~LabelNavGraph() = default;

   private:
   };
}

#endif // LABEL_NAV_GRAPH_H