#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
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
         in_degree.resize(num_nodes + 1, 0);
         out_degree.resize(num_nodes + 1, 0);
      };

      std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;
      std::vector<double> coverage_ratio;     // 每个 label set 的覆盖比例
      std::vector<int> in_degree, out_degree; // 入度和出度

      ~LabelNavGraph() = default;

   private:
   };
}

#endif // LABEL_NAV_GRAPH_H