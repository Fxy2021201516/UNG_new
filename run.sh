#!/bin/bash

# 定义数据集路径变量
DATA_DIR="../UNG_data/words" # 将此路径替换为数据集的实际位置

# 删除 build 文件夹及其所有内容
if [ -d "build" ]; then
    echo "删除 build 文件夹及其内容..."
    rm -rf build
fi

# 下载并解压数据
# cd $DATA_DIR
# tar -zxvf words.tar.gz
# cd ..
# cd UNG

# 创建 build 目录并编译代码
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../codes/ # Build with Release mode
make -j
cd ..


# 转换words_base数据格式
./build/tools/fvecs_to_bin --data_type float --input_file $DATA_DIR/words/words_base.fvecs --output_file $DATA_DIR/words/words_base.bin

# 构建index + 生成查询任务文件
./build/apps/build_UNG_index \
    --data_type float --dist_fn L2 --num_threads 32 --max_degree 32 --Lbuild 100 --alpha 1.2 \
    --base_bin_file $DATA_DIR/words/words_base.bin --base_label_file $DATA_DIR/words/words_base_labels.txt \
    --index_path_prefix $DATA_DIR/index_files/UNG/words_base_labels_general_cross6_R32_L100_A1.2/ \
    --scenario general --num_cross_edges 6 \
    --generate_query true --query_file_path $DATA_DIR/words/words_query 

# 转换words_query数据格式
./build/tools/fvecs_to_bin --data_type float --input_file $DATA_DIR/words/words_query.fvecs --output_file $DATA_DIR/words/words_query.bin

# 生成gt
./build/tools/compute_groundtruth \
    --data_type float --dist_fn L2 --scenario containment --K 10 --num_threads 32 \
    --base_bin_file $DATA_DIR/words/words_base.bin --base_label_file $DATA_DIR/words/words_base_labels.txt \
    --query_bin_file $DATA_DIR/words/words_query.bin --query_label_file $DATA_DIR/words/words_query_labels.txt \
    --gt_file $DATA_DIR/words/words_gt_labels_containment.bin

# 查询 gt_mode: 
# 0: ground truth is distance and in binary file, 1: ground truth is cost and in csv file, calculate costs, 
# 2: ground truth is cost and in csv file, read costs

# 定义存储结果的目录
RESULT_DIR="$DATA_DIR/results"
# mkdir -p $RESULT_DIR

for ((i=1; i<=1; i++))
do
    echo "Running iteration $i..."

   ./build/apps/search_UNG_index \
      --data_type float --dist_fn L2 --num_threads 16 --K 10 \
      --base_bin_file $DATA_DIR/words/words_base.bin --base_label_file $DATA_DIR/words/words_base_labels.txt \
      --query_bin_file $DATA_DIR/words/words_query.bin --query_label_file $DATA_DIR/words/words_query_labels.txt \
      --gt_file $DATA_DIR/words/words_gt_labels_containment.bin \
      --index_path_prefix $DATA_DIR/index_files/UNG/words_base_labels_general_cross6_R32_L100_A1.2/ \
      --result_path_prefix $RESULT_DIR/UNG/words_base_labels_containment_cross6_R32_L100_A1.2/ \
      --scenario containment --num_entry_points 16 --Lsearch 10 50 300 500 1000 1200 3000 3500 4000

    echo "Iteration $i"
done

