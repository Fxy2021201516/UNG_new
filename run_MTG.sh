#!/bin/bash

# 定义数据集路径变量
DATASET="MTG"
DATA_DIR="../UNG_data/MTG" 
NUM_QUERY_SETS=10 

# 删除 build 文件夹及其所有内容
if [ -d "build_MTG" ]; then
    echo "删除 build_MTG 文件夹及其内容..."
    rm -rf build
fi

# # 下载并解压数据
# # cd $DATA_DIR
# # tar -zxvf MTG.tar.gz
# # cd ..
# # cd UNG

# 创建 build 目录并编译代码
mkdir -p build_MTG
cd build_MTG
cmake -DCMAKE_BUILD_TYPE=Release ../codes/ # Build with Release mode
make -j
cd ..


# # 转换MTG_base数据格式
# ./build_MTG/tools/fvecs_to_bin --data_type float --input_file $DATA_DIR/MTG/MTG_base.fvecs --output_file $DATA_DIR/MTG/MTG_base.bin

# 构建index + 生成查询任务文件
./build_MTG/apps/build_UNG_index \
    --data_type float --dist_fn L2 --num_threads 1 --max_degree 32 --Lbuild 100 --alpha 1.2 \
    --base_bin_file $DATA_DIR/MTG/MTG_base.bin --base_label_file $DATA_DIR/MTG/MTG_base_labels.txt \
    --index_path_prefix $DATA_DIR/index_files/UNG/MTG_base_labels_general_cross6_R32_L100_A1.2/ \
    --scenario general --num_cross_edges 6 \
    --generate_query false --query_file_path $DATA_DIR/MTG/MTG_query \
    --dataset $DATASET> $DATA_DIR/MTG_build_thread=1.log

# # 转换MTG_query数据格式
# for ((i=1; i<=$NUM_QUERY_SETS; i++))
# do
#     INPUT_FILE="$DATA_DIR/MTG/MTG_query/MTG_query_$i/MTG_query.fvecs"
#     OUTPUT_FILE="$DATA_DIR/MTG/MTG_query/MTG_query_$i/MTG_query.bin"
    
#     echo "Processing set $i: $INPUT_FILE -> $OUTPUT_FILE"
#     ./build_MTG/tools/fvecs_to_bin --data_type float --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
# done

# # 生成gt
# for ((i=1; i<=$NUM_QUERY_SETS; i++))
# do
#     QUERY_DIR="$DATA_DIR/MTG/MTG_query/MTG_query_$i"
#     QUERY_BIN="$QUERY_DIR/MTG_query.bin"
#     QUERY_LABELS="$QUERY_DIR/MTG_query_labels.txt"
#     GT_FILE="$QUERY_DIR/MTG_gt_labels_containment.bin"

#     ./build_MTG/tools/compute_groundtruth \
#         --data_type float --dist_fn L2 --scenario containment --K 10 --num_threads 32 \
#         --base_bin_file "$DATA_DIR/MTG/MTG_base.bin" \
#         --base_label_file "$DATA_DIR/MTG/MTG_base_labels.txt" \
#         --query_bin_file "$QUERY_BIN" \
#         --query_label_file "$QUERY_LABELS" \
#         --gt_file "$GT_FILE"

#     # 检查是否成功
#     if [ $? -eq 0 ]; then
#         echo "Successfully generated GT for set $i"
#     else
#         echo "Error generating GT for set $i"
#         exit 1
#     fi
# done
# echo -e "\nAll ground truth files generated successfully!"



RESULT_DIR="$DATA_DIR/results_thread=1"
mkdir -p "$RESULT_DIR"

for ((i=1; i<=$NUM_QUERY_SETS; i++))
do    
    echo -e "\nRunning iteration $i with query set $QUERY_DIR..."
    QUERY_DIR="$DATA_DIR/MTG/MTG_query/MTG_query_$i"

    # 创建当前查询集的结果目录
    CURRENT_RESULT_DIR="$RESULT_DIR/MTG_query_$i"
    mkdir -p "$CURRENT_RESULT_DIR"

    ./build_MTG/apps/search_UNG_index \
        --data_type float --dist_fn L2 --num_threads 1 --K 10 \
        --base_bin_file "$DATA_DIR/MTG/MTG_base.bin" \
        --base_label_file "$DATA_DIR/MTG/MTG_base_labels.txt" \
        --query_bin_file "$QUERY_DIR/MTG_query.bin" \
        --query_label_file "$QUERY_DIR/MTG_query_labels.txt" \
        --gt_file "$QUERY_DIR/MTG_gt_labels_containment.bin" \
        --index_path_prefix "$DATA_DIR/index_files/UNG/MTG_base_labels_general_cross6_R32_L100_A1.2/" \
        --result_path_prefix "$CURRENT_RESULT_DIR/MTG_" \
        --scenario containment \
        --num_entry_points 16 \
        --Lsearch 10 50 300 500 1000 1200 3000 3500 4000 4500 5000 5500 6000

    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo "Successfully completed iteration $i"
    else
        echo "Error in iteration $i"
        exit 1
    fi
done

echo -e "\nAll search iterations completed successfully!"

