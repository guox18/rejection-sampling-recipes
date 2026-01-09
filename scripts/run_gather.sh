#!/bin/bash
set -euo pipefail

# 数据处理脚本调用示例

cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts

# 路径推导脚本
# 输入: 235b 文件路径列表
# 输出: 自动推导出 30b 路径、原文件路径和 train 路径，并调用 python 脚本

# 235b 文件路径列表
INPUT_235B_LIST=(
  # # gb p0
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d5ff1f93bf-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d677b6f47f-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/sft-235b/20251220_100343/part-68d4b8afcdba-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d4c584252f-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d4c539ae1e-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d4c44d1ad9-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/sft-235b/20251219_144900/part-68d4c4a0aff3-000086_abs_sft-30b_sft-235b.jsonl"

  # # gb p1
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Chart_Markdown~unknown~synthChart_flowchart_md_v1_0_jsonl~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d628a12d2e-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_common_sense_v2~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d603fd2b58-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_language_v2~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d6117db8b9-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_reasoning_text_v2~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d611cfba8d-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Multiple_Image_K12_Comprehensive~zh~k12_merge_ab_zh_20240812~1.0.0~0.0/jsonl/sft-235b/20251224_155002/part-68d595016ec7-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~mmmu_wimg_test_subqa_33k_v1_20241127check~1.0.0~0.0/jsonl/sft-235b/20251224_131934/part-68d5ed381400-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~VisuLogic~1.0.0~0.0/jsonl/sft-235b/20251224_155002/part-68dc48e039ce-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~gaokao_chinese_0809_train_r1461~1.0.0~0.0/jsonl/sft-235b/20251224_131934/part-68d5e3f2ab51-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~gaokao_history_0809_train_r2324~1.0.0~0.0/jsonl/sft-235b/20251224_131934/part-68d5e41db409-000086_abs_sft-30b_sft-235b.jsonl"
  # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Conversation_LongQA~zh~crawler_emoji_gpt4o_zh_20240628~1.0.0~0.0/jsonl/sft-235b/20251224_131934/part-68d5895485c0-000086_abs_sft-30b_sft-235b.jsonl"

  # sy p0
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~kaoyan_mllm_by_doc_parse~2.0.0~0.0/jsonl/sft-235b/20251221_160803/part-68dc4ada2601-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~shyc_mllm~1.0.0~0.0/jsonl/sft-235b/20251221_160803/part-68d75a9610db-000086_abs_sft-30b_sft-235b.jsonl"
  ## 缺了 100 多条 # "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Math_MCQ~en~tqa_en_20240402_sft_final~1.0.0~0.0/jsonl/sft-235b/20251221_160803/part-68d3e0ca9187-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~ai2d_en_20240410~1.0.0~0.0/jsonl/sft-235b/20251221_160803/part-68d4c4997fb1-000086_abs_sft-30b_sft-235b.jsonl"

  # sy p1
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_2~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d7cc9feef7-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_3~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d7cbd81b75-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_4~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d7cbca21fd-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~mmmu_tiku_gpt4o_20240626_18k_filtered_valid16k_20240816_v1~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d5e794d66d-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Math_CoT~en~tqa_cot_gpt4o_en_20240621_final~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d3e186a678-000086_abs_sft-30b_sft-235b.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_CoT~en~ai2d_cot_gpt4o_en_20240805~1.0.0~0.0/jsonl/sft-235b/20251224_105026/part-68d4c4421465-000086_abs_sft-30b_sft-235b.jsonl"
)

# 遍历文件路径列表
for INPUT_235B in "${INPUT_235B_LIST[@]}"; do
    # 提取文件名
    FILENAME=$(basename "$INPUT_235B")
    
    # 去除 _abs 和 .jsonl 之间的部分，得到原始文件名
    ORIGINAL_FILENAME=$(echo "$FILENAME" | sed 's/_abs.*\.jsonl/.jsonl/')
    
    # 获取 235b 文件的目录
    DIR_235B=$(dirname "$INPUT_235B")
    
    # 倒退两级目录，得到 jsonl 目录
    JSONL_DIR=$(dirname $(dirname "$DIR_235B"))
    
    # 原文件路径
    ORIGINAL_PATH="${JSONL_DIR}/${ORIGINAL_FILENAME}"
    
    # train 路径（原文件名去掉 .jsonl 后缀，加上 _train.jsonl）
    TRAIN_PATH="${JSONL_DIR}/$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_train.jsonl/')"
    
    # 找到 sft-30b 目录下最新的日期子文件夹
    SFT_30B_DIR="${JSONL_DIR}/sft-30b"
    LATEST_30B_DATE=$(ls -1 "$SFT_30B_DIR" | sort -r | head -n 1)
    
    # 30b 文件路径（原始文件名去掉 .jsonl，加上 _abs_sft-30b.jsonl）
    FILENAME_30B=$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_abs_sft-30b.jsonl/')
    PATH_30B="${SFT_30B_DIR}/${LATEST_30B_DATE}/${FILENAME_30B}"
    
    # 打印推导结果
    echo "train 路径: $TRAIN_PATH"
    
    # 检查文件是否存在（除了 train）
    if [ ! -f "$INPUT_235B" ]; then
        echo "错误: 235b 文件不存在: $INPUT_235B"
        exit 1
    fi
    
    if [ ! -f "$PATH_30B" ]; then
        echo "错误: 30b 文件不存在: $PATH_30B"
        exit 1
    fi
    
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "错误: 原文件不存在: $ORIGINAL_PATH"
        exit 1
    fi
    
    # 调用 gather.py
    python gather.py \
        --input_path "$ORIGINAL_PATH" \
        --qwen3vl30ba3bthinking_path "$PATH_30B" \
        --qwen3vl235ba22bthinking_path "$INPUT_235B" \
        --output_path "$TRAIN_PATH"
done

echo "所有文件处理完成！"
