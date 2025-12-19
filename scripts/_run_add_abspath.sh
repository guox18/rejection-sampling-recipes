#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

INPUT_FILES=(
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/part-68d5ff1f93bf-000086_abs_sft.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/part-68d677b6f47f-000086_abs_sft.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/part-68d4b8afcdba-000086_abs_sft.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/part-68d4c584252f-000086_abs_sft.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/part-68d4c539ae1e-000086_abs_sft.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/part-68d4c44d1ad9-000086_abs_sft_abs.jsonl"
  "/mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/part-68d4c4a0aff3-000086_abs_sft.jsonl"
)

for input_file in "${INPUT_FILES[@]}"; do
  # 去除 _sft 后缀，得到对应的 abs 文件路径
  file_with_abs_path="${input_file/_sft.jsonl/.jsonl}"
  
  # 生成输出文件路径（在原文件名后添加 _abs）
  output_file="${input_file/.jsonl/_abs.jsonl}"
  
  echo "Processing: ${input_file}"
  echo "Using abs path file: ${file_with_abs_path}"
  echo "Output to: ${output_file}"
  
  # 运行 add_abspath.py
  python "${SCRIPT_DIR}/_add_abspath.py" \
    --input_path "${input_file}" \
    --file_with_abs_path "${file_with_abs_path}" \
    --output_path "${output_file}"
  
  echo "Completed: ${output_file}"

done