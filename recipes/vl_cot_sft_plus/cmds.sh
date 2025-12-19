# step:1 建立绝对路径, 执行一次就好. 文件后缀名变为 _abs.jsonl
cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes

uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
  --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
  -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/part-68d4c4a0aff3-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/part-68d4c44d1ad9-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/part-68d4c539ae1e-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/part-68d4c584252f-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/part-68d4b8afcdba-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/part-68d677b6f47f-000086.jsonl \
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/part-68d5ff1f93bf-000086.jsonl

# step2: 复制路径填入脚本, 配置子目录和后缀 (如, sft _sft) , 确定开启 latest (从最新 resume), 启动管线
bash entrypoint/run.sh

# step3: 复制输出路径, 清理框架内部字段, 结果回到原数据路径 带上了后缀 ()
input_paths=(
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d4c4a0aff3-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d4c44d1ad9-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/sft/20251218_150300/part-68d4c539ae1e-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/sft/20251218_040735/part-68d4c584252f-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/sft/20251218_091244/part-68d4b8afcdba-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/sft/20251218_091244/part-68d677b6f47f-000086_abs_sft.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d5ff1f93bf-000086_abs_sft.jsonl
)

for input_path in "${input_paths[@]}"; do
  echo "$(dirname $(dirname $(dirname ${input_path})))/$(basename ${input_path})"
  uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/src/utils/framework.py \
    -i ${input_path} \
    -o  $(dirname $(dirname $(dirname ${input_path})))/$(basename ${input_path})
done

# 更换模型 (qwen30b->235b), 重复 step 2, step3, 管线自动跳过已经完成的任务
