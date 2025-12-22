input_paths=(
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d4c4a0aff3-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d4c44d1ad9-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/sft/20251218_150300/part-68d4c539ae1e-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/sft/20251218_040735/part-68d4c584252f-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/sft/20251218_091244/part-68d4b8afcdba-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/sft/20251218_091244/part-68d677b6f47f-000086_abs_sft.jsonl
  # /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/sft/20251218_041512/part-68d5ff1f93bf-000086_abs_sft.jsonl

  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~kaoyan_mllm_by_doc_parse~2.0.0~0.0/jsonl/sft-30b/20251220_070142/part-68dc4ada2601-000086_abs_sft-30b.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~shyc_mllm~1.0.0~0.0/jsonl/sft-30b/20251220_091500/part-68d75a9610db-000086_abs_sft-30b.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Math_MCQ~en~tqa_en_20240402_sft_final~1.0.0~0.0/jsonl/sft-30b/20251220_070142/part-68d3e0ca9187-000086_abs_sft-30b.jsonl
  /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~ai2d_en_20240410~1.0.0~0.0/jsonl/sft-30b/20251220_070142/part-68d4c4997fb1-000086_abs_sft-30b.jsonl
)

for input_path in "${input_paths[@]}"; do
  echo "$(dirname $(dirname $(dirname ${input_path})))/$(basename ${input_path})"
  uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/src/utils/framework.py \
    -i ${input_path} \
    -o  $(dirname $(dirname $(dirname ${input_path})))/$(basename ${input_path})
done