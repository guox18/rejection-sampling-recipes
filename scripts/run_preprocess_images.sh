cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes

# # gb p0
# uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
#   --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
#   -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/part-68d4c4a0aff3-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/part-68d4c44d1ad9-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_ShortQA~en~viquae_en_20240402~1.0.0~0.0/jsonl/part-68d4c539ae1e-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Knowledge_MCQ~en~koniq10k_en_20240403~1.0.0~0.0/jsonl/part-68d4c584252f-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_General_MCQ~en~ccbench_inhouse_part1_zh_20240401~1.0.0~0.0/jsonl/part-68d4b8afcdba-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~en~scienceqa_choice_augment_en_20240402~1.0.0~0.0/jsonl/part-68d677b6f47f-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_QA~unknown~Calligraphy_Recognition_qa_d20241104_jsonl~1.0.0~0.0/jsonl/part-68d5ff1f93bf-000086.jsonl

# gb p1
## 有个文件有 53w 行: /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Multiple_Image_K12_Comprehensive~zh~k12_merge_ab_zh_20240812~1.0.0~0.0/jsonl/part-68d595016ec7-000086.jsonl

# uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
#   --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
#   -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Conversation_LongQA~zh~crawler_emoji_gpt4o_zh_20240628~1.0.0~0.0/jsonl/part-68d5895485c0-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~gaokao_history_0809_train_r2324~1.0.0~0.0/jsonl/part-68d5e41db409-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~gaokao_chinese_0809_train_r1461~1.0.0~0.0/jsonl/part-68d5e3f2ab51-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~VisuLogic~1.0.0~0.0/jsonl/part-68dc48e039ce-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~gaokao_chinese_0809_train_r1461~1.0.0~0.0/jsonl/part-68d5e3f2ab51-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Multiple_Image_K12_Comprehensive~zh~k12_merge_ab_zh_20240812~1.0.0~0.0/jsonl/part-68d595016ec7-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_reasoning_text_v2~1.0.0~0.0/jsonl/part-68d611cfba8d-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_language_v2~1.0.0~0.0/jsonl/part-68d6117db8b9-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Document_Chart_QA~en~xingce_zhenti_common_sense_v2~1.0.0~0.0/jsonl/part-68d603fd2b58-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Chart_Markdown~unknown~synthChart_flowchart_md_v1_0_jsonl~1.0.0~0.0/jsonl/part-68d628a12d2e-000086.jsonl


## 漏了一个
uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
  --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
  -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~mmmu_wimg_test_subqa_33k_v1_20241127check~1.0.0~0.0/jsonl/part-68d5ed381400-000086.jsonl \

# # sy p0
# uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
#   --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
#   -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~shyc_mllm~1.0.0~0.0/jsonl/part-68d75a9610db-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~kaoyan_mllm_by_doc_parse~2.0.0~0.0/jsonl/part-68dc4ada2601-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_MCQ~en~ai2d_en_20240410~1.0.0~0.0/jsonl/part-68d4c4997fb1-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p0/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Math_MCQ~en~tqa_en_20240402_sft_final~1.0.0~0.0/jsonl/part-68d3e0ca9187-000086.jsonl

# # sy p1
# uv run python /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/preprocess_images.py \
#   --image-base-dir /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-h-delivery/internvl_delivery/internvl3_5 \
#   -i /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Science_CoT~en~ai2d_cot_gpt4o_en_20240805~1.0.0~0.0/jsonl/part-68d4c4421465-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Single_Image_Math_CoT~en~tqa_cot_gpt4o_en_20240621_final~1.0.0~0.0/jsonl/part-68d3e186a678-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~Science~unknown~mmmu_tiku_gpt4o_20240626_18k_filtered_valid16k_20240816_v1~1.0.0~0.0/jsonl/part-68d5e794d66d-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~MSEarth_num7667160_20250621~3.0.0~0.0/jsonl/part-68e87f0f1440-000186.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~MSEarth_num7667160_20250621~3.0.0~0.0/jsonl/part-68e87f0f1440-000057.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_4~1.0.0~0.0/jsonl/part-68d7cbca21fd-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_3~1.0.0~0.0/jsonl/part-68d7cbd81b75-000086.jsonl \
#   /mnt/shared-storage-user/songdemin/user/guoxu/tanghuanze/local_bak_1219_sunyu_p1/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~high_school_mllm_choice_2~1.0.0~0.0/jsonl/part-68d7cc9feef7-000086.jsonl


