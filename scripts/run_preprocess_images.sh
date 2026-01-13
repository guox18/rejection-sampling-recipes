cd /path/to/rejection-sampling-recipes

# gb p0
uv run python /path/to/rejection-sampling-recipes/scripts/preprocess_images.py \
  --image-base-dir /path/to/data/internvl3_5 \
  -i /path/to/data/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_multi_choice_en_20240402~1.0.0~0.0/jsonl/part-68d4c4a0aff3-000086.jsonl \
  /path/to/data/internvl3_5/P~Single_Image_Science_MCQ~en~scienceqa_choice_aug_en_20240402~1.0.0~0.0/jsonl/part-68d4c44d1ad9-000086.jsonl