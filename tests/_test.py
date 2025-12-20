import json

# 读取 jsonl 文件的第一条数据
with open("/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/tests/_test.jsonl", "r") as f:
    item = json.loads(f.readline())

print(type(item.get("metadata")))
print(type(item.get("metadata")))

existing_answer = (item.get("metadata") or {}).get("answer")
print(existing_answer)
