import random
import os

root_path = '/workspace/data/avikus/motorboat/005'

# 파일에서 전체 목록 읽기
with open(os.path.join(root_path, "frame_list.txt"), "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# 고정 시드로 셔플
random.seed(42)
random.shuffle(lines)

# 분할
test_lines = lines[:10]
val_lines = lines[10:20]
train_lines = lines[20:]

# 인덱스 재정렬
test_lines = [f"{i} {line.split()[1]}" for i, line in enumerate(test_lines)]
val_lines = [f"{i} {line.split()[1]}" for i, line in enumerate(val_lines)]
train_lines = [f"{i} {line.split()[1]}" for i, line in enumerate(train_lines)]

# 결과 저장
with open(os.path.join(root_path, "test.txt"), "w") as f:
    f.write("\n".join(test_lines))

with open(os.path.join(root_path, "val.txt"), "w") as f:
    f.write("\n".join(val_lines))

with open(os.path.join(root_path, "train.txt"), "w") as f:
    f.write("\n".join(train_lines))

print("Done: test.txt / val.txt / train.txt 생성 완료")
