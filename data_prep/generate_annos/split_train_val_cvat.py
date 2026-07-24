import random
import os
import argparse
import pdb
from generate_label import EXCLUDE_PATH

def main(args):
    data_root = args.data_root

    # 파일에서 전체 목록 읽기
    lines = []
    for data_name in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, data_name)) or data_name in  EXCLUDE_PATH:
            continue
        with open(os.path.join(data_root, data_name, "frame_list.txt"), "r") as f:
            for line in f.readlines():
                if line.strip():
                    path = os.path.join(data_name, 'label', line.split()[-1]+'.txt')
                    lines.append(path)

    # 고정 시드로 셔플
    random.seed(42)
    random.shuffle(lines)

    # 분할
    total_lines = len(lines)
    num_test = int(0.1 * total_lines)
    num_val = int(0.1 * total_lines)
    num_train = total_lines - num_test - num_val

    test_lines = lines[:num_test]
    val_lines = lines[num_test:num_test+num_val]
    train_lines = lines[num_test+num_val:]

    assert (num_test + num_val + num_train == total_lines)

    # 인덱스 재정렬
    test_lines = [f"{i} {line}" for i, line in enumerate(test_lines)]
    val_lines = [f"{i} {line}" for i, line in enumerate(val_lines)]
    train_lines = [f"{i} {line}" for i, line in enumerate(train_lines)]

    # 결과 저장
    with open(os.path.join(data_root, "test.txt"), "w") as f:
        f.write("\n".join(test_lines))

    with open(os.path.join(data_root, "val.txt"), "w") as f:
        f.write("\n".join(val_lines))

    with open(os.path.join(data_root, "train.txt"), "w") as f:
        f.write("\n".join(train_lines))

    print("Done: test.txt / val.txt / train.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='pillar_logs')
    args = parser.parse_args()
    main(args)