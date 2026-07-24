import random
import os
import argparse
import pdb 

RATIO_VAL = 0.1
RATIO_TEST = 0.1

def shorten_path(path, root_dir):
    if root_dir in path:
        return path.split(root_dir, 1)[1].lstrip(os.sep).replace("\\", "/")
    return path

def main(args):
    args.data_root = os.path.normpath(args.data_root)
    root_path = args.data_root  # data 전체의 root path

    # 파일에서 전체 목록 읽기
    lines = []
    i = 0
    dir_path_list = {}
    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == "label":
            for fname in filenames:
                if fname.endswith('.txt'):
                    full_path = os.path.join(dirpath, fname)
                    # # full_path를 읽었을 때 비어있지 않다면 append
                    # with open(full_path, "r") as f:
                    #     content = f.read().strip()
                    #     if not content:   # 내용이 비어있다면
                    #         continue
                    # -> negative sample도 학습 데이터에 추가하도록 변경
                    lines.append(full_path)
                    i += 1
                    if dirpath not in dir_path_list:
                        dir_path_list[dirpath] = 0
                    else:
                        dir_path_list[dirpath] += 1

    # 고정 시드로 셔플
    random.seed(42)
    random.shuffle(lines)

    # 분할
    total_lines = len(lines)
    num_test = int(RATIO_TEST * total_lines)
    num_val = int(RATIO_VAL * total_lines)

    test_lines = lines[:num_test]
    val_lines = lines[num_test:num_test+num_val]
    train_lines = lines[num_test+num_val:]

    # 인덱스 재정렬
    root_dir = root_path.split('/')[-1]
    test_lines = [f"{i} {shorten_path(line, root_dir)}" for i, line in enumerate(test_lines)]
    val_lines = [f"{i} {shorten_path(line, root_dir)}" for i, line in enumerate(val_lines)]
    train_lines = [f"{i} {shorten_path(line, root_dir)}" for i, line in enumerate(train_lines)]

    # 결과 저장
    with open(os.path.join(root_path, "test.txt"), "w") as f:
        f.write("\n".join(test_lines))

    with open(os.path.join(root_path, "val.txt"), "w") as f:
        f.write("\n".join(val_lines))

    with open(os.path.join(root_path, "train.txt"), "w") as f:
        f.write("\n".join(train_lines))

    print("Done: test.txt / val.txt / train.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='pillar_logs')
    args = parser.parse_args()
    main(args)