import random
import os
import argparse

def main(args):
    root_path = args.data_root

    # 파일에서 전체 목록 읽기
    with open(os.path.join(root_path, "frame_list.txt"), "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', required=True, default='pillar_logs')
    args = parser.parse_args()
    main(args)