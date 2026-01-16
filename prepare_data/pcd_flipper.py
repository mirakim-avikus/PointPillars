import os
import numpy as np
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Flip Avikus PCD Data")
    parser.add_argument("--dataset_dir", type=str, required=True, nargs='+', help="Dataset directory")
    return parser.parse_args()

def read_binary_pcd(file_path):
    with open(file_path, 'rb') as f:
        lines = []
        while True:
            line = f.readline()
            lines.append(line)
            if line.startswith(b'DATA binary'):
                break

        header = b''.join(lines).decode('utf-8')
        
        # 파싱
        for line in header.splitlines():
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes = list(map(int, line.split()[1:]))
            elif line.startswith('TYPE'):
                types = line.split()[1:]
            elif line.startswith('COUNT'):
                counts = list(map(int, line.split()[1:]))
            elif line.startswith('POINTS'):
                num_points = int(line.split()[1])

        point_size = sum([s * c for s, c in zip(sizes, counts)])
        raw_data = f.read(point_size * num_points)

        dtype_list = []
        for field, size, type_, count in zip(fields, sizes, types, counts):
            np_type = None
            if type_ == 'F':
                np_type = np.float32 if size == 4 else np.float64
            elif type_ == 'U':
                np_type = np.uint8
            elif type_ == 'I':
                np_type = np.int32
            if count == 1:
                dtype_list.append((field, np_type))
            else:
                dtype_list.append((field, np_type, (count,)))

        dtype = np.dtype(dtype_list)
        points = np.frombuffer(raw_data, dtype=dtype)

        return header, points

def save_pcd_with_intensity(file_path, header, points):
    # 포인트 개수 업데이트
    num_points = points.shape[0]
    header = re.sub(r'POINTS \d+', f'POINTS {num_points}', header)
    header = re.sub(r'WIDTH \d+', f'WIDTH {num_points}', header)

    with open(file_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(points.astype(np.float32).tobytes())  # binary 형식 저장

def main():
    args = parse_args()
    dataset_dir = args.dataset_dir[0]
    input_dir = os.path.join(dataset_dir, "lidar", "Data")
    output_dir = os.path.join(dataset_dir, "lidar", "flippedData")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".avikus.pcd"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # PCD 읽기 (binary)
            header, pcd_data = read_binary_pcd(input_path)
            points = np.stack([pcd_data['x'], pcd_data['y'], pcd_data['z'], pcd_data['intensity']], axis=1)

            # y, z 축 반전
            points[:, 1] *= -1  # y
            points[:, 2] *= -1  # z

            # 저장
            save_pcd_with_intensity(output_path, header, points)
            print(f"✅ Saved flipped PCD to {output_path}")

if __name__ == "__main__":
    main()
