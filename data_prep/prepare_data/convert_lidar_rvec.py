import numpy as np
import cv2
import yaml
import argparse
import os

def convert_rvec(lidar_yaml_path, output_path=None):
    # 디렉토리 및 파일 경로 설정
    lidar_yaml_path = os.path.abspath(lidar_yaml_path)
    dir_path = os.path.dirname(lidar_yaml_path)
    if output_path is None:
        output_path = os.path.join(dir_path, 'lidar.yaml')
    else:
        output_path = os.path.abspath(output_path)

    # 1. YAML 파일 읽기
    with open(lidar_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # 2. 기존 camera2lidar rvec 추출
    rvec = np.array([
        data['camera2lidar']['rvec_1'],
        data['camera2lidar']['rvec_2'],
        data['camera2lidar']['rvec_3']
    ], dtype=np.float64)

    # 3. rvec → 회전 행렬
    R, _ = cv2.Rodrigues(rvec)

    # 4. 회전 행렬 곱셈
    R_flip = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0,  0, -1]
    ], dtype=np.float64)
    R_new = R @ R_flip

    # 5. 회전 행렬 → rvec
    rvec_new, _ = cv2.Rodrigues(R_new)

    # 6. 새로운 rvec으로 대체
    data['camera2lidar']['rvec_1'] = float(rvec_new[0][0])
    data['camera2lidar']['rvec_2'] = float(rvec_new[1][0])
    data['camera2lidar']['rvec_3'] = float(rvec_new[2][0])

    # 7. 기존 reference2lidar 도 똑같이 변환
    rvec = np.array([
        data['reference2lidar']['rvec_1'],
        data['reference2lidar']['rvec_2'],
        data['reference2lidar']['rvec_3']
    ], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    R_new = R @ R_flip
    rvec_new, _ = cv2.Rodrigues(R_new)
    data['reference2lidar']['rvec_1'] = float(rvec_new[0][0])
    data['reference2lidar']['rvec_2'] = float(rvec_new[1][0])
    data['reference2lidar']['rvec_3'] = float(rvec_new[2][0])

    # 8. 결과 저장
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ 변환 완료: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='camera2lidar rvec 변환 후 저장')
    parser.add_argument('--lidar_yaml_path', type=str, help='lidar.yaml의 경로')
    parser.add_argument('--output_path', type=str, default=None, help='결과 저장 경로 (기본값: 입력과 같은 폴더의 lidar.yaml)')
    args = parser.parse_args()
    convert_rvec(args.lidar_yaml_path, args.output_path)
