import argparse
import os

'''
Matches each pcd/*.avikus.pcd to its nearest-timestamp images/*.jpg and saves
the mapping to matching.txt (sibling to lidar.yaml in the session dir).
'''


def find_closest_img(images_dir, lidar_ts, max_diff=200):
    image_list = sorted(img for img in os.listdir(images_dir) if img.endswith('.jpg'))
    if not image_list:
        return None
    image_ts_list = [int(name.split('.')[0]) for name in image_list]
    diffs = [abs(ts - lidar_ts) for ts in image_ts_list]
    min_idx = min(range(len(diffs)), key=diffs.__getitem__)
    if diffs[min_idx] > max_diff:
        return None
    return image_list[min_idx]


def match_jpg_pcd(dataset_dir, max_diff=200):
    pcd_dir = os.path.join(dataset_dir, 'pcd')
    images_dir = os.path.join(dataset_dir, 'images')

    pcd_list = sorted(f for f in os.listdir(pcd_dir) if f.endswith('.avikus.pcd'))
    if not pcd_list:
        print(f'[matching_jpg_pcd.py] no pcd files under {pcd_dir}, skip')
        return

    lines = []
    n_matched, n_unmatched = 0, 0
    for pcd_name in pcd_list:
        pcd_ts = int(pcd_name.split('.')[0])
        img_name = find_closest_img(images_dir, pcd_ts, max_diff=max_diff)
        if img_name is None:
            n_unmatched += 1
            continue
        lines.append(f'{pcd_ts} {img_name}')
        n_matched += 1

    output_path = os.path.join(dataset_dir, 'matching.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

    print(f'[matching_jpg_pcd.py] {dataset_dir}: {n_matched} matched, {n_unmatched} unmatched (no image within {max_diff}ms) -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match each pcd/*.avikus.pcd to its nearest-timestamp images/*.jpg, save to matching.txt')
    parser.add_argument('--dataset_dir', required=True, help='session dir, e.g. data_root/batch04 (needs pcd/, images/)')
    parser.add_argument('--max_diff', type=int, default=200, help='max allowed timestamp gap in ms')
    args = parser.parse_args()
    match_jpg_pcd(args.dataset_dir, args.max_diff)
