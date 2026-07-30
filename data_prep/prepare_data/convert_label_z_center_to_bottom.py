import argparse
import glob
import os

'''
generate_superb_label.py converts SuperbAI's raw center-z labels to
bottom-center-z, which is the convention bbox3d2corners (and therefore
Avikus.__getitem__'s gt_bboxes_3d for training) expects. Raw captures that
arrive with label/*.txt already populated (no label_original/, meta/*.json,
or tracklet_labels.xml to trigger generate_superb_label.py/
generate_cvat_label.py) skip that conversion entirely, so their z is still
center-z. This does the same center -> bottom conversion for those.

Label line format: name truncated occluded alpha bbox(4) dimensions(3)
location(3) rotation_y [trailing_id]. dimensions/location are fields 8:11
and 11:14 (0-indexed).
'''


def convert_dataset_dir(dataset_dir):
    label_paths = sorted(glob.glob(os.path.join(dataset_dir, 'label', '*.txt')))
    n_converted = 0
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        new_lines = []
        for line in lines:
            fields = line.split(' ')
            height = float(fields[10])
            z_center = float(fields[13])
            fields[13] = f'{z_center - height / 2:.2f}'
            new_lines.append(' '.join(fields))

        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))
        n_converted += 1

    print(f'✅ Converted z center -> bottom for {n_converted} label file(s) in {dataset_dir}/label')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert label z from center to bottom-center (raw-capture sessions only)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='session dir, e.g. data_root/batch04')
    args = parser.parse_args()
    convert_dataset_dir(args.dataset_dir)
