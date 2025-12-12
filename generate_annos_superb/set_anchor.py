import os
import pickle 
import argparse

def main(args):
    root_path = args.data_root
    with open(os.path.join(root_path, "avikus_dbinfos_train.pkl"), 'rb') as f :
        data = pickle.load(f)
    
    classes = data.keys()
    print("L W H")
    for cls in classes:
        lwh = [0, 0, 0]
        for dct in data[cls]:
            lwh += dct['box3d_lidar'][3:6]
        lwh /= len(data[cls])
        print(f'{cls} : {lwh}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="anchor setting script")
    parser.add_argument("--data_root", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    main(args)