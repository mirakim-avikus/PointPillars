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
        wlh = [0, 0, 0]
        for dct in data[cls]:
            lwh += dct['box3d_lidar'][3:6]
        lwh /= len(data[cls])
        # wlh for anchor
        wlh[0] = lwh[1]
        wlh[1] = lwh[0]
        wlh[2] = lwh[2]
    
        print(f'{cls} : {round(wlh[0], 2)}, {round(wlh[1], 2)}, {round(wlh[2], 2)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="anchor setting script")
    parser.add_argument("--data_root", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    main(args)