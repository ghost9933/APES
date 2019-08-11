import os
from utils import create_data
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess corpus dataset')
    parser.add_argument('--folder_path', type=str, required=True,
                        help='required path to questions')
    parser.add_argument('--output', type=str, required=True,
                        help='data output filename')
    args = parser.parse_args()

    for data_type in ['training', 'test']:
        create_data(os.path.join(args.folder_path, 'questions/'+data_type), data_type+'_'+args.output)