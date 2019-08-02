import qa_module
import utils
import os
import pickle
from time import sleep
import argparse

def eval_acc(data):
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(data, word_dict, entity_dict, args)
    all_dev = qa_model.gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc = qa_model.eval_acc(test_fn, all_dev)
    return dev_acc

def read_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.loads(f.read())
	return data

if __name__ == "__main__":
    print("*****************************Started answering to questions****************************")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, required=True,
                        help='required path to rc train')
    parser.add_argument('--dev_path', type=str, required=True,
                        help='required path to rc train')
    parser.add_argument('--queries_folder', type=str, required=True,
                        help='Where to read and write queries.pkl and rewards.txt')
    parser.add_argument('--glove_path', type=str, required=True,
                        help='path to embeddings as required in the original paper. APES uses the 100 dim vectors.')
    parser.add_argument('--trained_model_path', type=str, default='./model.pkl.gz',
                        help='Path to trained model')

    args = parser.parse_args()

    query_path = args.queries_folder + '/queries.pkl'
    rewards_path = args.queries_folder + '/rewards.txt'

    print('query_path: ' + query_path)
    print('rewards_path: ' + rewards_path)

    args, word_dict, entity_dict, _, test_fn, params = qa_model.load_model(embedding_file=args.glove_path,
                                                                                model_file=args.trained_model_path)


    while(True):
        while(not os.path.isfile(query_path)):
            sleep(0.2)
        data = read_pickle(query_path)
        reward = eval_acc(data[:-1])
        os.remove(query_path)
        rewards_file = open(rewards_path, 'w')
        rewards_file.write(str(reward))
        rewards_file.close()
        # except Exception:
        #     print(query_path)
        #     print(data)
        #     print(reward)
        #     print(str(reward))

