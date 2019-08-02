import re, os
from time import sleep
from qa_system import utils, qa_module

from utils import write_pickle, read_pickle

# def eval_acc(data, word_dict):
#     dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(data, word_dict, entity_dict, args)
#     all_dev = qa_module.gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
#     dev_acc = qa_module.eval_acc(test_fn, all_dev)
#     return dev_acc


def apes(preds_file, filenames_path, questions_mapping_path, glove_path, qa_model_path):

    total_correct, total_questions = 0, 0
    args, word_dict, entity_dict, _, test_fn, params = qa_module.load_model(embedding_file=glove_path,
                                                                                model_file=qa_model_path)
    questions_mapping, summaries, filenames = read_files(preds_file, filenames_path, questions_mapping_path)

    print('answer_questions')
    for i, (summary, art_hash) in enumerate(zip(summaries, filenames)):
        print("iter: " + str(i))
        entitized_summary = entitize(summary, questions_mapping[art_hash]['mapping'])
        curr_questions, curr_answers = zip(*[(q['question'], q['answer']) for q in questions_mapping[art_hash]['questions'].values()])
        num_questions = len(curr_questions)
        num_correct = 0

        if '@' in entitized_summary:
            query = [[entitized_summary]*num_questions, curr_questions, curr_answers, []]
            dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(query, word_dict, entity_dict, args)
            all_dev = qa_module.gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
            dev_acc = qa_module.eval_acc(test_fn, all_dev)
            acc = dev_acc/100
            num_correct = acc * num_questions

        print("NUM CORRECT: " + str(num_correct))
        print("NUM QUESTIONS: " + str(num_questions))
        total_correct += num_correct
        total_questions += num_questions

    print("TOTAL CORRECT: " + str(total_correct))
    print("TOTAL QUESTIONS: " + str(total_questions))

def read_files(preds_file, filenames_path, questions_mapping_path):
    questions_mapping = read_pickle(questions_mapping_path)
    filenames_path = filenames_path

    with open(preds_file, 'r', encoding='utf-8') as f:
        summaries = f.read().splitlines()

    with open(filenames_path, 'r', encoding='utf-8') as f:
        filenames = f.read().splitlines()

    return questions_mapping, summaries, filenames

def entitize(summary, entities):
    entitized_summary = summary
    for ent_id, ent_name in sorted(entities.items(), key=lambda item: len(item[1]), reverse=True): 
        entitized_summary = re.sub(r'\b' + re.escape(ent_name) + r'\b', ent_id, entitized_summary, flags=re.IGNORECASE)

    return entitized_summary

# def eval_acc(data):
#     query_path = './queries.pkl'
#     rewards_path = './rewards.txt'

#     write_pickle(query_path, data)
#     while(not os.path.isfile(rewards_path)):
#         sleep(0.1)

#     rewards_file = open(rewards_path, 'r')
#     acc = rewards_file.read()
#     os.remove(rewards_path)
#     try:
#         acc = float(acc)
#     except Exception:
#         acc = 0.0

#     return acc
