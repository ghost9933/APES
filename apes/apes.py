import re, os, glob
from apes.qa_system import utils, qa_module
import numpy as np
import logging
logger = logging.getLogger(__name__)
import argparse
import os, pickle, gzip

qa_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl.gz' )

def read_pickle(file):
    if '.gz' in file:
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

def read_file(pred_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        summary = ''.join(f.readlines())
    return summary

def entitize(summary, entities):
    entitized_summary = summary
    for ent_id, ent_name in sorted(entities.items(), key=lambda item: len(item[1]), reverse=True): 
        entitized_summary = re.sub(r'\b' + re.escape(ent_name) + r'\b', ent_id, entitized_summary, flags=re.IGNORECASE)

    return entitized_summary

def evaluate(prediction_filepattern, glove_path, questions_mapping_path, output_filename):
    questions_mapping = read_pickle(questions_mapping_path)
    scores = []
    total_correct, total_questions = 0, 0
    matched_summary = 0
    total_files = 0
    params, word_dict, entity_dict, _, test_fn, _ = qa_module.load_model(embedding_file=glove_path,
                                                                                model_file=qa_model_path)
    for filename in glob.glob(prediction_filepattern):

        summary_id = os.path.splitext(os.path.basename(filename))[0]
        summary = read_file(filename)
        total_files += 1
        if summary_id not in questions_mapping:
            continue
        entitized_summary = entitize(summary, questions_mapping[summary_id]['mapping'])
        curr_questions, curr_answers = zip(*[(q['question'], q['answer']) for q in questions_mapping[summary_id]['questions'].values()])
        num_questions = len(curr_questions)
        num_correct = 0

        if '@' in entitized_summary:
            query = [[entitized_summary]*num_questions, curr_questions, curr_answers, []]
            dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(query, word_dict, entity_dict, params)
            all_dev = qa_module.gen_examples(dev_x1, dev_x2, dev_l, dev_y, 16)
            dev_acc = qa_module.eval_acc(test_fn, all_dev)
            acc = dev_acc / 100
            num_correct = acc * num_questions

        scores.append(num_correct/num_questions)
        total_correct += num_correct
        total_questions += num_questions
        matched_summary += 1

    with open(output_filename, 'w') as f:
        f.write('Summary found,Summary Matched,APES scores,Question Asked,accuracy per question\n')
        f.write('{},{},{:.4f},{},{:.4f}'.format(total_files, matched_summary, total_correct/total_questions, total_questions,np.mean(scores) ))
        # f.write('Summary Matched           : {}/{}\n'.format(matched_summary, total_files))
        # f.write('APES scores               : {:.4f}, {}/{}\n'.format(total_correct/total_questions, total_correct, total_questions))
        # f.write('Avg accuracy per question :{:.4f} ({:.4f})'.format(np.mean(scores), np.std(scores)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='APES : summary assesment using question answering')
    parser.add_argument('--glove_path', required=True, type=str)
    parser.add_argument('--prediction_filepattern', required=True, type=str)
    parser.add_argument('--questions_mapping_path', required=True, type=str)
    parser.add_argument('--output_filename', required=True, type=str)
    args = parser.parse_args()

    evaluate(args.prediction_filepattern, 
        args.glove_path, args.questions_mapping_path, args.output_filename)