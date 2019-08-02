import os, pickle, shutil, gzip

def summaries_to_rouge_format(summaries_file, outputs_dir, extention):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    with open(summaries_file, "r") as source: 
        lines = source.readlines()
        for i, line in enumerate(lines):
            with open(outputs_dir + str(i) + "." + extention, "w") as destination:
                destination.write(line)

def read_pickle(file):
    if '.gz' in file:
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data
        
def write_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def delete_dir(directory):
    shutil.rmtree(directory)