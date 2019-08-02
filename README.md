# APES

This repository is an implementation of the summaries evaluation metric presented in [Question Answering as an Automatic Evaluation Metric for News Article Summarization](https://www.aclweb.org/anthology/N19-1395).
~~A Trained QA model on which we evaluate our generated summaries can be found in [here](https://github.com/mataney/rc-cnn-dailymail). (Notice these have different python versions so different environments are required) ~~

This repo include the trained QA model which save the time to setup and run two different Python versions.

## Quick Setup

1. Download Glove 6B 100 dimension embeddings

    You need to download glove embeddings and change run.py at line 15 '/media/backup1/glove/glove.6B.100d.txt' to your glove embedding path

2. Download question mapping pkl.gz and place into this repo (where run.py is placed)

    Google Drive : [Link](https://drive.google.com/file/d/11qqk4tUwoAATEMEoZOhNvwyQYb99lkYh/view?usp=sharing), size 221 MB

    Or you can generate your own, for more details please refer to preprocessing section ( which requires the full CNN/Dailymail datasets)

3. Setup requirements

  We will use virtualenv to create a Python3.6 environment and install the required dependencies
  
  ```
  virtualenv -p python3  env
  source env/bin/activate
  pip install -r requirements.txt
  ```

3. Test

  If everything works fine, you should be able to run the following command

  ```
  python run.py --preds_file testdata/test.txt.pred --targets_file testdata/test.txt.tgt --questions_mapping_path ./  questions_data.pkl --filenames_path ./filenames/filenames-test.txt
  ```

## Preprocessing : create article to entity mapping

First, run:

`python create_questions_mapping.py --cnn_questions_path path/to/cnn.tgz --dm_questions_path path/to/cnn.tgz`

This script creates a a pickle with a mapping from an article hash to its respective entities mapping (entity name to entity number), and questions.

# Execute

Then run both APES and ROUGE on your generated summaries. An example of a small subset of the CNN\Daily Mail:

`python run.py --preds_file testdata/test.txt.pred --targets_file testdata/test.txt.tgt --questions_mapping_path ./questions_data.pkl --filenames_path ./filenames/filenames-test.txt`

Filenames hold the mapping from CNN\Dail Mail article id to its hash.

# Citation
```
@inproceedings{eyal-etal-2019-question,
    title = "Question Answering as an Automatic Evaluation Metric for News Article Summarization",
    author = "Eyal, Matan  and
      Baumel, Tal  and
      Elhadad, Michael",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1395",
    pages = "3938--3948",
}
```
