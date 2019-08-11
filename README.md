# APES

This is a python package to quickly use APES from [Question Answering as an Automatic Evaluation Metric for News Article Summarization](https://www.aclweb.org/anthology/N19-1395), as a summary evaluation metrics. **This repo is currently only test in Python 3.6**

APES currently only support summary evalution from CNN/Dailymail dataset, as the underlying model only contains questions regarding to this dataset.

## How to Use

0. Install it using the command below:
    ```
    pip install git+https://github.com/theblackcat102/APES.git
    ```

1. Requirements:
    1. Download Glove 6B 100 dimension embeddings

        You need to download glove embeddings and place it in some place this package can access

    2. Download question_mapping pkl.gz 

        Google Drive : [Link](https://drive.google.com/file/d/11qqk4tUwoAATEMEoZOhNvwyQYb99lkYh/view?usp=sharing), size 221 MB

        Or you can generate your own, for more details please refer to preprocessing section ( which requires the full CNN/Dailymail datasets)    

2. Data Format:

    Your summary should be placed inside a folder with **each name of the file matched to the CNN/Dailymail dataset story id.**

    ```
    test/
        469c6ac05092ca5997728c9dfc19f9ab6b936e40.pred
        ca1c3b587b7216654c8c719a66738de80d495179.pred
        <CNN/Dailymail story id>.pred
        ...
    ```

3. You can now evaluate your summary score by running this package:

    Example:
    ```
    python -m apes.apes --prediction_filepattern=./test/*.pred \
        --output_filename=apes_score.csv
        --glove_path=/nlp/data/glove/glove.6B.100d.txt \ 
        --questions_mapping_path=/nlp/module/questions_data.pkl.gz  
    ```


## Preprocessing : create article to entity mapping

First, run:

`python create_questions_mapping.py --cnn_questions_path path/to/cnn.tgz --dm_questions_path path/to/cnn.tgz`

This script creates a a pickle with a mapping from an article hash to its respective entities mapping (entity name to entity number), and questions.

# Execute

Then run both APES and ROUGE on your generated summaries. An example of a small subset of the CNN\Daily Mail:

`python run.py --preds_file testdata/test.txt.pred --targets_file testdata/test.txt.tgt --questions_mapping_path ./questions_data.pkl.gz --filenames_path ./filenames/filenames-test.txt`

```
APES score (5/29) : 0.1724
```

`python run.py --preds_file testdata/test.txt.tgt --targets_file testdata/test.txt.tgt --questions_mapping_path ./questions_data.pkl.gz --filenames_path ./filenames/filenames-test.txt`

```
APES score (11/29) : 0.3793
```

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
