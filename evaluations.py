import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

cresci2015_userdata_file_path = './data/inputs/cresci-2015/userdata.txt'
cresci2015_output_emb_file_path = './data/outputs/cresci-2015/cresci-2015.emb'
cresci2015_emb_model = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path)

id_label_dict = {}

with open(cresci2015_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        splits = line.split('\t')
        user_id = splits[0]
        label = splits[4].replace('\n', '')
        id_label_dict.update({user_id: int(label)})

X = []
y = []

for idx, key in enumerate(cresci2015_emb_model.wv.vocab):
    emb_vector = cresci2015_emb_model.wv[key]
    X.append(emb_vector)
    y.append(id_label_dict[key])

# initialize classifiers: SVM and LR (with L2 regularization)
svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X, y)
lr_classifier = LogisticRegression(random_state=0, penalty='l2')
lr_classifier.fit(X, y)

_10_folds_cross_val_scores_svm = cross_val_score(svm_classifier, X, y, cv=10)
_10_folds_cross_val_scores_lr = cross_val_score(lr_classifier, X, y, cv=10)

print('Evaluating accuracy performance of Bot2Vec model for bot classification task...')
print('Average 10-folds cross validation accuracy (SVM): {}'.format(np.mean(_10_folds_cross_val_scores_svm)))
print('Average 10-folds cross validation accuracy (LR-L2): {}'.format(np.mean(_10_folds_cross_val_scores_lr)))
