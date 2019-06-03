"""
Find the "optimal" feature combination for helpfulness prediction.
- using a wrapper method called forward selection
- 3 scenarios: individual candidates, candidates in one category, all candidates

30 features candidates identified from survey papers:
- Semantics   (5): uni-grams, bi-grams, lda, w2v, g2v
- Sentiment   (7): gi, swn, liu, ss, liwc, galc, vader
- Readability (6): ari, cli, fre, fkg, fog, smog
- Structure   (7): char, word, sent, average length, !sent, ?sent, vader
- Syntax      (5): noun, verb, adj, adv, comparison
"""

import queue

from preparation.common_utils import TRAINING, VALIDATION, TESTING
from preparation.common_utils import get_domain_labels, make_dir
from traditional.classifiers import get_classifier, calc_aprfc
from scipy.sparse import hstack, coo_matrix
import time
import pandas as pd
import numpy as np
from preparation.logger_utils import get_logger
from selection.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


def gather_features(feat_extractor, feature_list):
    phases = [TRAINING, VALIDATION, TESTING]
    return dict(zip(
        phases,
        [
            hstack([
                coo_matrix(feat_extractor.get_feature(feature)[p])
                for feature in feature_list
            ])
            for p in phases
        ]
    ))


def get_metric_dict(valid_true_labels, valid_predicted_labels, test_true_labels=None, test_predicted_labels=None):
    metric_dict = dict(zip(
        ['val_' + x for x in ['a', 'p', 'r', 'f', 'c']],
        calc_aprfc(
            true_labels=valid_true_labels,
            predicted_labels=valid_predicted_labels
        )
    ))
    if test_true_labels is not None and test_predicted_labels is not None:
        metric_dict = {
            **metric_dict,
            **dict(zip(
                ['test_' + x for x in ['a', 'p', 'r', 'f', 'c']],
                calc_aprfc(
                    true_labels=test_true_labels,
                    predicted_labels=test_predicted_labels
                )
            ))
        }
    return metric_dict


if __name__ == '__main__':
    for domain_label in get_domain_labels('amazon'):
        out_dir = '../../previous/' + domain_label + '/'
        make_dir(out_dir)

        # this assumes feature extraction is done
        extractor = FeatureExtractor(domain=domain_label, need_extraction=False)

        results = list()

        # all_feat_names = feat_col.get_feature_names()
        # TODO: change [get_feat_names_by_cate]+itertool to obtain feature candidates from arbitrary categories
        for f_type in ['sem', 'sen', 'read', 'str', 'syn', 'all']:
            feat_candis = extractor.get_feat_names_by_cate(f_type)
            print('[%s] has %d features' % (f_type, len(feat_candis)))
            # for f in feat_candis:
            #     item = extractor.get_feature(f)[TRAINING][0]
            #     f_len = item.shape[0] if f in ['unigram', 'bigram'] else len(item)
            #     print('%s->%d' % (f, f_len), end='\t')
            # print()
            for clf_name in ['linearsvc']:
                for clf in get_classifier(clf_name, tuning=False):
                    que = queue.Queue(maxsize=len(feat_candis))
                    init = True
                    while que is not None:
                        print(que.qsize(), ' combinations in queue.')
                        cur_feat_comb, best_acc = ([], -np.inf) if init else que.get()
                        init = False
                        feat_left = set(feat_candis) - set(cur_feat_comb)

                        acc_list = list()
                        for feat_name in feat_left:
                            comb_to_try = cur_feat_comb + [feat_name]
                            # check if currently trying an individual feature
                            individual_flag = True if f_type == 'all' and len(comb_to_try) == 1 else False
                            X = gather_features(feat_extractor=extractor, feature_list=comb_to_try)
                            start = time.time()
                            clf.fit(X[TRAINING], extractor.labels[TRAINING])
                            end = time.time()
                            metrics = get_metric_dict(
                                valid_true_labels=extractor.labels[VALIDATION],
                                valid_predicted_labels=clf.predict(X[VALIDATION]),
                                test_true_labels=extractor.labels[TESTING] if individual_flag else None,
                                test_predicted_labels=clf.predict(X[TESTING]) if individual_flag else None
                            )
                            metrics['comb'] = '+'.join(comb_to_try)
                            metrics['clf'] = clf_name
                            metrics['time'] = end - start
                            metrics['type'] = f_type
                            acc_list.append(metrics['val_a'])

                            # output performance of individual features on testing sets
                            if individual_flag:
                                print(comb_to_try, metrics['val_a'])
                                metrics['type'] = 'single'
                                results.append(metrics)

                        # sort the accuracy of the current to-try feature combinations
                        #   acc_list[0] is the first comb
                        #   acc_list[0][1] is the val_acc of the first comb
                        acc_list = list(zip(feat_left, acc_list))
                        acc_list = sorted(acc_list, key=lambda x: x[1], reverse=True)
                        cur_best_acc = acc_list[0][1]

                        # check if forward selection ends
                        reach_end = True
                        if cur_best_acc >= best_acc:
                            # len(feat_left) == 1 means all features are selected
                            if len(feat_left) > 1:
                                # there are still feature candidates for testing
                                for feat_name, acc in acc_list:
                                    # if more than one largest accuracy
                                    # find the current best combination(s)
                                    if acc == cur_best_acc:
                                        print('put', str((cur_feat_comb + [feat_name], cur_best_acc)))
                                        que.put((cur_feat_comb + [feat_name], cur_best_acc))
                                        # search does not end
                                        reach_end = False
                                    else:
                                        break

                        # if none of the remaining features can help
                        # run all optimal combinations again to get performance data
                        if reach_end:
                            print('best', str(cur_feat_comb))
                            X_optimal = gather_features(feat_extractor=extractor, feature_list=cur_feat_comb)
                            start = time.time()
                            clf.fit(X_optimal[TRAINING], extractor.labels[TRAINING])
                            end = time.time()
                            metrics = get_metric_dict(
                                valid_true_labels=extractor.labels[VALIDATION],
                                valid_predicted_labels=clf.predict(X_optimal[VALIDATION]),
                                test_true_labels=extractor.labels[TESTING],
                                test_predicted_labels=clf.predict(X_optimal[TESTING])
                            )
                            metrics['comb'] = '+'.join(cur_feat_comb)
                            metrics['clf'] = clf_name
                            metrics['time'] = end - start
                            metrics['type'] = f_type
                            results.append(metrics)

                            # all combinations tested
                            if que.empty():
                                que = None
                            print('search ends.')

        results = pd.DataFrame(results)
        results.to_csv(out_dir + 'trad_results.csv', header=True, index=False)
