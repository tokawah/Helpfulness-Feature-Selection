from preparation.amazon_reader import DomainReader
from selection import sentiment_utils, semantics_utils, auxiliary_utils
from preparation.preprocessing import get_tokens
from preparation.common_utils import TRAINING, VALIDATION, TESTING
from preparation.common_utils import make_dir, get_domain_labels

import pandas as pd
import numpy as np
import pickle

from pathlib import Path

# from traditional.train_task import TaskTemplate

from preparation.logger_utils import get_logger
from preparation.preprocessing import punct_list


logger = get_logger(__name__)


def _normalize_scores(scores, method):
    assert method in ['percentage', 'z-score']
    scores = np.asarray(scores)

    if method == 'percentage':
        if scores.sum() == 0:
            return scores
        return scores / scores.sum()
    else:
        # if scores.std() == 0:
        #     return scores
        return (scores - scores.mean()) / scores.std()


class FeatureExtractor:

    def __init__(self, domain, need_extraction):
        self.domain_label = domain
        self.out_dir = '../../previous/' + domain + '/'
        make_dir(self.out_dir)

        # init a dataset
        dataset = DomainReader(domain=domain)
        word_vocab = dataset.get_word_vocab()
        self.texts = dataset.get_texts()
        self.phase_locators = dataset.get_phase_locators()
        self.labels = dataset.get_labels()
        self.ratings = dataset.get_ratings(mode='as_single_output', normalized=True)
        dataset = None

        # semantics unit only covers lda extraction
        # even features are extracted, sem_utils is still used to generate tfidf and average embeddings
        self.sem = semantics_utils.Semantics()

        if need_extraction:
            # get reviews where each is separated into sentences
            # with proper a proper tokenization step already
            # use str.split() equals to preparation.get_tokens()
            logger.info('preparing sentence boundaries...')
            txts = pd.read_csv('../../datasets/raw/' + domain + '.csv', usecols=['txt'])['txt']
            with open('../../datasets/preprocessed/' + domain + '.bounds', 'r') as f:
                bounds = f.readlines()
                bounds = [
                    [list(map(int, bs.split('-'))) for bs in bound_str.split('\t')]
                    for bound_str in bounds
                ]
            # df['bounds'] = bounds
            self.sentence_list = [
                [
                    ' '.join([
                        token for token in get_tokens(text[bound[0]: bound[1]].lower())
                        if token in word_vocab or token in punct_list
                    ])
                    for bound in bounds
                ]
                for text, bounds in zip(txts, bounds)  # (df['txt'], df['bounds'])
            ]
            # print(self.sentence_list[0])
            # df = None

            self.sen = sentiment_utils.Sentiment()
            self.aux = auxiliary_utils.Auxiliary()
            # print()

        sem_dict = self.extract_semantics()
        sen_dict = self.extract_sentiments()
        read_dict = self.extract_readability()
        str_dict = self.extract_structure()
        syn_dict = self.extract_syntax()

        # load pre-trained features
        self._extracted = {
            **sem_dict, **sen_dict, **read_dict, **str_dict, **syn_dict,
            'ss': sentiment_utils.load_sentistrength(
                domain, phase_locators=self.phase_locators, unary_output=False
            ),
            'liwc': sentiment_utils.load_liwc(domain, phase_locators=self.phase_locators)
        }

        # train features on the fly
        logger.info('train remaining semantic features on the fly...')
        phases = list(self.phase_locators.keys())
        for feat_name, n in [('unigram', 1), ('bigram', 2)]:
            tfidf = self.sem.get_word_tfidf_previous(
                self.texts[TRAINING], self.texts[VALIDATION], self.texts[TESTING], n=n
                # *self.texts, n=n
            )
            self._extracted[feat_name] = dict(zip(phases, tfidf))

        for feat_name, func in [
            ('wv', self.sem.get_sgns_vector),
            ('gv', self.sem.get_glove_vector)
        ]:
            self._extracted[feat_name] = dict(zip(
                phases, [[func(x) for x in self.texts[p]] for p in phases]
                # phases, [list(map(func, self.texts[p])) for p in phases]
            ))
            self.sem.clear_embedding_file()

        self._feat_names = {
            'sem': list(sem_dict.keys()) + ['unigram', 'bigram', 'wv', 'gv'],
            'sen': list(sen_dict.keys()) + ['ss', 'liwc'],
            'read': list(read_dict.keys()),
            'str': list(str_dict.keys()),
            'syn': list(syn_dict.keys())
            # feat_name: list(d.keys())
            # for feat_name, d in [
            #     ('sen', sen_dict), ('read', read_dict),
            #     ('str', str_dict), ('syn', syn_dict)
            # ]
        }

    def get_feature(self, feat_name):
        return self._extracted[feat_name]

    # def set_feature(self, feat_name, feature):
    #     self._extracted[feat_name] = feature

    def get_feat_names_by_cate(self, category):
        # if category == 'semantics':
        #     return ['unigram', 'bigram', 'lda', 'wv', 'gv']
        # elif category == 'sentiment':
        #     return ['vader', 'swn', 'gi', 'liu', 'galc', 'ss', 'liwc']
        # elif category == 'readability':
        #     return ['fre', 'smog', 'fkg', 'cli', 'ari', 'fog']
        # elif category == 'structure':
        #     return [
        #         'num_chars', 'num_tokens', 'num_sents', 'avg_sent_len',
        #         'num_interrogative_sents', 'num_exclamatory_sents', 'num_mis'
        #     ]
        # elif category == 'syntax':
        #     return ['num_nouns', 'num_verbs', 'num_adj', 'num_advs', 'num_comp_sents']
        if category == 'all':
            return list(self._extracted.keys())
        else:
            assert category in self._feat_names
            return self._feat_names[category]

    def split_by_phase(self, features):
        features = np.asarray(features)
        return {
            phase: features[locator]
            for phase, locator in self.phase_locators.items()
        }

    # ---------------------
    # Semantics
    # ---------------------

    def extract_semantics(self):
        logger.info('preparing semantic features...')

        # lda
        logger.info('preparing lda...')
        pretrained_lda = self.out_dir + self.domain_label+'.lda'
        pretrained_lda_model = self.out_dir + self.domain_label + '.lda_model'
        if Path(pretrained_lda).is_file():
            logger.info('loading pretrained lda...')
            with open(pretrained_lda, 'rb') as fp:
                lda = pickle.load(fp)
        else:
            logger.info('training lda...')
            lda_model, lda_training, lda_validation, lda_testing = self.sem.get_lda(
                *self.sem.get_word_tfidf_previous(
                    self.texts[TRAINING], self.texts[VALIDATION], self.texts[TESTING], n=1
                ),
                output_model=True
            )
            lda = {TRAINING: lda_training, VALIDATION: lda_validation, TESTING: lda_testing}
            with open(pretrained_lda, 'wb') as fp:
                pickle.dump(lda, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open(pretrained_lda_model, 'wb') as fp:
                pickle.dump(lda_model, fp, protocol=pickle.HIGHEST_PROTOCOL)
        lda = {'lda': lda}
        return lda

        # tfidf features are fast to obtain
        # sgns features are ready
        # glove features are ready

    # ---------------------
    # Sentiment
    # ---------------------

    # todo normalize features after loading
    def extract_sentiments(self):
        pretrained_senti = self.out_dir + self.domain_label + '.sentiment'
        if Path(pretrained_senti).is_file():
            logger.info('loading pretrained sentiment features...')
            with open(pretrained_senti, 'rb') as fp:
                batch_senti = pickle.load(fp)
        else:
            logger.info('preparing sentiment features...')
            batch_senti = self.sen.get_batch_sentiment_scores(
                sent_list=self.sentence_list,
                lexicons=['all']
            )
            with open(pretrained_senti, 'wb') as fp:
                pickle.dump(batch_senti, fp)

        # normalization and phase split
        for lexicon, score_list in batch_senti.items():
            if lexicon not in ['galc', 'gi']:
                score_list = [
                    [scores.count(-1), scores.count(0), scores.count(1)]
                    for scores in score_list
                ]
            # if lexicon == 'galc':
            #     print('****************', len(score_list[0]))
            score_list = [_normalize_scores(scores, method='percentage') for scores in score_list]
            batch_senti[lexicon] = self.split_by_phase(score_list)

        return batch_senti

        # liwc features are ready
        # ss features are ready

    # ---------------------
    # Readability
    # ---------------------

    def extract_readability(self):

        pretrained_read = self.out_dir + self.domain_label + '.readability'
        if Path(pretrained_read).is_file():
            logger.info('loading pretrained readability features...')
            with open(pretrained_read, 'rb') as fp:
                batch_read = pickle.load(fp)
        else:
            logger.info('preparing readability features...')
            batch_read = self.aux.get_batch_readability_scores(
                sent_list=self.sentence_list,
                # 'smog' is not available since it requires at least 30 sentences
                # amazon reviews usually have less than 30 sentences
                measures=['all']
            )
            with open(pretrained_read, 'wb') as fp:
                pickle.dump(batch_read, fp)

        # normalization and phase split
        # ['fre', 'fkg', 'cli', 'ari', 'fog']
        for measure, score_list in batch_read.items():
            score_list = [[score] for score in _normalize_scores(score_list, method='z-score')]
            batch_read[measure] = self.split_by_phase(score_list)

        return batch_read

    # ---------------------
    # Structure
    # ---------------------

    def extract_structure(self):
        pretrained_structure = self.out_dir + self.domain_label + '.structure'
        if Path(pretrained_structure).is_file():
            logger.info('loading pretrained structural features...')
            with open(pretrained_structure, 'rb') as fp:
                batch_structure = pickle.load(fp)
        else:
            logger.info('preparing structural features...')
            batch_structure = self.aux.get_batch_structure_scores(
                sent_list=self.sentence_list,
                measures=['all']
            )
            with open(pretrained_structure, 'wb') as fp:
                pickle.dump(batch_structure, fp)

        for measure, score_list in batch_structure.items():
            score_list = [[score] for score in _normalize_scores(score_list, method='z-score')]
            batch_structure[measure] = self.split_by_phase(score_list)

        return batch_structure

    # ---------------------
    # Syntax
    # ---------------------

    def extract_syntax(self):
        pretrained_syntax = self.out_dir + self.domain_label + '.syntax'
        if Path(pretrained_syntax).is_file():
            logger.info('loading pretrained syntactic features...')
            with open(pretrained_syntax, 'rb') as fp:
                batch_syntax = pickle.load(fp)
        else:
            logger.info('preparing syntactic features...')
            batch_syntax = self.aux.get_batch_syntax_scores(
                sent_list=self.sentence_list,
                measures=['all']
            )
            with open(pretrained_syntax, 'wb') as fp:
                pickle.dump(batch_syntax, fp)

        # normalization and phase split
        # ['num_passive_sents', 'num_sugg_sents', 'num_comp_sents'] +
        # ['num_mis_words', 'num_nouns', 'num_verbs', 'num_advs', 'num_adjs']
        for measure, score_list in batch_syntax.items():
            score_list = [[score] for score in _normalize_scores(score_list, method='z-score')]
            batch_syntax[measure] = self.split_by_phase(score_list)

        return batch_syntax


if __name__ == '__main__':

    for domain_label in get_domain_labels('amazon'):
        logger.info(domain_label)
        # todo: train new lda features
        # todo: cope with two features having the same val_acc
        feat_utils = FeatureExtractor(domain=domain_label, need_extraction=True)
        # # print(feat.get_all_feature_names())
        # # print(feat.get_texts()[TRAINING][1])
        # for feat_str in feat.get_all_feature_names():
        #     print(feat_str)
        #     print(feat.get_feature(feat_str)[TRAINING][1])
        #
        # print(len(feat.get_all_feature_names()))

