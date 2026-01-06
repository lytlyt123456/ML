import unicodedata
import numpy as np
from numpy.random import RandomState
import math
import config as cfg

class LanguageWordList:
    def __init__(self, language: str):
        self.language = language
        self.num_words = 2 # <SOS>, <EOS>
        self.word_to_index = {'<SOS>': cfg.SOS, '<EOS>': cfg.EOS}
        self.index_to_word = {cfg.SOS: '<SOS>', cfg.EOS: '<EOS>'}

    def add_word(self, word: str):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.index_to_word[self.num_words] = word
            self.num_words = self.num_words + 1

    def add_sentence(self, sentence: str):
        words = sentence.split(' ')
        for word in words:
            self.add_word(word)

def load_and_preprocess_data(data_path: str):
    with open(data_path, encoding='utf-8') as file:
        lines = file.readlines()
    language_pairs = [line.split('\t')[:2] for line in lines] # [[English, Chinese], [English, Chinese]]
    language_pairs = [list(reversed(pair)) for pair in language_pairs] # [[Chinese, English], [Chinese, English]]

    def normalize_english_string(s: str):
        # 去掉前后空格，大写转换为小写
        s = s.strip().lower()
        # 将s中的非ASCII字符转换为ASCII字符，如重音符号（cafe中e上有重音符号，带重音符号的e不属于ASCII范畴，需要将重音符号去掉）
        s = ''.join([c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'])
        # 处理缩写
        s = s.replace('can\'t', 'can not')
        s = s.replace('won\'t', 'will not')
        s = s.replace('n\'t', ' not')
        s = s.replace('\'m', ' am')
        s = s.replace('\'re', ' are')
        s = s.replace('\'', ' ')
        # 处理标点等其他符号，前后加上空格
        signals = [',', '.', '?', '!', ';', ':', '\'', '\"',
                   '@', '#', '$', '%', '^', '&', '*', '(', ')',
                   '-' ,'_', '+', '=', '{', '[', '}', ']', '|',
                   '\\', '/', '<', '>']
        for signal in signals:
           s = (s.replace(f'{signal} ', f'{signal}').replace(f' {signal}', f'{signal}')
                .replace(f'{signal}', f' {signal} '))
        s = s.strip()
        return s

    def normalize_chinese_string(s: str):
        s = ''.join([c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'])

        def is_ascii(c):
            return True if ord(c) < 128 else False

        s = s.strip()
        s_list = []
        ascii_str = ''
        pre_ascii = False
        for c in s:
            if not is_ascii(c):
                if pre_ascii:
                    s_list.append(ascii_str)
                pre_ascii = False
                s_list.append(c)
            elif c == ' ' or c == '\t' or c == '\r' or c == '\n':
                if pre_ascii:
                    s_list.append(ascii_str)
                pre_ascii = False
                continue
            else:
                if not pre_ascii:
                    ascii_str = c
                    pre_ascii = True
                else:
                    ascii_str += c
        if pre_ascii:
            s_list.append(ascii_str)

        s = ' '.join(s_list)
        return s

    language_pairs = [[normalize_chinese_string(pair[0]), normalize_english_string(pair[1])] for pair in language_pairs]

    english_prefixes = (
        "i am ", "i m ",
        "he is ", "he s ",
        "she is ", "she s ",
        "you are ", "you re ",
        "we are ", "we re ",
        "they are ", "they re "
    )

    language_pairs = [pair for pair in language_pairs if pair[1].startswith(english_prefixes)
                               and len(pair[0].split(' ')) <= cfg.MAX_SEQ_LENGTH
                               and len(pair[1].split(' ')) <= cfg.MAX_SEQ_LENGTH]

    chinese_word_list = LanguageWordList('Chinese')
    for pair in language_pairs:
        chinese_word_list.add_sentence(pair[0])

    english_word_list = LanguageWordList('English')
    for pair in language_pairs:
        english_word_list.add_sentence(pair[1])

    # 将中英翻译对中的中英文字符串转化为单词序号的序列，并在结尾加上<EOS> token
    lp_to_num_seq = []
    for pair in language_pairs:
        chinese_sentence = pair[0]
        english_sentence = pair[1]
        chinese_sentence = [chinese_word_list.word_to_index[word] for word in chinese_sentence.split(' ')]
        chinese_sentence.append(cfg.EOS)
        english_sentence = [english_word_list.word_to_index[word] for word in english_sentence.split(' ')]
        english_sentence.append(cfg.EOS)
        lp_to_num_seq.append([chinese_sentence, english_sentence])

    return chinese_word_list, english_word_list, language_pairs, lp_to_num_seq

def train_test_split(lp_to_num_seq, seed: int = 42, train_size: float = 0.6):
    # 将中英文翻译对打乱顺序
    n = len(lp_to_num_seq)
    indexes = np.arange(n)
    rand = RandomState(seed=seed)
    rand.shuffle(indexes)
    shuffled_dataset = [lp_to_num_seq[i] for i in indexes]

    # 划分训练集和测试集
    n_train = math.ceil(train_size * n)
    train_set = shuffled_dataset[:n_train]
    test_set = shuffled_dataset[n_train:]

    return train_set, test_set