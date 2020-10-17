# coding=utf-8

from transformers import BertTokenizer
import jieba
import re

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


class Tokenizer():
    def __init__(self, vocab_dir=None):
        if vocab_dir is not None:
            bert_tokenizer = BertTokenizer.from_pretrained(vocab_dir)
            self.cutter = bert_tokenizer.tokenize
            self.convert_tokens_to_ids = bert_tokenizer.convert_tokens_to_ids
        else:
            self.cutter =  jieba.lcut
            self.convert_tokens_to_ids = None

    def cut(self, content):
        seg_content = []
        sentences = cut_sent(content)
        for sent in sentences:
            words = self.cutter(sent)
            seg_content.extend(words)
        return seg_content

