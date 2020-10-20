# coding=utf-8

from utils.tokenizer import Tokenizer
from .generator import TrainDataGenerator, DevTestGenerator, ToBertInput


class QaDataset2():
    def __init__(self, vocab_dir, args):

        self.tokenizer = Tokenizer(vocab_dir=vocab_dir)


        callback = ToBertInput(args=args, tokenizer=self.tokenizer)

        self.train_generator = TrainDataGenerator(self, dataset_name='train',
                                                  tokenizer=self.tokenizer,
                                                  args=args,transform=[callback])

        self.test_generator = DevTestGenerator(self, dataset_name='test',
                                               tokenizer=self.tokenizer,
                                               args=args, transform=[callback])

        self.dev_generator = DevTestGenerator(self, dataset_name='dev',
                                              tokenizer=self.tokenizer,
                                              args=args, transform=[callback])


