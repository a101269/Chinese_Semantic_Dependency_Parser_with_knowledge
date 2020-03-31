#    Author:  a101269
#    Date  :  2020/3/4

import os
import torch
from argparse import ArgumentParser
from config.config import configs
from pre_process.vocab import load_vocab
from pre_process.con_process_know_bert.conll_processor import ConllProcessor
from pre_process.dataloader import Dataloader
from utils.utils import init_logger,logger,device,seed_everything
from model.trainer import Trainer
from model.parser_model import Parser_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--do_pre', action='store_true')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')  # O1混合精度训练
    args = parser.parse_args()
    for k, v in configs.items():
        setattr(args, k, v)
    args.device = device(args.use_cuda)

    if not args.do_test:
        args.gold_file = args.dev_file

    seed_everything(seed=args.seed)
    init_logger(log_file='myparser.log')
    logger.info(args)

    vocabs = load_vocab(args)
    processor = ConllProcessor(args,vocabs)
    dataloader= Dataloader(args, processor)

    model = Parser_model(args, vocabs)

    def test():
        model.load_state_dict(torch.load(args.saved_model_path + '/pytorch_model.bin'))
        logger.info('Start testing-----------------')
        for test_file in [args.test_file1, args.test_file2]:
            args.gold_file = test_file
            test_dataloader, test_conllu_file = dataloader.load_data(test_file, args.batch_size, args.max_seq_len,
                                                                     mode='test')
            batch_num = len(test_dataloader)
            trainer = Trainer(args, model, batch_num)
            LAS, UAS = trainer.predict(test_dataloader, test_conllu_file)
            logger.warning(f"Test result in {test_file}: LAS:{LAS:.4f}, UAS:{UAS:.4f}")
            reference_file = args.gold_file + '.sem16.sdp'
            os.system('python evalute.py --reference ' + reference_file)

    if args.do_test:
        test()
     
    else:
        train_dataloader, _ = dataloader.load_data(args.train_file,args.batch_size,args.max_seq_len, mode='train')
        dev_dataloader, dev_conllu_file = dataloader.load_data(args.dev_file,args.batch_size,args.max_seq_len, mode='dev')
        batch_num = len(train_dataloader)
        logger.info('Start training-----------------')
        trainer=Trainer(args,model,batch_num)
        trainer.train(train_dataloader,dev_dataloader,dev_conllu_file)
        test()

if __name__ == '__main__':

    main()