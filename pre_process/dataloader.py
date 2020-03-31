#    Author:  a101269
#    Date  :  2020/3/4
import torch
import pathlib
from utils.utils import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class Dataloader(object):

    def __init__(self,args, processor):
        self.processor=processor
        self.args=args


    def load_data(self,file_path,batch_size,max_seq_len,mode='train'):  # mode'train','dev'
        abs_path, file_name = pathlib.Path(file_path).cwd(), pathlib.Path(file_path).name
        cashed_name ="cached_example_" + file_name.strip('sdp_')
        cached_examples_file = abs_path /self.args.cached_path / cashed_name
        if cached_examples_file.exists():
            conllu_file,_ = self.processor.read_data(file_path)
            examples = torch.load(cached_examples_file)
            logger.info('Load example from cached file.')
        else:
            conllu_file, conllu_data = self.processor.read_data(file_path)
            examples = self.processor.create_examples(conllu_data,mode,cached_examples_file)
            logger.info('Load example from raw file.')

        cashed_name = "cached_feature_" + file_name.strip('sdp_')
        cached_feature_file = abs_path / self.args.cached_path /cashed_name
        if cached_feature_file.exists():
            features = torch.load(cached_feature_file)
            logger.info('Load feature from cached file.')
        else:
            features = self.processor.create_features(examples,cached_feature_file,max_seq_len)
            logger.info('Load feature from raw file.')

        dataset=self.processor.create_dataset(features, is_sorted=False)

        if mode=='test' or mode=='dev':
            sampler = SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        logger.info('Dataset loaded.')
        return data_loader, conllu_file
