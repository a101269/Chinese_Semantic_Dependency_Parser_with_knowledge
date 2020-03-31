#    Author':  a101269
#    Date  ':  2020/3/4

configs={
  'use_cuda': True,
  'seed': 123,
  'epochs': 80,
  'eval_interval':500,
  'early_stop': False,
  'early_stop_epochs': 10,
  'batch_size': 14,
  'columns':[0,1,3,8,9],   # 5，9misc杂项可放知识库信息
  # {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8,'misc': 9}

  'learning_rate': 3e-5,
  'weight_decay': 3.0e-9,  # 0.01,
  'beta1': 0.9,
  'beta2': 0.99,
  # 'eps': 1.0e-12,
  'adam_epsilon': 1e-8,
  'warmup_prop': 8,  # 单位 epoch
  'grad_clip_max_norm': 5.0,

  'max_seq_len': 250,  # 最长句子不会超过他，目前不用关注
  'bert_dim': 768,
  'do_lower_case': True,
  'bert_dropout': 0.1,
  'pretrain_model_path': '../bert_chn',

  'biaffine_hidden_dim': 600,
  'biaffine_dropout': 0.33,

  'use_pos':True,
  'use_knowledge':True,
  'bert_trans_dim':512,
  'pos_dim':256,
  'knowledge_dim':128,

  'use_transformer':True,
  'transformer_layer':2,
  'transformer_dropout':0.3,


  'dataset_path': 'dataset/',
  'cached_path':'dataset/cached',
  'train_file': 'dataset/sdp_mix_train.conllu_ner',
  'dev_file':  'dataset/sdp_text_dev.conllu_ner',
  'test_file1': 'dataset/sdp_text_test.conllu_ner',

  #'dev_file': 'dataset/sdp_news_dev.conllu_ner',
  'test_file2': 'dataset/sdp_news_test.conllu_ner',

  'eval_temp_file':'dataset/eval_temp.conllu',
  'debug_file': 'dataset/test.conllu_ner',

  'relation_vocab_path':'dataset/vocabs/rel.vocab',
  'logger_name': 'mylog',
  'output_path': 'output',
  'saved_model_path':'output/saved_model',

  # parser.add_argument("--warmup_proportion", default=0.1, type=int, )
  #   parser.add_argument("--weight_decay", default=0.01, type=float)
  # parser.add_argument("--adam_epsilon", default=1e-8, type=float)
  #   parser.add_argument("--grad_clip", default=1.0, type=float)
  # parser.add_argument("--learning_rate", default=2e-5, type=float)
}


