#    Author:  a101269
#    Date  :  2020/3/12
from transformers import AdamW, get_linear_schedule_with_warmup


# from transformers import WarmupLinearSchedule
def get_optimizer(args, batch_num, model):
    t_total = batch_num * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = batch_num* args.warmup_prop
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=t_total)
    # lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    return optimizer, lr_scheduler
