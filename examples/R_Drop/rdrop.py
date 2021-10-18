# Copyright (c) danqiao5@gmail.com
import logging
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.utils import safe_hasattr
import torch
logger = logging.getLogger(__name__)

@register_task("rdrop_multilingual_translation")
class MultilingualTranslationTaskLatentDepth(MultilingualTranslationTask):
    """A task for multiple translation with r-drop. """
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        parser.add_argument('--reg-alpha', default=0, type=int,
                            help='settings for the weight of kl_loss in total loss')

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.criterion_reg_alpha = getattr(args, 'reg_alpha', 0)
    
    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
    
        langs_need_rdrop = ["eng-slk","eng-tur","eng-rus","eng-pos","eng-ces"]
        if lang_pair in langs_need_rdrop: 
            model.train()
            model.set_num_updates(update_num)
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion.forward_reg( model.models[lang_pair], sample[lang_pair], optimizer, self.criterion_reg_alpha, ignore_grad)
                return loss, sample_size, logging_output
        else:
            model.train()
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            return loss, sample_size, logging_output


    