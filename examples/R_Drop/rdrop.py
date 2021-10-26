# Copyright (c) danqiao5@gmail.com
import logging
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.utils import safe_hasattr
import torch
logger = logging.getLogger(__name__)

lang_dic={"eng-aze":0,"eng-bel":1,"eng-glg":2,"eng-slk":3,"eng-tur":4,"eng-rus":5,"eng-por":6,"eng-ces":7}

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
        self.alpha_map = {"eng-aze":0,"eng-bel":0,"eng-glg":0,"eng-slk":0,"eng-tur":0,"eng-rus":0,"eng-por":0,"eng-ces":0}
        # self.alpha_map = {"eng-aze":0,"eng-bel":0,"eng-glg":1,"eng-slk":2,"eng-tur":3,"eng-rus":3,"eng-por":3,"eng-ces":3}
    
    def set_dropout(self,model,dropout_rate):
        for layer in model.encoder.layers:
            layer.dropout_module.p = dropout_rate
            layer.self_attn.dropout_module.p = dropout_rate
        for layer in model.decoder.layers:
            layer.dropout_module.p = dropout_rate
            layer.self_attn.dropout_module.p = dropout_rate  

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        #self.set_dropout(model,self.drop_map[lang_pair])
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion.forward_reg( model.models[lang_pair], 
            sample[lang_pair], lang_pair[-3:], optimizer, self.alpha_map[lang_pair], ignore_grad)
            return loss, sample_size, logging_output



    