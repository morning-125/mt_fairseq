import torch
import logging
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


logger = logging.getLogger(__name__)

@register_task("rdrop_translation")
class RDropTranslation(TranslationTask):

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--reg-alpha', default=0, type=int)
        parser.add_argument('--encoder_embed_dim', default=512, type=int)
        parser.add_argument('--encoder_ffn_embed_dim', default=1024, type=int)
        parser.add_argument('--encoder_attention_heads', default=4, type=int)
        parser.add_argument('--encoder_layers', default=6, type=int)
        
        parser.add_argument('--decoder_embed_dim', default=512, type=int)
        parser.add_argument('--decoder_ffn_embed_dim', default=1024, type=int)
        parser.add_argument('--decoder_attention_heads', default=4, type=int)
        parser.add_argument('--decoder_layers', default=6, type=int)
        parser.add_argument('--dynamic',default=False,type=bool)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.criterion_reg_alpha = getattr(args, 'reg_alpha', 0)
        self.dynamic = getattr(args, 'dynamic', 0)

    def cal(self,update_num):
        return self.criterion_reg_alpha
        
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        if self.dynamic:
            self.criterion_reg_alpha = cal(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion.forward_reg(model, sample, optimizer, self.criterion_reg_alpha, ignore_grad)
            return loss, sample_size, logging_output