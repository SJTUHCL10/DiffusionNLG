import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
from tqdm.auto import tqdm
from random import random
from nltk.translate.bleu_score import sentence_bleu


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).long()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BertForDiffusionAddSelfCondition(nn.Module, ModuleUtilsMixin):
    """
    Transformer model for diffusion.
    Self condition embeddings are added to the decoder input embeddings
    """
    def __init__(self,
                 config,
                 self_condition=False,
                 time_emb_type='sin',
                 conditional_gen=False,
                 encoder_config=None,
                 encoder_type='frozen',
                 encoder_name_or_path='bert-base-uncased',):

        super().__init__()
        self.config = config
        self.self_condition = self_condition
        self.conditional_gen = conditional_gen

        if conditional_gen:
            assert encoder_type in ['frozen', 'fine-tune', 'from-scratch']
            self.encoder_type = encoder_type

            if encoder_type == 'frozen':    # use pretrained transformer encoder with frozen parameters
                self.encoder = BertModel.from_pretrained(encoder_name_or_path).requires_grad_(False)
                self.encoder_config = self.encoder.config

            elif encoder_type == 'fine-tune':   # use pretrained transformer encoder and fine tune
                self.encoder = BertModel.from_pretrained(encoder_name_or_path)
                self.encoder_config = self.encoder.config

            elif encoder_type == 'from-scratch':    # train encoder from scratch
                if encoder_config is None:
                    encoder_config = config
                self.encoder_word_embeddings = nn.Embedding(encoder_config.vocab_size, encoder_config.hidden_size)
                self.encoder_position_embeddings = nn.Embedding(encoder_config.max_position_embeddings,
                                                                encoder_config.hidden_size)
                self.encoder_LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
                self.encoder_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
                self.register_buffer("encoder_position_ids",
                                     torch.arange(encoder_config.max_position_embeddings).expand((1, -1)))
                self.bert_encoder = BertEncoder(encoder_config)
                self.encoder_config = encoder_config

            else:
                raise NotImplementedError

            # input_transformer as decoder
            config.is_decoder = True
            config.add_cross_attention = True

            self.encoder_output_proj = nn.Linear(self.encoder_config.hidden_size, config.hidden_size)

        self.input_transformer = BertEncoder(config)

        self.input_proj = nn.Sequential(nn.Linear(config.word_embedding_dim, config.hidden_size),
                                        nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.time_emb_type = time_emb_type
        if time_emb_type == 'abs':  # absolute time embedding
            self.time_embeddings = nn.Embedding(config.T, config.hidden_size)
        elif time_emb_type == 'sin':    # sinusoidal time embedding:
            sinu_pos_emb = SinusoidalPosEmb(dim=config.hidden_size//4)
            self.time_embeddings = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(config.hidden_size//4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif time_emb_type == 'none':   # do not add time embedding
            pass
        else:
            raise NotImplementedError

        if self_condition:
            self.cond_proj = nn.Sequential(nn.Linear(config.word_embedding_dim, config.hidden_size),
                                           nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.Tanh(), nn.Linear(config.hidden_size, config.word_embedding_dim))

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x, timesteps, x_self_cond=None, attention_mask=None, src_ids=None, src_attention_mask=None):
        """
        forward
        :param x: input text embeddings [batch, seq, word_embedding_dim]
        :param timesteps: time steps [batch,]
        :param x_self_cond: previous predicted x_0 for self-conditioning [batch,seq,word_emb_dim]
        :param attention_mask: attention mask [batch, seq]
        :param src_ids: source input ids when conditional generation [batch, seq,]
        :param src_attention_mask: source attention mask when conditional generation [batch, seq,]
        :return: projecction of last hidden states of bert encoder [batch, seq, word_emb_dim]
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        if self.conditional_gen:

            assert (src_ids is not None) and (src_attention_mask is not None)

            if self.encoder_type in ['frozen', 'fine-tune']:
                encoder_hidden_states = self.encoder(input_ids=src_ids,
                                                     attention_mask=src_attention_mask).last_hidden_state

            else:
                encoder_input_embeddings = self.encoder_word_embeddings(src_ids)    # [batch, seq, encoder_hidden_size]
                position_ids = self.position_ids[:, :src_ids.shape[1]]
                encoder_position_embeddings = self.encoder_position_embeddings(position_ids)
                encoder_embeddings = encoder_input_embeddings + encoder_position_embeddings
                encoder_embeddings = self.encoder_LayerNorm(encoder_embeddings)
                encoder_embeddings = self.encoder_dropout(encoder_embeddings)
                encoder_extended_attention_mask = self.invert_attention_mask(src_attention_mask,)
                encoder_hidden_states = self.bert_encoder(hidden_states=encoder_embeddings,
                                                          attention_mask=encoder_extended_attention_mask
                                                          ).last_hidden_state

            encoder_hidden_states = self.encoder_output_proj(encoder_hidden_states)

        input_embeddings = self.input_proj(x)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        if self.time_emb_type == 'none':
            time_embeddings = torch.zeros_like(input_embeddings)
        else:
            time_embeddings = self.time_embeddings(timesteps).unsqueeze(1).expand(-1, seq_length, -1)

        embeddings = input_embeddings + position_embeddings + time_embeddings

        if self.self_condition and x_self_cond is not None:
            cond_embeddings = self.cond_proj(x_self_cond)
            embeddings += cond_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=x.device)

        extended_attention_mask = self.invert_attention_mask(attention_mask,)
        encoder_extended_attention_mask = self.invert_attention_mask(src_attention_mask, )

        if self.conditional_gen:
            encoder_outputs = self.input_transformer(hidden_states=embeddings,
                                                     attention_mask=extended_attention_mask,
                                                     encoder_hidden_states=encoder_hidden_states,
                                                     encoder_attention_mask=encoder_extended_attention_mask,
            ).last_hidden_state

        else:
            encoder_outputs = self.input_transformer(hidden_states=embeddings,
                                                     attention_mask=extended_attention_mask).last_hidden_state

        outputs = self.output_proj(encoder_outputs)

        return outputs


class BertForDiffusion(nn.Module, ModuleUtilsMixin):
    """
    Transformer model for diffusion.
    Self condition embeddings are concatenated with the decoder input embeddings
    """
    def __init__(self,
                 config,
                 self_condition=False,
                 time_emb_type='sin',
                 conditional_gen=False,
                 encoder_config=None,
                 encoder_type='frozen',
                 encoder_name_or_path='bert-base-uncased',):

        super().__init__()
        self.config = config
        self.self_condition = self_condition
        self.conditional_gen = conditional_gen

        if conditional_gen:
            assert encoder_type in ['frozen', 'fine-tune', 'from-scratch']
            self.encoder_type = encoder_type

            if encoder_type == 'frozen':    # use pretrained transformer encoder with frozen parameters
                self.encoder = BertModel.from_pretrained(encoder_name_or_path).requires_grad_(False)
                self.encoder_config = self.encoder.config

            elif encoder_type == 'fine-tune':   # use pretrained transformer encoder and fine tune
                self.encoder = BertModel.from_pretrained(encoder_name_or_path)
                self.encoder_config = self.encoder.config

            elif encoder_type == 'from-scratch':    # train encoder from scratch
                if encoder_config is None:
                    encoder_config = config
                self.encoder_word_embeddings = nn.Embedding(encoder_config.vocab_size, encoder_config.hidden_size)
                self.encoder_position_embeddings = nn.Embedding(encoder_config.max_position_embeddings,
                                                                encoder_config.hidden_size)
                self.encoder_LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
                self.encoder_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
                self.register_buffer("encoder_position_ids",
                                     torch.arange(encoder_config.max_position_embeddings).expand((1, -1)))
                self.bert_encoder = BertEncoder(encoder_config)
                self.encoder_config = encoder_config

            else:
                raise NotImplementedError

            # input_transformer as decoder
            config.is_decoder = True
            config.add_cross_attention = True

            self.encoder_output_proj = nn.Linear(self.encoder_config.hidden_size, config.hidden_size)

        self.input_transformer = BertEncoder(config)

        if self.self_condition:
            # concatenate word
            self.input_proj = nn.Sequential(nn.Linear(2 * config.word_embedding_dim, config.hidden_size),
                                        nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        else:
            self.input_proj = nn.Sequential(nn.Linear(config.word_embedding_dim, config.hidden_size),
                                        nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.time_emb_type = time_emb_type
        if time_emb_type == 'abs':  # absolute time embedding
            self.time_embeddings = nn.Embedding(config.T, config.hidden_size)
        elif time_emb_type == 'sin':    # sinusoidal time embedding:
            sinu_pos_emb = SinusoidalPosEmb(dim=config.hidden_size//4)
            self.time_embeddings = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(config.hidden_size//4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif time_emb_type == 'none':   # do not add time embedding
            pass
        else:
            raise NotImplementedError

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.Tanh(), nn.Linear(config.hidden_size, config.word_embedding_dim))

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, x, timesteps, x_self_cond=None, attention_mask=None, src_ids=None, src_attention_mask=None):
        """
        forward
        :param x: input text embeddings [batch, seq, word_embedding_dim]
        :param timesteps: time steps [batch,]
        :param x_self_cond: previous predicted x_0 for self-conditioning [batch, seq, word_emb_dim]
        :param attention_mask: attention mask [batch, seq]
        :param src_ids: source input ids when conditional generation [batch, seq,]
        :param src_attention_mask: source attention mask when conditional generation [batch, seq,]
        :return: projecction of last hidden states of bert encoder [batch, seq, word_emb_dim]
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        if self.conditional_gen:

            assert (src_ids is not None) and (src_attention_mask is not None)

            if self.encoder_type in ['frozen', 'fine-tune']:
                encoder_hidden_states = self.encoder(input_ids=src_ids,
                                                     attention_mask=src_attention_mask).last_hidden_state

            else:
                encoder_input_embeddings = self.encoder_word_embeddings(src_ids)    # [batch, seq, encoder_hidden_size]
                position_ids = self.position_ids[:, :src_ids.shape[1]]
                encoder_position_embeddings = self.encoder_position_embeddings(position_ids)
                encoder_embeddings = encoder_input_embeddings + encoder_position_embeddings
                encoder_embeddings = self.encoder_LayerNorm(encoder_embeddings)
                encoder_embeddings = self.encoder_dropout(encoder_embeddings)
                encoder_extended_attention_mask = self.invert_attention_mask(src_attention_mask,)
                encoder_hidden_states = self.bert_encoder(hidden_states=encoder_embeddings,
                                                          attention_mask=encoder_extended_attention_mask
                                                          ).last_hidden_state

            encoder_hidden_states = self.encoder_output_proj(encoder_hidden_states)

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim = -1)

        input_embeddings = self.input_proj(x)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        if self.time_emb_type == 'none':
            time_embeddings = torch.zeros_like(input_embeddings)
        else:
            time_embeddings = self.time_embeddings(timesteps).unsqueeze(1).expand(-1, seq_length, -1)

        embeddings = input_embeddings + position_embeddings + time_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=x.device)

        extended_attention_mask = self.invert_attention_mask(attention_mask,)
        encoder_extended_attention_mask = self.invert_attention_mask(src_attention_mask, )

        if self.conditional_gen:
            encoder_outputs = self.input_transformer(hidden_states=embeddings,
                                                     attention_mask=extended_attention_mask,
                                                     encoder_hidden_states=encoder_hidden_states,
                                                     encoder_attention_mask=encoder_extended_attention_mask,
            ).last_hidden_state

        else:
            encoder_outputs = self.input_transformer(hidden_states=embeddings,
                                                     attention_mask=extended_attention_mask).last_hidden_state

        outputs = self.output_proj(encoder_outputs)

        return outputs


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


"""class DiffusionLM(nn.Module):
    def __init__(self, config, betas, model=None, use_attention_mask=False, 
                    use_shared_weight=True, lm_head_bias=False, add_emb_noise=True):

        super().__init__()

        if not hasattr(config, 'T'):
            config.T = len(betas)

        if not hasattr(config, 'word_embedding_dim'):
            config.word_embedding_dim = config.hidden_size

        self.config = config
        self.loss_terms = ['mse', 'L_T', 'rounding']

        self.use_attention_mask = use_attention_mask
        self.add_emb_noise = add_emb_noise

        # p(x_0 | w)
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_embedding_dim)
        # p(w | x_0)
        self.lm_head = nn.Linear(config.word_embedding_dim, config.vocab_size, bias=lm_head_bias)
        # initialize with the same weights
        # with torch.no_grad():
        #    self.lm_head.weight.data = torch.clone(self.word_embeddings.weight.data)
        if use_shared_weight:
            with torch.no_grad():
                self.lm_head.weight = self.word_embeddings.weight

        # use transformer to predict x_0
        # x_0 = model(x_t, t)
        if model is None:
            self.model = BertForDiffusion(config)
        else:
            self.model = model

        self.register_buffer("betas", betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.T = config.T
        assert self.T == betas.shape[0]

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # alpha_{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]
        # posterior mean
        # \tilde{mu}_t = coeff1 * x_t + coeff2 * x_0
        self.register_buffer(
            'posterior_mean_coeff1', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coeff2', self.betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        # posterior variance
        # \tilde{beta}_t
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def forward(self, input_ids, attention_mask=None):

        loss_terms = {}
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        device = input_ids.device
        t = torch.randint(low=0, high=self.T, size=(bs,), device=device)  # random timesteps

        # EMB(w) i.e. mean of p(x_0 | w)
        # [bs, seq, word_emb_dim]
        emb_w = self.word_embeddings(input_ids)
        if not self.add_emb_noise:
            # x_0 = emb_w when sigma_0 = 0
            x_0 = emb_w
        else:
            std = extract(self.sqrt_one_minus_alphas_bar, torch.LongTensor([0]).to(device), emb_w.shape)
            x_0 = emb_w + std * torch.randn_like(emb_w)

        noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        # f(x_t, t) i.e. predicted x_0
        if self.use_attention_mask:
            model_output = self.model(x_t, timesteps=t, attention_mask=attention_mask)
        else:
            model_output = self.model(x_t, timesteps=t, attention_mask=None)
        assert model_output.shape == x_0.shape
        t0_mask = (t == 0)
        # rescaling weight of [PAD] token in mse loss
        # pad_rescale = attention_mask.unsqueeze(-1)*0.9+0.1     # [batch, seq, 1]
        L_t = F.mse_loss(model_output, x_0, reduction='none')
        # L_t = L_t * pad_rescale
        L_t = L_t.mean(dim=list(range(1, len(L_t.shape))))
        L_0 = F.mse_loss(model_output, emb_w, reduction='none')
        # L_0 = L_0 * pad_rescale
        L_0 = L_0.mean(dim=list(range(1, len(L_0.shape))))
        loss_terms["mse"] = torch.where(t0_mask, L_0, L_t)

        x_T_mean = extract(self.sqrt_alphas_bar, torch.LongTensor([self.T - 1] * bs).to(device), x_0.shape) * x_0
        L_T = (x_T_mean ** 2).mean(dim=list(range(1, len(x_T_mean.shape))))
        loss_terms["L_T"] = L_T

        logits = self.lm_head(x_0)  # [batch, seq, vocab]

        # rescaling weight of [PAD] token in rounding loss
        # ce_weight = torch.ones(self.config.vocab_size).to(device)
        # ce_weight[self.config.pad_token_id] = 0.1
        rounding_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1),
                                        reduction='none')
        loss_terms["rounding"] = rounding_loss.view(input_ids.shape).mean(dim=-1)

        return loss_terms

    def p_mean_variance(self, x_t, t, clamp, attention_mask=None):
        # mean and var of p(x_{t-1} | x_t)
        var = extract(self.posterior_var, t, x_t.shape)
        # use whole attention mask when sampling

        if self.use_attention_mask:
            predicted_x0 = self.model(x_t, t, attention_mask=attention_mask)
        else:
            predicted_x0 = self.model(x_t, t, attention_mask=None)

        # x0 [batch, seq, word_emb_dim]
        if clamp == 'none':
            clamped_x0 = predicted_x0
        elif clamp == 'cosine':
            normed_emb = self.word_embeddings.weight.data / \
                         torch.norm(self.word_embeddings.weight.data, dim=1, keepdim=True)
            sim = torch.matmul(predicted_x0, normed_emb.T)  # [batch, seq, vocab_size]
            clamped_x0 = self.word_embeddings(torch.argmax(sim, dim=-1))
        elif clamp == 'l2':
            raise NotImplementedError
        elif clamp == 'rounding':
            logits = self.lm_head(predicted_x0)  # [batch, seq, vocab_size]
            clamped_x0 = self.word_embeddings(torch.argmax(logits, dim=-1))
        else:
            raise NotImplementedError

        mean = (
                extract(self.posterior_mean_coeff1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coeff2, t, x_t.shape) * clamped_x0
        )

        return mean, var

    def sample(self, x_T, clamp='none', attention_mask=None, return_hidden_states=False, verbose=False):
        assert clamp in ['none', 'cosine', 'l2', 'rounding']

        x_t = x_T  # [batch, seq, word_emb_dim]
        bs = x_T.shape[0]

        if verbose:  # display progress bar
            progress_bar = tqdm(range(self.T))

        # return noisy images in hidden states
        hidden_states = []

        with torch.no_grad():
            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([bs, ], dtype=torch.long) * time_step
                mean, var = self.p_mean_variance(x_t, t, clamp, attention_mask)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
                hidden_states.append(x_t)

                if verbose:
                    progress_bar.update(1)

        logits = self.lm_head(x_t)

        if return_hidden_states:
            return logits, hidden_states
        else:
            return logits
"""


class DiffusionLM(nn.Module):
    def __init__(self,
                 config,
                 betas,
                 model=None,
                 emb_type='learned',
                 self_condition=True,
                 use_attention_mask=False,
                 add_emb_noise=False,
                 use_shared_weight=True,
                 lm_head_bias=False,
                 time_emb_type='sin',
                 ):
        """
        Difussion LM class with several types of embedding (learned, randn, bit, ...)
        :param config: config for BertEncoder
        :param betas: noise schedule
        :param model: transformer model
        :param emb_type: word embedding type
        :param self_condition: use self-condition
        :param use_attention_mask: input samples with attention mask
        :param add_emb_noise: add noise to p(x_0 | w)
        :param use_shared_weight: when use learned embedding, embedding and rounding can share weights of linear module
        :param lm_head_bias: when use learned embedding, rounding with nn.Linear(bias=True)
        :param time_emb_type: time embedding type (sin, abs, none, ...)
        """
        super().__init__()

        assert emb_type in ['learned', 'randn', 'bit']
        self.emb_type = emb_type
        print('using ' + emb_type + ' word embedding')

        if not hasattr(config, 'T'):
            config.T = len(betas)

        if not hasattr(config, 'word_embedding_dim'):
            if emb_type in ['learned', 'randn']:
                config.word_embedding_dim = config.hidden_size
                print('set word_embedding_dim to:', config.word_embedding_dim)
            elif emb_type == 'bit':
                config.word_embedding_dim = math.ceil(math.log2(config.vocab_size))
                print('set word_embedding_dim to:', config.word_embedding_dim)

        self.config = config

        if emb_type == 'learned':
            self.loss_terms = ['mse', 'L_T', 'rounding']
        else:
            self.loss_terms = ['mse', 'L_T']

        self.self_condition = self_condition
        self.use_attention_mask = use_attention_mask
        self.add_emb_noise = add_emb_noise

        if emb_type == 'learned':
            self.word_embeddings = nn.Embedding(config.vocab_size, config.word_embedding_dim)
            self.lm_head = nn.Linear(config.word_embedding_dim, config.vocab_size, bias=lm_head_bias)
            if use_shared_weight:
                with torch.no_grad():
                    self.lm_head.weight = self.word_embeddings.weight
        elif emb_type == 'randn':
            emb_matrix = torch.randn(size=(config.vocab_size, config.word_embedding_dim))
            normed_emb_matrix = emb_matrix / torch.norm(emb_matrix, dim=1, keepdim=True)
            # self.emb_matrix = normed_emb_matrix
            self.register_buffer('emb_matrix', normed_emb_matrix)
        elif emb_type == 'bit':
            pass
        else:
            raise NotImplementedError

        # use transformer to predict x_0
        # x_0 = model(x_t, t)
        if model is None:
            self.model = BertForDiffusion(config, self_condition=self_condition, time_emb_type=time_emb_type)
        else:
            self.model = model

        self.register_buffer("betas", betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.T = config.T
        assert self.T == betas.shape[0]

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # alpha_{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]
        # posterior mean
        # \tilde{mu}_t = coeff1 * x_t + coeff2 * x_0
        self.register_buffer(
            'posterior_mean_coeff1', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coeff2', self.betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        # posterior variance
        # \tilde{beta}_t
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def get_embedding(self, input_ids,):
        """
        :param input_ids: [batch, seq]
        :return: word embedding [batch, seq, word_emb_dim]
        """
        if self.emb_type == 'learned':
            return self.word_embeddings(input_ids)
        elif self.emb_type == 'randn':
            one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()   # [batch, seq, vocab]
            return torch.matmul(one_hot, self.emb_matrix)    # [batch, seq, word_emb_dim]
        elif self.emb_type == 'bit':
            bits = dec2bin(input_ids, bits=self.config.word_embedding_dim)
            return bits * 2 - 1   # scale to [-1, 1]
        else:
            raise NotImplementedError

    def rounding(self, text_emb):
        """
        :param text_emb: [batch, seq, word_emb_dim]
        :return: token ids [batch, seq, ]
        """
        if self.emb_type == 'learned':
            logits = self.lm_head(text_emb)
            ids = torch.argmax(logits, dim=-1)
        elif self.emb_type == 'randn':
            similarity = torch.matmul(text_emb, self.emb_matrix.T)  # [batch, seq, vocab]
            ids = torch.argmax(similarity, dim=-1)  # [batch, seq]
        elif self.emb_type == 'bit':
            ids = bin2dec((text_emb > 0).long(), bits=self.config.word_embedding_dim)
        else:
            raise NotImplementedError

        return ids

    def forward(self, input_ids, attention_mask=None):
        loss_terms = {}
        bs = input_ids.shape[0]
        # seq_length = input_ids.shape[1]
        device = input_ids.device
        t = torch.randint(low=0, high=self.T, size=(bs,), device=device)  # random timesteps

        # EMB(w) i.e. mean of p(x_0 | w)
        # [bs, seq, word_emb_dim]
        emb_w = self.get_embedding(input_ids)
        if self.add_emb_noise:
            std = extract(self.sqrt_one_minus_alphas_bar, torch.LongTensor([0]).to(device), emb_w.shape)
            x_0 = emb_w + std * torch.randn_like(emb_w)
        else:
            x_0 = emb_w

        # q(x_t | x_0)
        noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        if not self.use_attention_mask:
            attention_mask = None

        x_self_cond = None
        if self.self_condition:
            if random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model(x_t, timesteps=t, attention_mask=attention_mask)
                    x_self_cond = x_self_cond.detach()
            else:
                x_self_cond = torch.zeros_like(x_0)

        model_output = self.model(x_t, timesteps=t, x_self_cond=x_self_cond, attention_mask=attention_mask)

        assert model_output.shape == x_0.shape

        t0_mask = (t == 0)

        L_t = F.mse_loss(model_output, x_0, reduction='none')
        L_t = L_t.mean(dim=list(range(1, len(L_t.shape))))
        L_0 = F.mse_loss(model_output, emb_w, reduction='none')
        L_0 = L_0.mean(dim=list(range(1, len(L_0.shape))))
        loss_terms["mse"] = torch.where(t0_mask, L_0, L_t)

        x_T_mean = extract(self.sqrt_alphas_bar, torch.LongTensor([self.T - 1] * bs).to(device), x_0.shape) * x_0
        L_T = (x_T_mean ** 2).mean(dim=list(range(1, len(x_T_mean.shape))))
        loss_terms["L_T"] = L_T

        if self.emb_type == 'learned':
            logits = self.lm_head(x_0)  # [batch, seq, vocab]
            rounding_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1),
                                            reduction='none')
            loss_terms["rounding"] = rounding_loss.view(input_ids.shape).mean(dim=-1)

        return loss_terms

    def p_mean_variance(self, x_t, t, clamp, x_self_cond=None, attention_mask=None):
        # mean and var of p(x_{t-1} | x_t)
        var = extract(self.posterior_var, t, x_t.shape)
        # use whole attention mask when sampling

        if not self.use_attention_mask:
            attention_mask = None

        predicted_x0 = self.model(x_t, t, x_self_cond=x_self_cond, attention_mask=attention_mask)

        # x0: [batch, seq, word_emb_dim]
        if self.emb_type in ['learned', 'randn']:
            if clamp == 'none':
                clamped_x0 = predicted_x0
            elif clamp == 'cosine':
                raise NotImplementedError
            elif clamp == 'l2':
                raise NotImplementedError
            elif clamp == 'rounding':
                new_ids = self.rounding(predicted_x0)
                clamped_x0 = self.get_embedding(new_ids)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        mean = (
                extract(self.posterior_mean_coeff1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coeff2, t, x_t.shape) * clamped_x0
        )

        return mean, var, clamped_x0

    @torch.no_grad()
    def sample(self, x_T, clamp='none', attention_mask=None, return_hidden_states=False, verbose=True):
        assert clamp in ['none', 'cosine', 'l2', 'rounding']

        x_t = x_T  # [batch, seq, word_emb_dim]
        bs = x_T.shape[0]
        device = self.betas.device

        if verbose:  # display progress bar
            progress_bar = tqdm(range(self.T))

        # return noisy images in hidden states
        hidden_states = []

        if not self.use_attention_mask:
            attention_mask = None

        if self.self_condition:
            x_self_cond = torch.zeros_like(x_t).to(device)
        else:
            x_self_cond = None

        with torch.no_grad():
            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([bs, ], dtype=torch.long) * time_step
                mean, var, predicted_x_0 = self.p_mean_variance(x_t, t, clamp, x_self_cond, attention_mask)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
                hidden_states.append(x_t)
                x_self_cond = predicted_x_0

                if verbose:
                    progress_bar.update(1)

        if return_hidden_states:
            return x_t, hidden_states
        else:
            return x_t


class ConditionalDiffusionLM(nn.Module):
    def __init__(self,
                 config,
                 betas,
                 model=None,
                 emb_type='learned',
                 self_condition=True,
                 use_attention_mask=False,
                 add_emb_noise=False,
                 use_shared_weight=True,
                 lm_head_bias=False,
                 time_emb_type='sin',
                 conditional_gen=True,
                 encoder_config=None,
                 encoder_type='frozen',
                 encoder_name_or_path='bert-base-uncased',
                 ):
        """
        Difussion LM class with several types of embedding (learned, randn, bit, ...)
        :param config: config for BertEncoder
        :param betas: noise schedule
        :param model: transformer model
        :param emb_type: word embedding type
        :param self_condition: use self-condition
        :param use_attention_mask: input samples with attention mask
        :param add_emb_noise: add noise to p(x_0 | w)
        :param use_shared_weight: when use learned embedding, embedding and rounding can share weights of linear module
        :param lm_head_bias: when use learned embedding, rounding with nn.Linear(bias=True)
        :param time_emb_type: time embedding type (sin, abs, none, ...)
        :param conditional_gen: generate with text condition
        :param encoder_config: config for text encoder for conditional generation
        :param encoder_type: 'frozen' bert encoder, 'fine-tune' bert encoder, or train an encoder with config 'from-scratch'
        :param encoder_name_or_path: encoder name or path
        """
        super().__init__()

        assert emb_type in ['learned', 'randn', 'bit']
        self.emb_type = emb_type
        print('using ' + emb_type + ' word embedding')

        if not hasattr(config, 'T'):
            config.T = len(betas)

        if not hasattr(config, 'word_embedding_dim'):
            if emb_type in ['learned', 'randn']:
                config.word_embedding_dim = config.hidden_size
                print('set word_embedding_dim to:', config.word_embedding_dim)
            elif emb_type == 'bit':
                config.word_embedding_dim = math.ceil(math.log2(config.vocab_size))
                print('set word_embedding_dim to:', config.word_embedding_dim)

        self.config = config

        if emb_type == 'learned':
            self.loss_terms = ['mse', 'L_T', 'rounding']
        else:
            self.loss_terms = ['mse', 'L_T']

        self.self_condition = self_condition
        self.use_attention_mask = use_attention_mask
        self.add_emb_noise = add_emb_noise

        if emb_type == 'learned':
            self.word_embeddings = nn.Embedding(config.vocab_size, config.word_embedding_dim)
            self.lm_head = nn.Linear(config.word_embedding_dim, config.vocab_size, bias=lm_head_bias)
            if use_shared_weight:
                with torch.no_grad():
                    self.lm_head.weight = self.word_embeddings.weight
        elif emb_type == 'randn':
            emb_matrix = torch.randn(size=(config.vocab_size, config.word_embedding_dim))
            normed_emb_matrix = emb_matrix / torch.norm(emb_matrix, dim=1, keepdim=True)
            # self.emb_matrix = normed_emb_matrix
            self.register_buffer('emb_matrix', normed_emb_matrix)
        elif emb_type == 'bit':
            pass
        else:
            raise NotImplementedError

        # use transformer to predict x_0
        # x_0 = model(x_t, t)
        if model is None:
            self.model = BertForDiffusion(config,
                                          self_condition=self_condition,
                                          time_emb_type=time_emb_type,
                                          conditional_gen=conditional_gen,
                                          encoder_config=encoder_config,
                                          encoder_type=encoder_type,
                                          encoder_name_or_path=encoder_name_or_path,)
        else:
            self.model = model

        self.register_buffer("betas", betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.T = config.T
        assert self.T == betas.shape[0]

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # alpha_{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.)[:self.T]
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)

        # posterior mean
        # \tilde{mu}_t = coeff1 * x_t + coeff2 * x_0
        self.register_buffer(
            'posterior_mean_coeff1', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coeff2', self.betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        # posterior variance
        # \tilde{beta}_t
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def get_embedding(self, input_ids,):
        """
        :param input_ids: [batch, seq]
        :return: word embedding [batch, seq, word_emb_dim]
        """
        if self.emb_type == 'learned':
            return self.word_embeddings(input_ids)
        elif self.emb_type == 'randn':
            one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()   # [batch, seq, vocab]
            return torch.matmul(one_hot, self.emb_matrix)    # [batch, seq, word_emb_dim]
        elif self.emb_type == 'bit':
            bits = dec2bin(input_ids, bits=self.config.word_embedding_dim)
            return bits * 2 - 1   # scale to [-1, 1]
        else:
            raise NotImplementedError

    def rounding(self, text_emb):
        """
        :param text_emb: [batch, seq, word_emb_dim]
        :return: token ids [batch, seq, ]
        """
        if self.emb_type == 'learned':
            logits = self.lm_head(text_emb)
            ids = torch.argmax(logits, dim=-1)
        elif self.emb_type == 'randn':
            similarity = torch.matmul(text_emb, self.emb_matrix.T)  # [batch, seq, vocab]
            ids = torch.argmax(similarity, dim=-1)  # [batch, seq]
        elif self.emb_type == 'bit':
            ids = bin2dec((text_emb > 0).long(), bits=self.config.word_embedding_dim)
        else:
            raise NotImplementedError

        return ids

    def clamp(self, x_0, clamp='none'):
        # x0: [batch, seq, word_emb_dim]
        if self.emb_type in ['learned', 'randn']:
            if clamp == 'none':
                clamped_x0 = x_0
            elif clamp == 'cosine':
                raise NotImplementedError
            elif clamp == 'l2':
                raise NotImplementedError
            elif clamp == 'rounding':
                new_ids = self.rounding(x_0)
                clamped_x0 = self.get_embedding(new_ids)
            else:
                raise NotImplementedError

        elif self.emb_type in ['bit']:
            if clamp == 'none':
                clamped_x0 = x_0
            else:
                # all dim clamp to -1 or 1
                clamped_x0 = ((x_0 > 0) * 2 - 1).float()

        else:
            raise NotImplementedError

        return clamped_x0

    def forward(self, input_ids, attention_mask=None, src_ids=None, src_attention_mask=None):

        loss_terms = {}
        bs = input_ids.shape[0]
        # seq_length = input_ids.shape[1]
        device = input_ids.device
        t = torch.randint(low=0, high=self.T, size=(bs,), device=device)  # random timesteps

        # EMB(w) i.e. mean of p(x_0 | w)
        # [bs, seq, word_emb_dim]
        emb_w = self.get_embedding(input_ids)
        if self.add_emb_noise:
            std = extract(self.sqrt_one_minus_alphas_bar, torch.LongTensor([0]).to(device), emb_w.shape)
            x_0 = emb_w + std * torch.randn_like(emb_w)
        else:
            x_0 = emb_w

        # q(x_t | x_0)
        noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        if not self.use_attention_mask:
            attention_mask = None

        x_self_cond = None
        if self.self_condition:
            if random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model(x_t, timesteps=t, attention_mask=attention_mask,
                                             src_ids=src_ids, src_attention_mask=src_attention_mask)
                    x_self_cond = x_self_cond.detach()
            else:
                x_self_cond = torch.zeros_like(x_0)

        model_output = self.model(x_t, timesteps=t, x_self_cond=x_self_cond, attention_mask=attention_mask,
                                  src_ids=src_ids, src_attention_mask=src_attention_mask)

        assert model_output.shape == x_0.shape

        t0_mask = (t == 0)

        L_t = F.mse_loss(model_output, x_0, reduction='none')
        L_t = L_t.mean(dim=list(range(1, len(L_t.shape))))
        L_0 = F.mse_loss(model_output, emb_w, reduction='none')
        L_0 = L_0.mean(dim=list(range(1, len(L_0.shape))))
        loss_terms["mse"] = torch.where(t0_mask, L_0, L_t)

        x_T_mean = extract(self.sqrt_alphas_bar, torch.LongTensor([self.T - 1] * bs).to(device), x_0.shape) * x_0
        L_T = (x_T_mean ** 2).mean(dim=list(range(1, len(x_T_mean.shape))))
        loss_terms["L_T"] = L_T

        if self.emb_type == 'learned':
            logits = self.lm_head(x_0)  # [batch, seq, vocab]
            rounding_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1),
                                            reduction='none')
            loss_terms["rounding"] = rounding_loss.view(input_ids.shape).mean(dim=-1)

        return loss_terms

    def p_mean_variance(self, x_t, t, clamp, x_self_cond=None, attention_mask=None,
                        src_ids=None, src_attention_mask=None):
        # mean and var of p(x_{t-1} | x_t)
        var = extract(self.posterior_var, t, x_t.shape)
        # use whole attention mask when sampling

        if not self.use_attention_mask:
            attention_mask = None

        predicted_x0 = self.model(x_t, t, x_self_cond=x_self_cond, attention_mask=attention_mask,
                                  src_ids=src_ids, src_attention_mask=src_attention_mask)

        # x0: [batch, seq, word_emb_dim]
        clamped_x0 = self.clamp(x_0=predicted_x0, clamp=clamp)

        mean = (
                extract(self.posterior_mean_coeff1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coeff2, t, x_t.shape) * clamped_x0
        )

        return mean, var, clamped_x0

    @torch.no_grad()
    def sample(self, x_T, clamp='none', attention_mask=None,
               src_ids=None, src_attention_mask=None,
               return_hidden_states=False, verbose=True):
        assert clamp in ['none', 'cosine', 'l2', 'rounding']

        x_t = x_T  # [batch, seq, word_emb_dim]
        bs = x_T.shape[0]
        device = self.betas.device

        if verbose:  # display progress bar
            progress_bar = tqdm(range(self.T))

        # return noisy images/texts in hidden states
        hidden_states = []

        if not self.use_attention_mask:
            attention_mask = None

        if self.self_condition:
            x_self_cond = torch.zeros_like(x_t).to(device)
        else:
            x_self_cond = None

        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([bs, ], dtype=torch.long) * time_step
            mean, var, predicted_x_0 = self.p_mean_variance(x_t, t, clamp, x_self_cond,
                                                            attention_mask, src_ids, src_attention_mask)
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(var) * noise
            if return_hidden_states:
                hidden_states.append(x_t)

            x_self_cond = predicted_x_0 if self.self_condition else None

            if verbose:
                progress_bar.update(1)

        if return_hidden_states:
            return x_t, hidden_states
        else:
            return x_t

    @torch.no_grad()
    def ddim_sample(self,
                    x_T,
                    sampling_timesteps=None,
                    eta=0,
                    attention_mask=None,
                    src_ids=None,
                    src_attention_mask=None,
                    return_hidden_states=False,
                    verbose=True,):

        assert sampling_timesteps <= self.T

        if sampling_timesteps is None:
            sampling_timesteps = self.T

        times = torch.linspace(-1, self.T - 1, steps=sampling_timesteps + 1)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_t = x_T  # [batch, seq, word_emb_dim]
        bs = x_T.shape[0]
        device = self.betas.device

        if verbose:  # display progress bar
            progress_bar = tqdm(range(sampling_timesteps))

        # return noisy images/texts in hidden states
        hidden_states = []

        if not self.use_attention_mask:
            attention_mask = None

        if self.self_condition:
            x_self_cond = torch.zeros_like(x_t).to(device)
        else:
            x_self_cond = None

        for time, time_next in time_pairs:
            t = x_t.new_ones([bs, ], dtype=torch.long) * time

            predicted_x0 = self.model(x_t, t, x_self_cond=x_self_cond, attention_mask=attention_mask,
                                      src_ids=src_ids, src_attention_mask=src_attention_mask)
            predicted_noise = (x_t - extract(self.sqrt_alphas_bar, t, x_t.shape) * predicted_x0) / \
                              extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)

            if time_next < 0:
                x_t = predicted_x0
            else:
                alpha = self.alphas_bar[time]
                alpha_next = self.alphas_bar[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)

                x_t = predicted_x0 * alpha_next.sqrt() + \
                      c * predicted_noise + \
                      sigma * noise

            x_self_cond = predicted_x0 if self.self_condition else None

            if return_hidden_states:
                hidden_states.append(x_t)

            if verbose:
                progress_bar.update(1)

        if return_hidden_states:
            return x_t, hidden_states
        else:
            return x_t

    @torch.no_grad()
    def generate(
            self,
            dataset,
            rev_tokenizer,
            sampling_timesteps=None,
            eta=0,
            mbr=1,
            batch_size=128,
            verbose=True,
    ):
        """
        Conditional text generation on the given dataset
        :param dataset: given src text data
        :param rev_tokenizer: id2token dict
        :param sampling_timesteps: ddim sampling timesteps
        :param eta: ddim eta
        :param mbr: candidate size of Minimum Bayes Risk Decoding
        :param batch_size: batch size of dataloader
        :param verbose: display tqdm
        :return: generated text list
        """
        device = self.model.device
        dataloader = tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)) if verbose else DataLoader(dataset, batch_size=batch_size, shuffle=False)
        generated_questions = []

        for batch in dataloader:
            bs = batch['question1_input_ids'].shape[0]
            if mbr > 1:     # using MBR decoding
                batch_questions = []
                for cnt in range(mbr):
                    x_T = torch.randn(size=(bs,
                                            self.config.max_position_embeddings,
                                            self.config.word_embedding_dim))
                    final_hidden_state = self.ddim_sample(x_T.to(device),
                                                          sampling_timesteps=sampling_timesteps,
                                                          eta=eta,
                                                          src_ids=batch['question1_input_ids'].to(device),
                                                          src_attention_mask=batch['question1_attention_mask'].to(device),
                                                          return_hidden_states=False,
                                                          verbose=False
                                                          )
                    sampled_ids = self.rounding(final_hidden_state).cpu()
                    # questions = [[rev_tokenizer[token_id.item()] for token_id in sampled_id] for sampled_id in sampled_ids]
                    questions = [[rev_tokenizer.get(token_id.item(), '[UNK]') for token_id in sampled_id] for sampled_id in sampled_ids]
                    batch_questions.append([list(filter(lambda x: x not in ['[PAD]', '[START]', '[END]'], question)) for question in questions])
                # batch_questions [mbr, bs, question]
                for batch_ind in range(bs):
                    candidates = [one_generation[batch_ind] for one_generation in batch_questions]      # [mbr, question]
                    bleu_scores = torch.zeros(mbr)
                    for candidate_ind, candidate in enumerate(candidates):
                        for ref_ind, ref in enumerate(candidates):
                            if ref_ind != candidate_ind:
                                bleu_scores[candidate_ind] += sentence_bleu([ref], candidate)
                    select_ind = torch.argmax(bleu_scores).item()
                    generated_questions.append(batch_questions[select_ind][batch_ind])

            elif mbr == 1:
                x_T = torch.randn(size=(bs,
                                        self.config.max_position_embeddings,
                                        self.config.word_embedding_dim))
                final_hidden_state = self.ddim_sample(x_T.to(device),
                                                      sampling_timesteps=sampling_timesteps,
                                                      eta=eta,
                                                      src_ids=batch['question1_input_ids'].to(device),
                                                      src_attention_mask=batch['question1_attention_mask'].to(device),
                                                      return_hidden_states=False,
                                                      verbose=False
                                                      )
                sampled_ids = self.rounding(final_hidden_state).cpu()
                # questions = [[rev_tokenizer[token_id.item()] for token_id in sampled_id] for sampled_id in sampled_ids]
                questions = [[rev_tokenizer.get(token_id.item(), '[UNK]') for token_id in sampled_id] for sampled_id in sampled_ids]
                generated_questions += [list(filter(lambda x: x not in ['[PAD]', '[START]', '[END]'], question)) for question in questions]

        return generated_questions

