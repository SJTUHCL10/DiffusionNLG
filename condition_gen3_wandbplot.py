import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
import time
import pickle
import wandb
import argparse

from transformers import BertConfig, BertTokenizerFast
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from models import BertForDiffusion, DiffusionLM, ConditionalDiffusionLM
from data_utils import load_qqp_dataset_and_tokenizer_from_disk, QQPParaphraseDataset, load_split_qqp_dataset_and_tokenizer_from_disk
from noise_schedule import get_named_beta_schedule
from train_utils import train_conditional, evaluate_conditional
from metric_utils import calculate_bleu, calculate_rouge

parser = argparse.ArgumentParser()
parser.add_argument("--word_emb_dim", type=int, default=512)

args = parser.parse_args()

wandb.init(
    project='chapter3_emb_collapse',
    config=vars(args),
)

# dataset args
max_len = 32

# training args
batch_size = 64
device = torch.device("cuda:3")
lr = 1e-4
num_epoch = 30
weight_decay = 0
num_warmup_steps = 100

# model args
word_embedding_dim = args.word_emb_dim
# hidden_size = 768
# num_hidden_layers = 12
# num_attention_heads = 12
# intermediate_size = 3072
hidden_size = 512
num_hidden_layers = 4
num_attention_heads = 8
intermediate_size = 2048
max_position_embeddings = max_len

encoder_type = 'from-scratch'

train_dataset, eval_dataset, tokenizer = load_split_qqp_dataset_and_tokenizer_from_disk(data_path="data")

# tokenized_qqp_train, tokenized_qqp_eval, tokenizer = load_qqp_dataset_and_tokenizer_from_disk(data_path="data")

rev_tokenizer = {v: k for k, v in tokenizer.items()}

print("Tokenizer vocab size:", len(tokenizer))

# train_dataset = QQPParaphraseDataset(dataset=tokenized_qqp_train, random_swap=True)
print("Training set size:", len(train_dataset))
# eval_dataset = QQPParaphraseDataset(dataset=tokenized_qqp_eval, random_swap=False)
print("Evaluation set size:", len(eval_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

config = BertConfig(vocab_size=len(tokenizer), hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, max_position_embeddings=max_position_embeddings, pad_token_id=tokenizer['[PAD]'])

config.T = 2000
# comment next line if using bit word embedding
config.word_embedding_dim = word_embedding_dim

print(config)

betas = torch.Tensor(get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=config.T))


diffusion_model = ConditionalDiffusionLM(
    config=config, 
    betas=betas, 
    use_shared_weight=False, 
    lm_head_bias=True, 
    add_emb_noise=False, 
    conditional_gen=True, 
    encoder_type=encoder_type, 
    encoder_name_or_path='bert-base-uncased', 
    emb_type='learned',
).to(device)

print("Diffusion model #parameters:")
print(sum([p.numel() for p in diffusion_model.parameters()]))

print("Diffusion model #trainable parameters")
print(sum([p.numel() for p in filter(lambda p:p.requires_grad, diffusion_model.parameters())]))


optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, diffusion_model.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_epoch*len(train_dataloader))

# train loop
# loss_terms_dict_lst = []
loss_terms_dict = defaultdict(list)
loss_terms_weights = None
verbose = True
print_steps=100
progress_bar = tqdm(range(num_epoch*len(train_dataloader)))

cnter = Counter()
for d in train_dataset:
    cnter.update(d['question1_input_ids'].tolist())
    cnter.update(d['question2_input_ids'].tolist())
id_gt10 = [id for id in cnter if cnter[id] >= 100 and id > 3]

trained_ts = diffusion_model.word_embeddings.weight.data.cpu()
embs = [trained_ts[id] for id in id_gt10]
N = len(embs)
trained_dist = 0
for i in tqdm(range(N)):
    for j in range(i+1, N):
        trained_dist += (embs[i] - embs[j]).norm()
trained_dist = trained_dist / (N * (N-1) / 2)
wandb.log({'Emb avg dist': trained_dist})

for epoch in range(num_epoch):
    print("epoch:", epoch + 1)
    # loss_terms_dict_lst.append(train_conditional(diffusion_model=diffusion_model, dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler ,progress_bar=progress_bar ,verbose=True))
    device = next(diffusion_model.parameters()).device

    if loss_terms_weights is None:
        loss_terms_weights = {}
        for term_name in diffusion_model.loss_terms:
            loss_terms_weights[term_name] = 1

    training_loss = {}
    for term_name in diffusion_model.loss_terms:
        training_loss[term_name] = 0
    sample_cnt = 0
    diffusion_model.train()
    for step, data in enumerate(train_dataloader):

        if diffusion_model.model.encoder_type in ['frozen', 'fine-tune']:
            # use input_ids and attention mask from bert tokenizer
            # question1 as source, question2 as target
            input_ids = data['question2_input_ids'].to(device)
            attention_mask = data['question2_attention_mask'].to(device)
            src_ids = data['question1_input_ids_bert'].to(device)
            src_attention_mask = data['question1_attention_mask_bert'].to(device)
            loss_terms = diffusion_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            src_ids=src_ids,
                                            src_attention_mask=src_attention_mask)
        elif diffusion_model.model.encoder_type in ['from-scratch']:
            # use input_ids and attention mask from custom tokenizer
            input_ids = data['question2_input_ids'].to(device)
            attention_mask = data['question2_attention_mask'].to(device)
            src_ids = data['question1_input_ids'].to(device)
            src_attention_mask = data['question1_attention_mask'].to(device)
            loss_terms = diffusion_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            src_ids=src_ids,
                                            src_attention_mask=src_attention_mask)
        else:
            raise NotImplementedError

        loss = sum([v.mean()*loss_terms_weights[k] for k, v in loss_terms.items()])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        progress_bar.update(1)

        bs = input_ids.shape[0]
        sample_cnt += bs
        # training_loss += loss.detach().cpu() * bs

        for k, v in loss_terms.items():
            loss_term = (v.detach().cpu().mean()*loss_terms_weights[k]).item()
            loss_terms_dict[k].append(loss_term)
            wandb.log({k: loss_term})
            training_loss[k] += loss_term * bs

        if verbose and step % print_steps == print_steps-1:
            # print training loss
            # training_loss /= sample_cnt
            # print('step:', step+1)
            for k, v in training_loss.items():
                # print(k, ' training loss={:.5f}'.format(v/sample_cnt))
                wandb.log({f"{k} training loss": v/sample_cnt})
            sample_cnt = 0
            training_loss = {}
            for term_name in diffusion_model.loss_terms:
                training_loss[term_name] = 0

    # evaluate_conditional(diffusion_model=diffusion_model, dataloader=eval_dataloader,)
    device = next(diffusion_model.parameters()).device

    trained_ts = diffusion_model.word_embeddings.weight.data.cpu()
    embs = [trained_ts[id] for id in id_gt10]
    N = len(embs)
    trained_dist = 0
    for i in tqdm(range(N)):
        for j in range(i+1, N):
            trained_dist += (embs[i] - embs[j]).norm()
    trained_dist = trained_dist / (N * (N-1) / 2)
    wandb.log({'Emb avg dist': trained_dist})

    # print("Evaluating...")
    diffusion_model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in eval_dataloader:
            
            if diffusion_model.model.encoder_type in ['frozen', 'fine-tune']:
                # use input_ids and attention mask from bert tokenizer
                # question1 as source, question2 as target
                input_ids = data['question2_input_ids'].to(device)
                attention_mask = data['question2_attention_mask'].to(device)
                src_ids = data['question1_input_ids_bert'].to(device)
                src_attention_mask = data['question1_attention_mask_bert'].to(device)
                loss_terms = diffusion_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             src_ids=src_ids,
                                             src_attention_mask=src_attention_mask)
            elif diffusion_model.model.encoder_type in ['from-scratch']:
                # use input_ids and attention mask from custom tokenizer
                input_ids = data['question2_input_ids'].to(device)
                attention_mask = data['question2_attention_mask'].to(device)
                src_ids = data['question1_input_ids'].to(device)
                src_attention_mask = data['question1_attention_mask'].to(device)
                loss_terms = diffusion_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             src_ids=src_ids,
                                             src_attention_mask=src_attention_mask)
            else:
                raise NotImplementedError

            loss = sum([v.mean()*loss_terms_weights[k] for k, v in loss_terms.items()])
            bs = input_ids.shape[0]
            test_loss += bs*loss

        test_loss /= len(eval_dataloader.dataset)
        if verbose:
            # print('eval loss={:.5f}'.format(test_loss))
            wandb.log({'eval loss': test_loss})

save_path = f"checkpoints/conditional_from_scratch_emb{args.word_emb_dim}.pth"
torch.save(diffusion_model.state_dict(), save_path)
print(f"saved to {save_path}")
wandb.finish()
