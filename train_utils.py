import torch
from tqdm.auto import tqdm


def train(diffusion_model, dataloader, optimizer, scheduler=None, num_epoch=1, progress_bar=None, loss_terms_weights=None, verbose=False, print_steps=100):
    if progress_bar is None:
        progress_bar = tqdm(range(num_epoch*len(dataloader)))
    device = next(diffusion_model.parameters()).device

    loss_terms_dict = {}
    for term_name in diffusion_model.loss_terms:
        loss_terms_dict[term_name] = []

    if loss_terms_weights is None:
        loss_terms_weights = {}
        for term_name in diffusion_model.loss_terms:
            loss_terms_weights[term_name] = 1

    for epoch in range(num_epoch):
        training_loss = {}
        for term_name in diffusion_model.loss_terms:
            training_loss[term_name] = 0
        sample_cnt = 0
        diffusion_model.train()
        for step, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            loss_terms = diffusion_model(input_ids=input_ids, attention_mask=attention_mask)
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
                training_loss[k] += loss_term * bs

            if verbose and step % print_steps == print_steps-1:
                # print training loss
                # training_loss /= sample_cnt
                print('step:', step+1)
                for k, v in training_loss.items():
                    print(k, ' training loss={:.5f}'.format(v/sample_cnt))
                sample_cnt = 0
                training_loss = {}
                for term_name in diffusion_model.loss_terms:
                    training_loss[term_name] = 0
                
    return loss_terms_dict


def evaluate(diffusion_model, dataloader, loss_terms_weights=None, verbose=True):
    progress_bar = tqdm(range(len(dataloader)))
    device = next(diffusion_model.parameters()).device

    if loss_terms_weights is None:
        loss_terms_weights = {}
        for term_name in diffusion_model.loss_terms:
            loss_terms_weights[term_name] = 1

    diffusion_model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            loss_terms = diffusion_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum([v.mean()*loss_terms_weights[k] for k, v in loss_terms.items()])
            bs = input_ids.shape[0]
            test_loss += bs*loss
            progress_bar.update(1)

        test_loss /= len(dataloader.dataset)
        if verbose:
            print('eval loss={:.5f}'.format(test_loss))

    return test_loss


def train_conditional(diffusion_model,
                      dataloader,
                      optimizer,
                      scheduler=None,
                      num_epoch=1,
                      progress_bar=None,
                      loss_terms_weights=None,
                      verbose=False,
                      print_steps=100):
    if progress_bar is None:
        progress_bar = tqdm(range(num_epoch*len(dataloader)))
    device = next(diffusion_model.parameters()).device

    loss_terms_dict = {}
    for term_name in diffusion_model.loss_terms:
        loss_terms_dict[term_name] = []

    if loss_terms_weights is None:
        loss_terms_weights = {}
        for term_name in diffusion_model.loss_terms:
            loss_terms_weights[term_name] = 1

    for epoch in range(num_epoch):
        training_loss = {}
        for term_name in diffusion_model.loss_terms:
            training_loss[term_name] = 0
        sample_cnt = 0
        diffusion_model.train()
        for step, data in enumerate(dataloader):

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
                training_loss[k] += loss_term * bs

            if verbose and step % print_steps == print_steps-1:
                # print training loss
                # training_loss /= sample_cnt
                print('step:', step+1)
                for k, v in training_loss.items():
                    print(k, ' training loss={:.5f}'.format(v/sample_cnt))
                sample_cnt = 0
                training_loss = {}
                for term_name in diffusion_model.loss_terms:
                    training_loss[term_name] = 0

    return loss_terms_dict


def evaluate_conditional(diffusion_model,
                         dataloader,
                         loss_terms_weights=None,
                         verbose=True):
    progress_bar = tqdm(range(len(dataloader)))
    device = next(diffusion_model.parameters()).device

    if loss_terms_weights is None:
        loss_terms_weights = {}
        for term_name in diffusion_model.loss_terms:
            loss_terms_weights[term_name] = 1

    diffusion_model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            
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
            progress_bar.update(1)

        test_loss /= len(dataloader.dataset)
        if verbose:
            print('eval loss={:.5f}'.format(test_loss))

    return test_loss
