# import libraries
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import numpy as np
import torch


from torch.utils.data import DataLoader, Dataset, RandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from vae_gpt2_model import VAE_config, VAE_GPT2

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}


"""====================== CLASSES AND METHODS DEFINITIONS ======================"""


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, args):
        
        # reading data file
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(args.block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)       

            # defining method for creating mask
            def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type):
                # args:
                #   sentence_length is length of real text, from [SOS] to <|endoftext|>
                #   seq_length is length with [PAD] (32, 64, 128, ...)
                
                if mask_type == "encoder_mask":
                    mask_one_head = np.zeros([seq_length, seq_length])
                    mask_one_head[:, :sentence_length] = 1   # the attention of [PAD] is also the input sentence
                    mask_all_heads = [mask_one_head] * gpt2_config.n_head
                    mask_all_heads = np.array(mask_all_heads)
                if mask_type == "decoder_mask":
                    # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
                    mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
                    mask_all_heads = [mask_one_head] * gpt2_config.n_head   
                    mask_all_heads = np.array(mask_all_heads)
                return mask_all_heads            
            
            def truncating_padding_sentence(tokens, block_size):
                if (len(tokens) > block_size):
                    original_tokens_len = block_size
                    tokens = tokens[:block_size]
                else:
                    original_tokens_len = len(tokens)
                    tokens = tokens + ["[PAD]"]*(block_size - len(tokens))
                return tokens, original_tokens_len    
                
            
            # reading file
            self.examples = []
            with open(file_path, encoding="utf-8") as file_p:
                for count, sentence in enumerate(file_p): 
                    
                    # tokenizing text
                    sentence_text = sentence
                    sentence_tokenized = tokenizer.tokenize(sentence_text)
                    
                    # encoder_input
                    encoder_input = sentence_tokenized
                    encoder_input, encoder_input_len = truncating_padding_sentence(encoder_input, args.block_size)
                    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
                    encoder_input = np.array(encoder_input)
                    # decoder_input
                    decoder_input = ["[SOS]"] + sentence_tokenized
                    decoder_input, decoder_input_len = truncating_padding_sentence(decoder_input, args.block_size)
                    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)
                    decoder_input = np.array(decoder_input)
                    # decoder_output
                    decoder_label = sentence_tokenized + ["<|endoftext|>"]
                    decoder_label, decoder_label_len = truncating_padding_sentence(decoder_label, args.block_size)
                    decoder_label = tokenizer.convert_tokens_to_ids(decoder_label)
                    decoder_label = np.array(decoder_label)

                    # encoder_attention_mask
                    encoder_attention_mask = create_attention_mask(encoder_input_len, args.block_size, args.gpt2_config, "encoder_mask")
                    # decoder_attention_mask
                    decoder_attention_mask = create_attention_mask(decoder_input_len, args.block_size, args.gpt2_config, "decoder_mask")


                    # append to examples list
                    training_sentence = dict({"sentence_text": sentence_text, "encoder_input": encoder_input, "encoder_attention_mask": encoder_attention_mask ,"decoder_input": decoder_input, "decoder_attention_mask": decoder_attention_mask, "decoder_label": decoder_label})  
                    self.examples.append(training_sentence)
                  
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
    
def load_and_cache_examples(args, file_path, tokenizer):
    dataset = TextDataset(tokenizer, file_path=file_path, args=args)
    return dataset    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)   
        
def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)            

def loss_fn(decoder_lm_logits, target, mean, logv, anneal_function, step, k, x0, ignore_index):
    
    # Negative Log Likelihood
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
    # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
    NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)
    
    return NLL_loss, KL_loss, KL_weight

"""====================== TRAIN/EVALUATE FUNCTION ======================"""

# train and evaluate function
def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # ===== Setting up
    # summary writer
    tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)    

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?

    
    # ===== Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * 1)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    if args.from_checkpoint:
        global_step = args.start_step
        t_total += args.start_step
    else:
        global_step = 0


    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    kl_loss_report = []
    nll_loss_report = []
    loss_report = []
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            encoder_input = batch["encoder_input"].long() 
            decoder_input = batch["decoder_input"].long() 
            decoder_label = batch["decoder_label"].long()        
            encoder_attention_mask = batch["encoder_attention_mask"].long()
            decoder_attention_mask = batch["decoder_attention_mask"].long()
            encoder_input = encoder_input.to(args.device)
            decoder_input = decoder_input.to(args.device)
            decoder_label = decoder_label.to(args.device)
            encoder_attention_mask = encoder_attention_mask.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)
            model.train()

            ## DEBUGGING
            # model.encoder.eval()
            # model.decoder.train()
            # model.vae.train()

            # forward pass (change and edit with VAE code)
            decoder_lm_logits, mean, logv, z = model(encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask)


            # compute loss  
            NLL_loss, KL_loss, KL_weight = loss_fn(decoder_lm_logits, decoder_label, mean, logv, args.anneal_function, step, args.k, args.x0, tokenizer.convert_tokens_to_ids(["[PAD]"])[0])    
            ## DEBUGGING
            KL_weight = 0
            loss = (NLL_loss + KL_weight * KL_loss)
            kl_loss = KL_loss
            nll_loss = NLL_loss


            # DEBUGGING
            predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
            logger.info("decoder_input: " + str(model.tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True).replace("[PAD]", ""))) 
            logger.info("decoder_label: " + str(model.tokenizer.decode(decoder_label[0].tolist(), clean_up_tokenization_spaces=True).replace("[PAD]", "")))
            prediction_text = model.tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
            first_endoftext = prediction_text.find("<|endoftext|>") 
            logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                kl_loss = kl_loss.mean()
                nll_loss = nll_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                kl_loss = kl_loss / args.gradient_accumulation_steps
                nll_loss = nll_loss / args.gradient_accumulation_steps
            
            ## DEBUGGING
            # logger.info("global_step: " + str(global_step) + "/" + str(int(t_total)))    
            # logger.info("train_loss: " + str(round(float(loss),3))) 
            # logger.info("kl_loss: " + str(round(float(kl_loss),3)) + " (weight: " +  str(round(float(KL_weight),3)) + ")" + " / " + "nll_loss: " + str(round(float(nll_loss),3))) 
            # logger.info("std: " + str(   torch.norm(torch.exp(0.5 * logv).contiguous().view(-1)).item()   ) )  
            
            kl_loss_report.append(kl_loss.data)
            nll_loss_report.append(nll_loss.data)
            loss_report.append(loss.data)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()


            # accummulte enough step, step backward
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # logging and save checkpooints
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # save model
                    loss_reports = [kl_loss_report, nll_loss_report, loss_report]
                    model.save_pretrained(args, output_dir, loss_reports)                    
                    logger.info("Saving model checkpoint to %s", output_dir)
                    _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # close summary writer
    tb_writer.close()

    return global_step, tr_loss, loss_reports


def evaluate(args, eval_dataset, model, tokenizer):
    """ Train the model """

    # ===== Setting up
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?

    
    # ===== Evaluate!
    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    kl_loss_report = 0
    nll_loss_report = 0
    nb_eval_steps = 0
    model.eval()        
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    for batch in epoch_iterator:
        nb_eval_steps += 1
        
        encoder_input = batch["encoder_input"].long() 
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        
        encoder_attention_mask = batch["encoder_attention_mask"].long()
        decoder_attention_mask = batch["decoder_attention_mask"].long()
        encoder_input = encoder_input.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        encoder_attention_mask = encoder_attention_mask.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)
        model.eval()


        # forward pass (change and edit with VAE code)
        decoder_lm_logits, mean, logv, z = model(encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask)


        # compute loss  
        NLL_loss, KL_loss, KL_weight = loss_fn(decoder_lm_logits, decoder_label, mean, logv, args.anneal_function, 0, args.k, args.x0, tokenizer.convert_tokens_to_ids(["[PAD]"])[0])    
        ## DEBUGGING
        loss = (NLL_loss + KL_weight * KL_loss)
        kl_loss = KL_loss
        nll_loss = NLL_loss


        ## DEBUGGING
        # predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
        # logger.info("decoder_input: " + str(model.tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True).replace("[PAD]", "")))
        # logger.info("decoder_label: " + str(model.tokenizer.decode(decoder_label[0].tolist(), clean_up_tokenization_spaces=True).replace("[PAD]", "")))
        # prediction_text = model.tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
        # first_endoftext = prediction_text.find("<|endoftext|>") 
        # logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13)])) 


        # process loss across GPUs, batches then backwards
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            kl_loss = kl_loss.mean()
            nll_loss = nll_loss.mean()
        kl_loss_report += kl_loss.data
        nll_loss_report += nll_loss.data

    kl_loss_report = kl_loss_report / nb_eval_steps
    nll_loss_report = nll_loss_report / nb_eval_steps
    perplexity = torch.exp(torch.tensor(nll_loss_report))

    result = {
        "perplexity": perplexity,
        "nll_loss": nll_loss_report,
        "kl_loss": kl_loss_report
    }


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result

def evaluate_factorization(args, eval_dataset, model, tokenizer, loadings_matrix):
    """ Train the model """

    # ===== Setting up
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # bring loadings_matrix to tensor
    loadings_matrix_inverse = np.array(np.matrix(loadings_matrix).I)
    reconstruct_matrix = np.matmul(loadings_matrix, loadings_matrix_inverse)
    logger.info("reconstruct_matrix: " + str(reconstruct_matrix))
    loadings_matrix = torch.FloatTensor(loadings_matrix).to(args.device)
    loadings_matrix_inverse = torch.FloatTensor(loadings_matrix_inverse).to(args.device)
    reconstruct_matrix = torch.FloatTensor(reconstruct_matrix).to(args.device)


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?

    
    # ===== Evaluate!
    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nll_loss_report = 0
    nb_eval_steps = 0
    model.eval()        
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    for batch in epoch_iterator:
        nb_eval_steps += 1
        
        encoder_input = batch["encoder_input"].long() 
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        
        encoder_attention_mask = batch["encoder_attention_mask"].long()
        decoder_attention_mask = batch["decoder_attention_mask"].long()
        encoder_input = encoder_input.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        encoder_attention_mask = encoder_attention_mask.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)
        model.eval()


        # forward pass (change and edit with VAE code)
        # get_embedding
        sentences_embeddings, _ = model.get_embedding(encoder_input, encoder_attention_mask, args.device)
        reconstructed_factorized_sentences_embeddings = torch.matmul(sentences_embeddings, reconstruct_matrix.transpose(0,1))        
        
        ## DEBUGGING
        # logger.info("reconstructed_factorized_sentences_embeddings: " + str(reconstructed_factorized_sentences_embeddings.shape))
        # logger.info("decoder_input: " + str(decoder_input.shape))
        # logger.info("decoder_attention_mask: " + str(decoder_attention_mask.shape))
        
        # decode_from_embedding
        decoder_lm_logits = model.decode_from_embedding(reconstructed_factorized_sentences_embeddings, args, args.device, decoder_input, decoder_attention_mask)
    

        # compute loss  
        mean = torch.zeros(sentences_embeddings.shape, device = args.device) # dumb value
        logv = torch.zeros(sentences_embeddings.shape, device = args.device) # dumb value
        NLL_loss, KL_loss, KL_weight = loss_fn(decoder_lm_logits, decoder_label, mean, logv, args.anneal_function, 0, args.k, args.x0, tokenizer.convert_tokens_to_ids(["[PAD]"])[0])    
        ## DEBUGGING
        nll_loss = NLL_loss

        # process loss across GPUs, batches then backwards
        if args.n_gpu > 1:
            nll_loss = nll_loss.mean()
        nll_loss_report += nll_loss.data

    nll_loss_report = nll_loss_report / nb_eval_steps
    perplexity = torch.exp(torch.tensor(nll_loss_report))

    result = {
        "perplexity": perplexity,
        "nll_loss": nll_loss_report,
    }


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return 

def evaluate_divergence(args, eval_dataset, model, tokenizer, dimension_embedding):
    """ Train the model """

    # ===== Setting up
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # bring loadings_matrix to tensor

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?
    
    # ===== Evaluate!
    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nll_loss_report = 0
    nb_eval_steps = 0
    model.eval()        
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    for batch in epoch_iterator:
        nb_eval_steps += 1
        
        encoder_input = batch["encoder_input"].long() 
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        
        encoder_attention_mask = batch["encoder_attention_mask"].long()
        decoder_attention_mask = batch["decoder_attention_mask"].long()
        encoder_input = encoder_input.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        encoder_attention_mask = encoder_attention_mask.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)
        model.eval()

        # create dimension_embedding_batch as embedding for decoders
        dimension_embedding_batch = np.array(encoder_input.shape[0]*[dimension_embedding])
        dimension_embedding_batch = torch.FloatTensor(dimension_embedding_batch).to(args.device)
        
        
        # decode_from_embedding
        decoder_lm_logits = model.decode_from_embedding(dimension_embedding_batch, args, args.device, decoder_input, decoder_attention_mask)
    
        # compute loss  
        mean = torch.zeros(decoder_lm_logits.shape, device = args.device) # dumb value
        logv = torch.zeros(decoder_lm_logits.shape, device = args.device) # dumb value
        NLL_loss, KL_loss, KL_weight = loss_fn(decoder_lm_logits, decoder_label, mean, logv, args.anneal_function, 0, args.k, args.x0, tokenizer.convert_tokens_to_ids(["[PAD]"])[0])    
        ## DEBUGGING
        nll_loss = NLL_loss/args.eval_batch_size


        # process loss across GPUs, batches then backwards
        if args.n_gpu > 1:
            nll_loss = nll_loss.mean()
        nll_loss_report += nll_loss.data

    nll_loss_report = nll_loss_report / nb_eval_steps
    perplexity = torch.exp(torch.tensor(nll_loss_report))

    result = {
        "perplexity": perplexity,
        "nll_loss": nll_loss_report,
    }


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return 

def get_embedding(args, input_dataset, model, tokenizer):
    """ Train the model """

    # ===== Setting up
    args.input_batch_size = args.per_gpu_input_batch_size * max(1, args.n_gpu)
    input_sampler = RandomSampler(input_dataset)
    input_dataloader = DataLoader(input_dataset, sampler=input_sampler, batch_size=args.input_batch_size)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    # CHECK! IF "model.to(args.device)" or "torch.nn.DataParallel(model)" first?

    
    # ===== Evaluate!
    logger.info("***** Running get embedding *****")
    logger.info("  Num examples = %d", len(input_dataset))
    logger.info("  Batch size = %d", args.input_batch_size)
    model.eval()        
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    sentences_embeddings = torch.Tensor([])
    sentences_text = []
    epoch_iterator = tqdm(input_dataloader, desc="Iteration", disable=False)
    for batch in epoch_iterator:
        
        encoder_input = batch["encoder_input"].long() 
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        
        sentences_text_batch = batch["sentence_text"]
        encoder_attention_mask = batch["encoder_attention_mask"].long()
        decoder_attention_mask = batch["decoder_attention_mask"].long()
        encoder_input = encoder_input.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        encoder_attention_mask = encoder_attention_mask.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)
        model.eval()

        # forward pass (change and edit with VAE code)
        mean, z = model.inference(encoder_input, encoder_attention_mask, args.device)
        
        # append 
        sentences_embeddings = sentences_embeddings.cat(mean, dim = 0)
        sentences_text.extent(sentences_text_batch)
        
    return        

"""====================== MAIN FUNCTION ======================"""


# main function
def main():
    
    # =========== parameters parsing =========== #
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")   
    parser.add_argument("--do_eval_factor", action='store_true',
                        help="Whether to run eval factorization on the dev set.")   
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--loading_matrix_file", default=None, type=str,
                        help="Factorization loading matrix")
    parser.add_argument("--from_checkpoint", action='store_true',
                        help="To initialize model or load from a checkpoint.")    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    
    # model parameters
    parser.add_argument("--gpt2_model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 

    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)                        
    parser.add_argument('--start_step', type=int, default=0)                        
    parser.add_argument("--frozen_layers", type=str, default='None', 
                        help="Layers to be frozen while training.")
   

    # other training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # other parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #
    # checking parameters
    if args.eval_data_file is None and args.do_eval :
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file or remove the --do_eval argument.")
    if args.from_checkpoint is None and args.do_eval :
        raise ValueError("Cannot do evaluation without specified checkpoint.")
    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir and not(args.from_checkpoint):
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    # setting things up    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")    # CHECK! make sure we use all 3 GPUs
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # Set seed
    set_seed(args)


    # =========== bulilding model and training/evaluating  =========== #

    # Building model
    gpt2_config_class, gpt2_class, tokenizer_class = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    vae_config = VAE_config(latent_size = args.latent_size)
    model = VAE_GPT2(gpt2_config, vae_config, args)
    
    
    # Initialize / Load from checkpoint model
    if args.from_checkpoint == False:
        model.initialize_model(args)    # initialize model with pretrained GPT2
    else:
        model.from_pretrained(args)
    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)


    # Send model to GPU
    model.to(args.device)


    # Logging info
    logger.info("Inference parameters %s", args)


    # Training
    if args.do_train:
        
        # DEBUGGING
        # print model parameters 
        logger.info("VAE")
        for name, param in model.named_parameters():
            logger.info(name + ' - ' + str(param.requires_grad))
        logger.info("ENCODER")    
        for name, param in model.encoder.named_parameters():
            logger.info(name + ' - ' + str(param.requires_grad))
        logger.info("DECODER")     
        for name, param in model.decoder.named_parameters():
            logger.info(name + ' - ' + str(param.requires_grad))
         
            
        #  freeze layers
        if args.frozen_layers is not None:
            frozen_layers = args.frozen_layers.split(" ")
            for name, param in model.named_parameters():
                if any(".{}.".format(str(frozen_layer)) in name for frozen_layer in frozen_layers):
                    logger.info("frozen params: " + name)
                    param.requires_grad = False
            
            
        # load train_dataset
        args.gpt2_config = model.gpt2_config
        train_dataset = load_and_cache_examples(args, args.train_data_file, model.tokenizer)

        # running train function
        global_step, tr_loss, loss_reports = train(args, train_dataset, model, model.tokenizer)
        logger.info(" global_step = %s", global_step)

        # saving model
        model.save_pretrained(args, args.output_dir, loss_reports)
        
        # good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))        

    # Evaluating
    if args.do_eval:
        # load dataset
        args.gpt2_config = model.gpt2_config
        eval_dataset = load_and_cache_examples(args, args.eval_data_file , model.tokenizer)        

        # reading loss_reports
        loss_reports = pickle.load( open( args.output_dir + "/loss_reports.pkl", "rb" ) )
        nll_loss_report = loss_reports[1]
        nll_loss_report = [nll_loss_report_tensor.item() for nll_loss_report_tensor in nll_loss_report]
        logger.info("training loss:")
        segment_step = 50
        for i in range(0,len(nll_loss_report), segment_step):
            nll_loss_divided = np.mean(nll_loss_report[i:(i + segment_step)])
            logger.info("train loss at step {}: {}".format(str(i), str(round(nll_loss_divided,3))))
            
            

        if not(args.eval_all_checkpoints):
            # running train function
            evaluate(args, eval_dataset, model, model.tokenizer)
        else:

            # set new checkpoint
            d = args.output_dir
            checkpoint_directories = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
            checkpoint_directories = [directory for directory in checkpoint_directories if "checkpoint" in directory]
            logger.info("all checkpoints:")
            logger.info(checkpoint_directories)
            
            result_all_checkpoints = []
            for i, checkpoint in enumerate(checkpoint_directories):
                
                # DEBUGGING
                checkpoint_number = checkpoint.split('-')[2]
                if int(checkpoint_number)>7500:
                    result_all_checkpoints.append({'nll_loss' : 0})
                    continue
                
                # reset model
                del model
                gpt2_config.vocab_size = gpt2_config.vocab_size - 2
                args.gpt2_config = gpt2_config
                model = VAE_GPT2(gpt2_config, vae_config, args)                
                # modify checkpoingt path
                args.output_dir = checkpoint
                logger.info("==> checkpoint: " + str(checkpoint))
                # load model
                model.from_pretrained(args)
                model.to(args.device)
                # evaluate
                result_all_checkpoints.append(evaluate(args, eval_dataset, model, model.tokenizer))

            # print all results
            for i, checkpoint in enumerate(checkpoint_directories):

                if int(checkpoint_number)>7500:    
                    continue
                
                checkpoint_number = checkpoint.split('-')[2]                
                result = result_all_checkpoints[i]
                logger.info("==> checkpoint {}: {}".format(str(checkpoint_number), str(result["nll_loss"])))
       

    # Evaluating factorization
    if args.do_eval_factor:
        # load dataset
        args.gpt2_config = model.gpt2_config
        eval_dataset = load_and_cache_examples(args, args.eval_data_file , model.tokenizer)        

        # get loading matrix
        with open(args.loading_matrix_file, 'rb') as handle:
            [nmf_score, nmf_loading_matrix] = pickle.load(handle)  
        loadings_matrix = nmf_loading_matrix

        # running train function
        evaluate_factorization(args, eval_dataset, model, model.tokenizer, loadings_matrix)    
 
        
if __name__ == "__main__":
    main()        