# import libraries
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from vae_gpt2_model import VAE_config, VAE_GPT2

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}

"""====================== CLASSES AND METHODS DEFINITIONS ======================"""

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# truncating/padding method
def truncating_padding_sentence(tokens, block_size):
    if (len(tokens) > block_size):
        original_tokens_len = block_size
        tokens = tokens[:block_size]
    else:
        original_tokens_len = len(tokens)
        tokens = tokens + ["[PAD]"]*(block_size - len(tokens))
    return tokens, original_tokens_len    
# creating attention mask method
def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type = "encoder_mask"):
    # args:
    #   sentence_length is length of real text, from [SOS] to <|endoftext|>
    #   seq_length is length with [PAD] (32, 64, 128, ...)
    
    if mask_type == "encoder_mask":
        mask_one_head = np.zeros([seq_length, seq_length])
        mask_one_head[:, :sentence_length] = 1  # the attention of [PAD] is also the input sentence
        mask_all_heads = [mask_one_head] * gpt2_config.n_head
        mask_all_heads = np.array(mask_all_heads)
    if mask_type == "decoder_mask":
        # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
        mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
        mask_all_heads = [mask_one_head] * gpt2_config.n_head   
        mask_all_heads = np.array(mask_all_heads)
    return mask_all_heads        


def convert_tensor_inference(model, sentence_embedding, args, device, current_generated_sentences):
    

    # convert to tensor and put on device
    sentence_embedding_converted = torch.FloatTensor(sentence_embedding).unsqueeze(0).to(device)
    
    # generate sentence
    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
    first_endoftext = generated_sample.find("<|endoftext|>") 
    generated_sample = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
    count = 1
    while ((generated_sample in current_generated_sentences) and (count<10)):
        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
        first_endoftext = generated_sample.find("<|endoftext|>") 
        generated_sample = str(generated_sample[:(first_endoftext)])
        count += 1
                
    current_generated_sentences.append(generated_sample)   
    
    ## DEBUGGING
    # logger.info("len(current_generated_sentences): " + str(len(current_generated_sentences)))    
    # logger.info("current_generated_sentences: " + str(current_generated_sentences))
    # logger.info("generated_sample: " + str(generated_sample))
    # print("count+: "+ str(count))

    # print generated sentence sample 
    generated_sample = generated_sample[6: ]
    print(str(generated_sample))
    
    return current_generated_sentences

def inference_test_1(model, args, device):
    print(" === inference_test_1 ===")
    print("Task 1: generate text from input sentence")
    
    # reading file 
    sentences = []
    with open(args.sentences_file, encoding="utf-8") as file_p:
        for sentence in file_p:
            sentences.append(sentence)
    
    
    # tokenizing, truncating/padding
    tokenized_sentences = [] 
    converted_tokenized_sentences = []    
    attention_mask_sentences = []
    for sentence in sentences:
        sentence_tokenized = model.tokenizer.tokenize(sentence)
        # encoder_input
        encoder_input = sentence_tokenized
        encoder_input, encoder_input_len = truncating_padding_sentence(encoder_input, args.block_size)
        encoder_input = model.tokenizer.convert_tokens_to_ids(encoder_input)
        encoder_input = np.array(encoder_input)        
        # encoder_attention_mask
        encoder_attention_mask = create_attention_mask(encoder_input_len, args.block_size, args.gpt2_config, "encoder_mask")
        
        tokenized_sentences.append(truncating_padding_sentence(sentence_tokenized, args.block_size))
        converted_tokenized_sentences.append(encoder_input)
        attention_mask_sentences.append(encoder_attention_mask)
    
    
    # get average embeddings and inferring
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_tokenized = tokenized_sentences[i]
        converted_tokenized_sentence = converted_tokenized_sentences[i]
        attention_mask_sentence = attention_mask_sentences[i]
        sentence_tokenized = model.tokenizer.decode(model.tokenizer.convert_tokens_to_ids(sentence_tokenized), clean_up_tokenization_spaces=True)
        converted_tokenized_sentence = torch.tensor(converted_tokenized_sentence).long().unsqueeze(0).to(device) # adding batch_size dimension
        attention_mask_sentence = torch.tensor(attention_mask_sentence).long().unsqueeze(0).to(device) # adding batch_size dimension
        
        sentence_embedding, z = model.get_embedding(converted_tokenized_sentence, attention_mask_sentence, device)
        
        ## DEBUGGING
        # logger.info("converted_tokenized_sentence_1: " + str(converted_tokenized_sentence_1.shape))
        # logger.info("attention_mask_sentence_1: " + str(attention_mask_sentence_1.shape))
        # logger.info("sentence_1_embedding: " + str(len(sentence_1_embedding)))
        # logger.info("sentence_1_embedding[0]: " + str(sentence_1_embedding[0].shape))            
        # logger.info("average_embedding: " + str(average_embedding[0][:5]))
        
        # inferring from average embedding
        generated, decoder_attentions = model.inference(sentence_embedding = sentence_embedding, args = args, device = device)
        generated_sentence =  model.tokenizer.decode(generated[0].tolist(), clean_up_tokenization_spaces=True)
        
        print("=====")
        print("input_sentence: " + str(sentence_tokenized))
        first_endoftext = generated_sentence.find("<|endoftext|>") 
        print("generated_sentence: " + str(generated_sentence[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sentence)]))
    
    return


def inference_test_2(model, args, device):
    print(" === inference_test_2 ===")
    print("Task 2: generate text from each dimension")
    
    
    # ==== Compute the embeddings    
    # check cached file    
    sentences_directory, sentences_filename = os.path.split(args.sentences_file)    
    model_directory = args.output_dir   
    cache_embeddings_file = os.path.join(model_directory, 'cached_inference_' + sentences_filename)
    logger.info("cache_embeddings_file: " + cache_embeddings_file)
    if os.path.exists(cache_embeddings_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cache_embeddings_file)
        with open(cache_embeddings_file, 'rb') as handle:
            # sentences_texts, sentences_embeddings = pickle.load(handle)        
            sentences_ids, sentences_texts, sentences_embeddings = pickle.load(handle)        

    
    # if no cached file    
    else:        
        # reading file
        print("reading input file.")
        sentences = []
        with open(args.sentences_file, encoding="utf-8") as file_p:
            for count, sentence in enumerate(file_p): 
                sentence = sentence.strip("\n")
                sentences.append(sentence) 
                
                
        # tokenizing, truncating/padding
        print("tokenizing and processing.")
        converted_tokenized_sentences = []    
        attention_mask_sentences = []
        for sentence in sentences:
            sentence_tokenized = model.tokenizer.tokenize(sentence)
            # encoder_input
            encoder_input = sentence_tokenized
            encoder_input, encoder_input_len = truncating_padding_sentence(encoder_input, args.block_size)
            encoder_input = model.tokenizer.convert_tokens_to_ids(encoder_input)
            encoder_input = np.array(encoder_input)        
            # encoder_attention_mask
            encoder_attention_mask = create_attention_mask(encoder_input_len, args.block_size, args.gpt2_config, "encoder_mask")
            
            converted_tokenized_sentences.append(encoder_input)
            attention_mask_sentences.append(encoder_attention_mask)
                
        # get embeddings
        batch_size = args.batch_size
        print("get embeddings with batch size {}.".format(str(batch_size)))
        sentences_ids = []
        sentences_embeddings = []
        sentences_texts = []
        for i in tqdm(list(range(0, len(sentences), batch_size))):
            sentences_text_batch = sentences[i:min(i+batch_size, len(sentences))]
            converted_tokenized_sentence = converted_tokenized_sentences[i:min(i+batch_size, len(sentences))]
            attention_mask_sentence = attention_mask_sentences[i:min(i+batch_size, len(sentences))]
            # convert to tensor, move to device
            converted_tokenized_sentence = torch.tensor(converted_tokenized_sentence).long().to(device) 
            attention_mask_sentence = torch.tensor(attention_mask_sentence).long().to(device) 
            # get embedding
            sentence_embedding, z = model.get_embedding(converted_tokenized_sentence, attention_mask_sentence, device)
            # append
            if sentences_embeddings == []:
                sentences_ids = list(range(i,min(i+batch_size, len(sentences))))
                sentences_embeddings = sentence_embedding.detach().cpu().numpy()
                sentences_texts = sentences_text_batch
            else:    
                sentences_ids.extend(list(range(i,min(i+batch_size, len(sentences)))))
                sentences_embeddings = np.concatenate((sentences_embeddings, sentence_embedding.detach().cpu().numpy()), axis = 0)
                sentences_texts.extend(sentences_text_batch)
        logger.info("sentences_embeddings.shape: " + str(sentences_embeddings.shape))
        
        # save to file
        with open(cache_embeddings_file, 'wb') as handle:
            pickle.dump([sentences_texts, sentences_embeddings], handle)    
            logger.info("dump file to: {}, number of records: {}, file size: {} MB".format(cache_embeddings_file, str(len(sentences_embeddings)), str(os.path.getsize(cache_embeddings_file)/pow(1024,2))))
        
        # save to file
        with open(cache_embeddings_file, 'wb') as handle:
            pickle.dump([sentences_ids, sentences_texts, sentences_embeddings], handle)    
            logger.info("dump file to: {}, number of records: {}, file size: {}".format(cache_embeddings_file, str(len(sentences_embeddings)), str(os.path.getsize(cache_embeddings_file))))
            # save as csv file
            cache_embeddings_file_csv, file_extension = os.path.splitext(cache_embeddings_file)
            cache_embeddings_file_csv += '.csv'
            logger.info("also save to csv file: {}".format(cache_embeddings_file_csv))
            text_df = pd.DataFrame(sentences_texts)
            sentences_ids = [str(sentences_id) for sentences_id in sentences_ids]
            text_df.index = sentences_ids
            text_df.insert(0, 'message_id', sentences_ids)
            text_df.columns = ["message_id", "message"]
            embeddings_df = pd.DataFrame(sentences_embeddings)
            embeddings_df.index = sentences_ids
            embeddings_df.columns = ["COMPONENT_" + str(i) for i in range(sentences_embeddings.shape[1])]
            df = text_df.merge(embeddings_df, left_on = text_df.index, right_on = embeddings_df.index)
            df = df.drop(columns=['key_0'])
            df.message_id = df.message_id.astype(str)
            df.to_csv(cache_embeddings_file_csv, index = False)

            
    # ==== Analyzing the embeddings  
    # method_1 for testing_2       
    if args.method == "method_1":
        # method 1 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)
        
        # set parameters
        std_steps = 5
        std_step_interval = 0.5
        std_random_level = 0
        print("UPDATED!")
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            # explore the positive direction
            for step in range(std_steps, -1, -1):
                std_position = (step+1)*std_step_interval
            
                print("samples around mean + {}*std:".format(std_position))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).unsqueeze(0)
           
                    # generate sentence
                    generatied_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
                        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                        generatied_count += 1
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                        if (generated_sample_clean not in generated_samples) or generatied_count >= 10:
                            generated_samples.append(generated_sample_clean)
                            break
                        
                    # print generated sentence sample
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))
        
            # explore the negative direction
            for step in range(0, std_steps + 1, 1):
                std_position = (step+1)*std_step_interval
                
                print("samples around mean - {}*std:".format(std_position))
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding - std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] - std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).unsqueeze(0)
              
                    # generate sentence
                    generatied_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
                        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                        generatied_count += 1
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                        if (generated_sample_clean not in generated_samples) or generatied_count >= 10:
                            generated_samples.append(generated_sample_clean)
                            break
                        
                    # print generated sentence sample
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))    


    # method_2 for testing_2
    if args.method == "method_2":
        
        # sentences_ids, sentences_texts, sentences_embeddings = pickle.load(handle)  
        df = pd.DataFrame(np.array(sentences_embeddings))
        df.columns = [str(latent_dim) for latent_dim in range(len(sentences_embeddings[0]))]
        df.index = sentences_ids
        df['text'] = sentences_texts
        print(df.columns)
        print(df.index[:5])
        

        # compute *std score
        hidden_size = len(sentences_embeddings[0])
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)    
        for i in range(hidden_size):
            df['*std_' + str(i)] = round((df[str(i)] - means[i])/stds[i],2)
            
            
        ## get unique sentence with highest score for each hidden dimension
        print("(=)(=)(=) unique sentence highest score (=)(=)(=)")
        for i in range(hidden_size):        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            top_count = 10
            print("(*) top max score: ")
            sorted_df = df.sort_values(by=[str(i)], ascending=False)
            printed_sentences = []
            for index, row in sorted_df.iterrows():
                if row['text'] not in printed_sentences:
                    count = len(sorted_df[sorted_df['text'] == row['text']])
                    print("{} - score: {} - count: {}".format(row['text'], round(row[str(i)],2), count))
                    printed_sentences.append(row['text'])
                if len(printed_sentences)== top_count:
                    break
                
            print("(*) top min score: ")
            sorted_df = df.sort_values(by=[str(i)], ascending=True)
            printed_sentences = []
            for index, row in sorted_df.iterrows():
                if row['text'] not in printed_sentences:
                    count = len(sorted_df[sorted_df['text'] == row['text']])
                    print("{} - *std score: {} - count: {}".format(row['text'], round(row['*std_' + str(i)],2), count))
                    printed_sentences.append(row['text'])
                if len(printed_sentences)== top_count:
                    break


    # method_3 for testing_2
    if args.method == "method_3":

        
        # sentences_ids, sentences_texts, sentences_embeddings = pickle.load(handle)  
        df = pd.DataFrame(np.array(sentences_embeddings))
        df.columns = [str(latent_dim) for latent_dim in range(len(sentences_embeddings[0]))]
        df.index = sentences_ids
        df['text'] = sentences_texts     
        print(df.columns)
        print(df.index[:5])
        

        # compute *std score
        hidden_size = len(sentences_embeddings[0])
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)    
        for i in range(hidden_size):
            df['*std_' + str(i)] = round((df[str(i)] - means[i])/stds[i],2)
        
        
        ## get sentence with highest frequency for each hidden dimension   
        print("(=)(=)(=) sentence highest frequency (=)(=)(=)")
        for i in range(hidden_size):        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            top_count = 8
            std_threshold = 1
            print("(*) top highest frequency above {}*std ".format(str(std_threshold)))
            df_pass_threshold = df.loc[df[str(i)] >= (means[i] + std_threshold*stds[i])]
            df_pass_threshold['text_std'] = df_pass_threshold['text'] + '    ' + df_pass_threshold[('*std_' + str(i))].astype(str)
            print(df_pass_threshold['text_std'].value_counts()[:top_count])
            print("(*) top highest frequency below -{}*std: ".format(str(std_threshold)))
            df_pass_threshold = df.loc[df[str(i)] <= (means[i] - std_threshold*stds[i])]
            df_pass_threshold['text_std'] = df_pass_threshold['text'] + '    ' + df_pass_threshold[('*std_' + str(i))].astype(str)
            print(df_pass_threshold['text_std'].value_counts()[:top_count])
            
            df_pass_threshold.groupby(['text', '*std_' + str(i)])    
        
    return
    


"""====================== MAIN FUNCTION ======================"""

# main function
def main():
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--sentences_file", default=None, type=str,
                        help="An optional input generate data file.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--primals_file", default=None, type=str,
                        help="An optional input generate data file.")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    # model parameters
    parser.add_argument("--gpt2_model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")    
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 
    
    # generating parameters
    parser.add_argument("--inference_test", default=0, type=int)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--generate_num", type=int, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--batch_size", type=int, default=None)

    # other generating parameters
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
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")    
    
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #
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


    # =========== bulilding model and inferencing  =========== #
    # Building model
    gpt2_config_class, gpt2_class, tokenizer_class = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    vae_config = VAE_config(latent_size = args.latent_size)
    model = VAE_GPT2(gpt2_config, vae_config, args)
    
    
    # Load from checkpoint model
    model.from_pretrained(args)
    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)
    
    # Send model to GPU
    model.to(args.device)    

    # Logging info
    logger.info("Inference parameters %s", args)
    

    # Testing inference
    args.gpt2_config = model.gpt2_config
    
    if args.inference_test == 1:        
        # Inference test 1 - regenerate text
        inference_test_1(model, args, device)    
    if args.inference_test == 2:            
        # Inference test 2 - sample for each of latent dimension
        inference_test_2(model, args, device)
    if args.inference_test == 3:        
        # Inference test 3 - interpolation generating
        inference_test_3(model, args, device)

    
if __name__ == "__main__":
    main()        




