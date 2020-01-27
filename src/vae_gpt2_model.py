# import libraries
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import logging
import torch.nn.functional as F
import os
import pickle

logger = logging.getLogger(__name__)

"""====================== CLASSES DEFINITIONS ======================"""

# define VAE_config
class VAE_config():
    def __init__(self, latent_size = None):
      self.latent_size = latent_size

# define class VAE
class VAE(nn.Module):
    def __init__(self, gpt2_config, vae_config):
        super(VAE, self).__init__()
        self.latent_size = vae_config.latent_size
        self.hidden2mean = nn.Linear(gpt2_config.n_embd * 2, vae_config.latent_size)    # now we take only layer [-2], and for each layer we take (key, value) pair
        self.hidden2logv = nn.Linear(gpt2_config.n_embd * 2, vae_config.latent_size)    # CHECK! because there are bias in here, make sure we take into account of that with the bias part
        self.latent2hidden = nn.Linear(vae_config.latent_size, gpt2_config.n_embd * 2)

    def forward(self, secondToLast_encoder_presents, device):

        # parameterization
        mean = self.hidden2mean(secondToLast_encoder_presents)
        logv = self.hidden2logv(secondToLast_encoder_presents)
        std = torch.exp(0.5 * logv) 
        batch_size = secondToLast_encoder_presents.shape[0] # CHECK! batch size here when splitted
        z = torch.randn([batch_size, self.latent_size]).to(device)    
        z = z * std + mean        

        ## DEBUGGING!
        z = mean

        ## reconstruct from hidden dimension         
        secondToLast_decoder_hidden = self.latent2hidden(z)  
        
        ## DEBUGGING!
        # secondToLast_decoder_hidden = secondToLast_encoder_presents

        return secondToLast_decoder_hidden, mean, logv, z

    def get_embedding(self,secondToLast_encoder_presents, device = None):
        
        mean = self.hidden2mean(secondToLast_encoder_presents)
        logv = self.hidden2logv(secondToLast_encoder_presents)
        std = torch.exp(0.5 * logv)
        batch_size = secondToLast_encoder_presents.shape[0]
        z = torch.randn([batch_size, self.latent_size]).to(device)
        z = z * std + mean   
        
        # DEBUGGING
        z = mean
        
        return mean, z


    def inference(self, gpt2_config, z = None, device = None):
        
        # sampling if z is None
        if z is None:
            z = torch.randn([1, self.latent_size], device = device)
            
        # reconstruct from hidden
        secondToLast_decoder_hidden = self.latent2hidden(z)
        
        return secondToLast_decoder_hidden
    
# define class VAE_GPT2
class VAE_GPT2(nn.Module):

    def __init__(self, gpt2_config, vae_config, args):
        super(VAE_GPT2, self).__init__()

        # set up encoder and decoder have the same config
        self.gpt2_config = gpt2_config
        self.vae_config = vae_config
        self.tokenizer = None
        self.encoder = None
        self.decoder = None
        self.vae = VAE(self.gpt2_config, self.vae_config)

        # set up gpt2_config
        self.gpt2_config.output_hidden_states = True
        self.gpt2_config.output_past = True
        self.gpt2_config.output_attentions = True


    def initialize_model(self, args):

        # load pretrained model and tokenizer for GPT2 encoder and decoder
        encoder_path = args.gpt2_model_name_or_path
        decoder_path = args.gpt2_model_name_or_path   
        tokenizer_path = args.gpt2_model_name_or_path
        self.encoder = GPT2Model.from_pretrained(encoder_path, from_tf=bool('.ckpt' in encoder_path), config=self.gpt2_config)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)


        # add [SOS] and [PAD] to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[SOS]"]})
        self.encoder.resize_token_embeddings(len(self.tokenizer))   
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        logger.info("tokenizer size: " + str(self.tokenizer.__len__()))
        logger.info("tokenizer.decode [50256, 50257, 50258]: " + str(self.tokenizer.decode([50256, 50257, 50258])) )        

        # No controled initialization for VAE
        logger.info("cautions: no init VAE")

        return


    def save_pretrained(self, args, output_dir, loss_reports):

        # set up output_dir to save sub-models
        output_dir_encoder = output_dir + "/encoder/"
        output_dir_decoder = output_dir + "/decoder/"
        output_dir_tokenizer = output_dir + "/tokenizer/"
        output_dir_vae = output_dir + "/vae/"
        if not os.path.exists(output_dir_encoder):
            os.makedirs(output_dir_encoder)
        if not os.path.exists(output_dir_decoder):
            os.makedirs(output_dir_decoder)            
        if not os.path.exists(output_dir_tokenizer):
            os.makedirs(output_dir_tokenizer)
        if not os.path.exists(output_dir_vae):
            os.makedirs(output_dir_vae)
        output_dir_vae = output_dir_vae + "/vae.weights"    


        # save model
        self.encoder.save_pretrained(output_dir_encoder)
        self.decoder.save_pretrained(output_dir_decoder)
        self.tokenizer.save_pretrained(output_dir_tokenizer)
        torch.save(self.vae.state_dict(),output_dir_vae)       

        # save training args and loss record
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
        loss_reports_file = open(output_dir + "/loss_reports.pkl", "wb")
        pickle.dump(loss_reports, loss_reports_file)
        
        return


    def from_pretrained(self, args):
        
        # loading from pre-trained
        encoder_path = args.output_dir + "/encoder/"
        decoder_path = args.output_dir + "/decoder/"
        vae_path = args.output_dir + "/vae/vae.weights"
        tokenizer_path = args.output_dir + "/tokenizer/"
        logger.info("gpt2_config: " + str(self.gpt2_config))
        self.gpt2_config.vocab_size = self.gpt2_config.vocab_size + 2 
        self.encoder = GPT2Model.from_pretrained(encoder_path, from_tf=bool('.ckpt' in encoder_path), config=self.gpt2_config)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        self.vae.load_state_dict(torch.load(vae_path))

        # set up for evaluating
        self.encoder.eval()
        self.decoder.eval()
        self.vae.eval()

        # load training args
        training_args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        logger.info("training_args: " + str(training_args))

        return  


    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):

        # encoder
        encoder_input_ids = encoder_input_ids
        encoder_attention_mask = encoder_attention_mask
        encoder_last_hidden_state, encoder_presents, encoder_hidden_states, encoder_attentions = self.encoder(input_ids = encoder_input_ids, attention_mask = encoder_attention_mask)
        # processing encoder to feed to vae
        encoder_presents = torch.stack(encoder_presents) 
        batch_size = encoder_presents.shape[1]
        device = encoder_presents.get_device()
        # process encoder_presents into 1 vector for each sample in batch
        secondToLast_encoder_presents = encoder_presents[-2]
        ## mean of all embeddings including [PAD]
        # logger.info("Taking average embeddings of secondToLast of all tokens including [PAD].")
        secondToLast_encoder_presents = secondToLast_encoder_presents.mean(dim = -2, keepdim = True)    # take into account [PAD], because [PAD] is also only attentioned on sentence
        # reshape to bring batch_size to be the 1st dimension
        secondToLast_encoder_presents = secondToLast_encoder_presents.reshape([batch_size, -1])  # reshape into [batch_size, hidden_size * 2] 

        
        # vae
        secondToLast_decoder_hidden, mean, logv, z = self.vae(secondToLast_encoder_presents, device)


        # decoder
        decoder_hidden = torch.zeros([self.gpt2_config.n_layer, batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)], device = device)
        secondToLast_decoder_hidden = secondToLast_decoder_hidden.reshape([batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        decoder_hidden[-2] = secondToLast_decoder_hidden
        decoder_input_ids = decoder_input_ids
        decoder_attention_mask = decoder_attention_mask
        past = decoder_hidden
        # decoder forward pass
        decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = decoder_input_ids, past = past, attention_mask = decoder_attention_mask)

        
        return decoder_lm_logits, mean, logv, z 

    def inference(self, sentence_embedding = None, args = None, device = None):
        if sentence_embedding is None:
            batch_size = 1
        else:    
            batch_size = sentence_embedding.shape[0]
        
        # construct hidden vector
        secondToLast_decoder_hidden = self.vae.inference(self.gpt2_config, z = sentence_embedding, device = device) 
        
        # decoder
        decoder_hidden = torch.zeros([self.gpt2_config.n_layer, batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)], device = device)
        secondToLast_decoder_hidden = secondToLast_decoder_hidden.reshape([batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        decoder_hidden[-2] = secondToLast_decoder_hidden
        decoder_input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("[SOS]")]*batch_size, device = device).reshape(batch_size,1)
        past = decoder_hidden
        
     
        # generate tokens
        generated = decoder_input_ids
        for _ in range(args.generate_length):

            ## DEBUGGING
            # logger.info("generated: " + str(generated))
            
            # decoder forward pass
            decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = generated, past = past, attention_mask = None)
            
            # sample from vocabulary
            decoder_lm_logits = decoder_lm_logits[:,-1,:]
            filtered_decoder_lm_logits = top_k_top_p_filtering(decoder_lm_logits, top_k=args.top_k, top_p=args.top_p)
            if args.temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_decoder_lm_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_decoder_lm_logits, dim=-1), num_samples=1)                
            generated = torch.cat((generated, next_token), dim=1)
    
            ## DEBUGGING
            # if generated.shape[1]==5:
            #    logger.info("decoder_lm_logits: " + str(decoder_lm_logits))
            #    logger.info("max(decoder_lm_logits): " + str(torch.max(decoder_lm_logits[0])))
    
        return generated, decoder_attentions


    def get_embedding(self, encoder_input_ids, encoder_attention_mask, device):
        
        # encoder
        encoder_input_ids = encoder_input_ids
        encoder_attention_mask = encoder_attention_mask
        encoder_last_hidden_state, encoder_presents, encoder_hidden_states, encoder_attentions = self.encoder(input_ids = encoder_input_ids, attention_mask = encoder_attention_mask)
        # processing encoder to feed to vae
        encoder_presents = torch.stack(encoder_presents) 
        batch_size = encoder_presents.shape[1]
        device = encoder_presents.get_device()
        # process encoder_presents into 1 vector for each sample in batch
        secondToLast_encoder_presents = encoder_presents[-2]
        ## mean of all embeddings including [PAD]
        # logger.info("Taking average embeddings of secondToLast of all tokens including [PAD].")
        secondToLast_encoder_presents = secondToLast_encoder_presents.mean(dim = -2, keepdim = True)    # take into account [PAD], because [PAD] is also only attentioned on sentence
        # reshape to bring batch_size to be the 1st dimension
        secondToLast_encoder_presents = secondToLast_encoder_presents.reshape([batch_size, -1])  # reshape into [batch_size, hidden_size * 2] 

        
        # vae
        mean, z = self.vae.get_embedding(secondToLast_encoder_presents, device)

        return mean, z

    def decode_from_embedding(self, sentence_embedding, args, device, decoder_input_ids, decoder_attention_mask):
  
        batch_size = sentence_embedding.shape[0]
        
        # construct hidden vector
        secondToLast_decoder_hidden = self.vae.inference(self.gpt2_config, z = sentence_embedding, device = device) 
        
        # decoder
        decoder_hidden = torch.zeros([self.gpt2_config.n_layer, batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)], device = device)
        secondToLast_decoder_hidden = secondToLast_decoder_hidden.reshape([batch_size, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        decoder_hidden[-2] = secondToLast_decoder_hidden
        decoder_input_ids = decoder_input_ids
        past = decoder_hidden

        ## DEBUGGING
        # logger.info("decoder_input_ids: " + str(decoder_input_ids.shape))
        # logger.info("past: " + str(past.shape))
        # logger.info("decoder_attention_mask: " + str(decoder_attention_mask.shape))
        
                
        # decoder forward pass
        decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = decoder_input_ids, past = past, attention_mask = decoder_attention_mask)

        
        return decoder_lm_logits 

"""====================== METHODS DEFINITIONS ======================"""

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits





