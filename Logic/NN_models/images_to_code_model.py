import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json


class SpecialTokenTokenizerWrapper:
    def __init__(self, raw_tokenizer, pad_token="<pad>", sos_token="[SOS]", eos_token="[EOS]"):
        self.raw_tokenizer = raw_tokenizer

        # Start with the raw tokenizer's vocabulary
        # For a real HF tokenizer, this would be `raw_tokenizer.vocab`
        self.vocab = self.raw_tokenizer.vocab.copy()

        # Define special tokens and their IDs, appending to the raw vocabulary
        self.special_tokens_map = {
            pad_token: "pad_token",
            sos_token: "bos_token",  # Using HF's common 'beginning of sentence'
            eos_token: "eos_token"  # Using HF's common 'end of sentence'
        }

        # We need to assign new unique IDs for special tokens if they are not already in raw_tokenizer.
        # A common practice is to pick IDs that are not yet in use.
        # Let's ensure our special token IDs are beyond the existing raw_tokenizer's vocab size.
        next_id = len(self.vocab)  # Start assigning IDs from here

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Assign IDs to special tokens. If they exist in raw_tokenizer.vocab, use their existing ID.
        # Otherwise, assign a new ID.
        self.pad_token_id = self.vocab.get(pad_token, next_id)
        if pad_token not in self.vocab:
            self.vocab[pad_token] = self.pad_token_id
            next_id += 1

        self.sos_token_id = self.vocab.get(sos_token, next_id)
        if sos_token not in self.vocab:
            self.vocab[sos_token] = self.sos_token_id
            next_id += 1

        self.eos_token_id = self.vocab.get(eos_token, next_id)
        if eos_token not in self.vocab:
            self.vocab[eos_token] = self.eos_token_id
            next_id += 1

        # Build the id_to_token mapping from the unified vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self._vocab_size = len(self.vocab)  # Total vocab size including special tokens

    def encode(self, text, add_special_tokens=True):
        """
        Encodes text, optionally adding SOS and EOS tokens.
        Returns a list of token IDs.
        """
        # The `raw_tokenizer.encode` method should be able to handle this.
        # For a real HF tokenizer, you might want to ensure it doesn't add its own special tokens here.
        raw_token_ids = self.raw_tokenizer.encode(text)

        if add_special_tokens:
            return [self.sos_token_id] + raw_token_ids + [self.eos_token_id]
        return raw_token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decodes a list of token IDs back to text.
        Optionally skips SOS, EOS, and PAD tokens.
        """
        filtered_ids = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.sos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            filtered_ids.append(token_id)

        # The raw tokenizer should be able to decode these IDs, including the ones
        # we added, as long as its internal `id_to_token` mapping is correctly updated,
        # or we use our wrapper's `id_to_token`.

        # For our mock YourActualHFTokenizer, we'll use our wrapper's mapping directly.
        words = [self.id_to_token.get(id, "<unk>") for id in filtered_ids]
        return "".join(words)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def pad_id(self):
        return self.pad_token_id

    @property
    def sos_id(self):
        return self.sos_token_id

    @property
    def eos_id(self):
        return self.eos_token_id


class ImageToCodeDecoder(nn.Module):
    def __init__(self,
                 embedder,
                 tokenizer,
                 vocab_size,
                 image_emb_dim,
                 model_dim=768,  # Adjusted
                 num_layers=12,  # Adjusted
                 num_heads=12,  # Adjusted
                 max_seq_len=256,
                 pad_token_id=0,
                 sos_token_id=1,
                 eos_token_id=2):
        super().__init__()

        self.embedder = embedder
        # It's good practice to set embedder to eval mode if it's frozen
        # and you don't intend to train it.
        # self.embedder.eval() # Uncomment if you want to freeze it

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        # Project image embedding to match transformer model dim
        self.image_proj = nn.Linear(image_emb_dim, model_dim)

        # Token embeddings and positional encodings
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_token_id)
        # Using a learned positional embedding as it's common.
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)

        # Transformer decoder
        # FFN dimension is usually 4 * model_dim by default in TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(model_dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with 0 on diagonal."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, tgt_tokens, tgt_padding_mask=None):
        """
        images: (B, C, H, W) - Input images
        tgt_tokens: (B, T) - Target token indices (input to decoder, e.g., shifted ground truth during training)
        tgt_padding_mask: (B, T) - Mask for padded elements in tgt_tokens (True where token is pad)
        """
        device = images.device

        # Step 1: Get image embeddings
        # Assuming the embedder is pre-trained and frozen for this forward pass.
        # If you want to train it, remove torch.no_grad().
        with torch.no_grad():
            img_emb = self.embedder(images)  # (B, image_emb_dim)
        memory = self.image_proj(img_emb).unsqueeze(1)  # (B, 1, model_dim) - memory for transformer decoder

        # Step 2: Token embedding + pos embedding
        # Create positional IDs from 0 to T-1
        pos_ids = torch.arange(tgt_tokens.shape[1], device=device).unsqueeze(0).repeat(tgt_tokens.shape[0], 1)
        token_emb = self.token_embedding(tgt_tokens)  # (B, T, model_dim)
        pos_emb = self.pos_embedding(pos_ids)  # (B, T, model_dim)
        token_emb = token_emb + pos_emb  # (B, T, model_dim)

        # Step 3: Generate look-ahead mask and combined mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_tokens.shape[1]).to(device)

        # Step 4: Decode
        # The `nn.TransformerDecoder` expects `batch_first=True` now.
        output = self.decoder(tgt=token_emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)

        # Step 5: Output projection
        logits = self.output_proj(output)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images, max_new_tokens=256, temperature=1.0, top_k=None, top_p=None, greedy=False):
        """
        Generates code tokens autoregressively from input images.
        images: (B, C, H, W) - Input images
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature for sampling (ignored if greedy=True).
        top_k: If not None, sample from the top_k most probable tokens (ignored if greedy=True).
        top_p: If not None, sample from the smallest set of tokens whose cumulative probability exceeds top_p (ignored if greedy=True).
        greedy: If True, always pick the most likely next token (disables temperature, top_k, top_p).
        """
        device = images.device
        batch_size = images.shape[0]

        # Get image embeddings (memory for transformer decoder)
        img_emb = self.embedder(images)
        memory = self.image_proj(img_emb).unsqueeze(1)  # (B, 1, model_dim)

        # Initialize the generated sequence with SOS tokens for each item in the batch
        generated_tokens = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)

        for i in range(max_new_tokens):
            # Get current target sequence (including SOS and previously generated tokens)
            current_tgt_tokens = generated_tokens
            current_tgt_len = current_tgt_tokens.shape[1]

            # Enforce max_seq_len during generation to avoid OOM or excessively long sequences
            if current_tgt_len > self.max_seq_len:
                current_tgt_tokens = current_tgt_tokens[:, -self.max_seq_len:]
                current_tgt_len = self.max_seq_len
                # Adjust pos_ids if we're truncating, so they always start from 0
                pos_ids = torch.arange(current_tgt_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            else:
                pos_ids = torch.arange(current_tgt_len, device=device).unsqueeze(0).repeat(batch_size, 1)

            # Create look-ahead mask for the current target length
            tgt_mask = self._generate_square_subsequent_mask(current_tgt_len).to(device)

            # Get token embeddings and positional embeddings for the current sequence
            token_emb = self.token_embedding(current_tgt_tokens)
            pos_emb = self.pos_embedding(pos_ids)
            token_emb = token_emb + pos_emb

            # Pass through decoder
            output = self.decoder(tgt=token_emb, memory=memory, tgt_mask=tgt_mask)

            # Get logits for the last token in the sequence
            logits = self.output_proj(output[:, -1, :])  # (B, vocab_size)

            if greedy: # ADD THIS BLOCK
                # For greedy decoding, just take the argmax
                next_token = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
            else: # EXISTING SAMPLING LOGIC
                # Apply temperature
                logits = logits / temperature

                # Apply top-k and top-p sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))

                # Sample the next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the next token to the generated sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # Check if all sequences have generated an EOS token
            if (next_token == self.eos_token_id).all():
                break

        # Remove SOS token and any tokens after EOS for each sequence
        final_sequences = []
        for i in range(batch_size):
            seq = generated_tokens[i].tolist()
            # Find the first EOS token
            try:
                eos_idx = seq.index(self.eos_token_id)
                seq = seq[1:eos_idx]  # Exclude SOS and everything after EOS
            except ValueError:
                seq = seq[1:]  # No EOS found, just exclude SOS
            final_sequences.append(seq)

        # Convert token IDs back to text using the tokenizer
        decoded_texts = [self.tokenizer.decode(seq) for seq in final_sequences]
        return decoded_texts

    def save_state_dict(self, path):
        """Saves the model's state_dict."""
        torch.save(self.state_dict(), path)
        print(f"Model state_dict saved to {path}")

    def load_state_dict_from_file(self, path, device='cpu'):
        """Loads the model's state_dict from a file."""
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model state_dict loaded from {path}")