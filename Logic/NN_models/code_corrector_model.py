import torch
from torch import nn as nn
from transformers import ModernBertModel, AutoTokenizer, ModernBertConfig, AutoModel

from Logic.NN_models.image_embedders import make_projection, load_image_embedder_for_code, make_siamese_resnet


class CodeCorrector(nn.Module):
    def __init__(
        self, bert_model: ModernBertModel, image_model, main_head_n_tokens, secondary_head_n_tokens, image_token=154
    ):
        super().__init__()
        self.bert = bert_model
        dim = bert_model.embeddings.tok_embeddings.embedding_dim
        self.image_model = image_model
        self.image_token = image_token
        self.image_projection = make_projection(input_dim=image_model[-1][-1].out_features, output_dim=dim)
        self.main_head_n_tokens = main_head_n_tokens
        self.secondary_head_n_tokens = secondary_head_n_tokens
        self.main_token_classifier = nn.Linear(dim, main_head_n_tokens)
        self.secondary_token_classifier = nn.Linear(dim, secondary_head_n_tokens)
        self.cls_classifier = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask, target_imgs, source_imgs, add_cur_images):
        """
        Forward pass of the CodeCorrector model, the input_ids should have image tokens at positions 2 and 4,
        and the target_imgs and source_imgs are the images to be embedded and inserted into those positions.
        """
        embeddings = self.bert.embeddings.tok_embeddings(input_ids)  # (batch_size, seq_len, embedding_dim)
        target_img_embeddings = self.image_projection(self.image_model(target_imgs))  # (batch_size, embedding_dim)
        source_img_embeddings = self.image_projection(self.image_model(source_imgs))  # (batch_size, embedding_dim)

        assert (input_ids[:, 2] == self.image_token).all() and (
            input_ids[:, 4] == self.image_token
        ).all(), "Image tokens must be placed in positions 2 and 4!"

        embeddings[:, 2, :] = target_img_embeddings  # insert target image embedding
        # replace only where we have those images
        embeddings[add_cur_images, 4, :] = source_img_embeddings[add_cur_images]

        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]  # (batch_size, dim)

        main_token_cls = self.main_token_classifier(last_hidden_state)
        secondary_token_cls = self.secondary_token_classifier(last_hidden_state)
        match_value = self.cls_classifier(cls_embedding)
        return match_value, main_token_cls, secondary_token_cls

    @staticmethod
    def new_model(tokenizer_path, image_path, main_head_n_tokens, secondary_head_n_tokens, image_token=154) -> "CodeCorrector":
        image_model = load_image_embedder_for_code(image_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        configuration = ModernBertConfig(
            vocab_size=len(tokenizer.get_vocab()),
            classifier_pooling="mean",
            pad_token_id=tokenizer("[PAD]")["input_ids"][0],
            bos_token_id=tokenizer("NODES:")["input_ids"][0],
            cls_token_id=tokenizer("[CLS]")["input_ids"][0],
            sep_token_id=tokenizer("[SEP]")["input_ids"][0],
        )
        base_model = ModernBertModel(configuration)
        model = CodeCorrector(base_model, image_model, main_head_n_tokens, secondary_head_n_tokens, image_token)
        return model


    def make_state_dict(self):
        return {
            "image_projection_state_dict": self.image_projection.state_dict(),
            "image_model_state_dict": self.image_model.state_dict(),
            "bert_state_dict": self.bert.state_dict(),
            "bert_config": self.bert.config,
            "main_token_classifier": self.main_token_classifier.state_dict(),
            "secondary_token_classifier": self.secondary_token_classifier.state_dict(),
            "cls_classifier_state_dict": self.cls_classifier.state_dict(),
            "heads_sizes": {
                "main_head_n_tokens": self.main_head_n_tokens,
                "secondary_head_n_tokens": self.secondary_head_n_tokens,
            },
        }

    @staticmethod
    def load(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        layer_size = (7, 256)  # that's the constant we went for, this should be saved in the checkpoint data
        image_model = make_siamese_resnet(
            layers_to_take_and_size=layer_size, final_emb=128, use_avg_pool=False, with_weights=False
        )
        bert_model = ModernBertModel(checkpoint["bert_config"])
        image_model.load_state_dict(checkpoint["image_model_state_dict"])
        bert_model.load_state_dict(checkpoint["bert_state_dict"])
        head_config = checkpoint["heads_sizes"]
        model = CodeCorrector(
            bert_model, image_model, head_config["main_head_n_tokens"], head_config["secondary_head_n_tokens"]
        )
        model.image_projection.load_state_dict(checkpoint["image_projection_state_dict"])
        model.main_token_classifier.load_state_dict(checkpoint["main_token_classifier"])
        model.secondary_token_classifier.load_state_dict(checkpoint["secondary_token_classifier"])
        model.cls_classifier.load_state_dict(checkpoint["cls_classifier_state_dict"])
        return model
