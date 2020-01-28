from typing import Optional, Tuple
import torch
import torchvision.models as models

class word2vec(torch.nn.Module):
    """
    word2vec class
    """
    def __init__(self, vocab_size: int, embedding_size: int = 300):
        """
        Constructor

        Args:
            vocab_size (int) - Size of the dictionary
            embedding_size (int) - Dimension of the token embedding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Forward pass

        Args:
            x (torch.Tensor) - Input tensor
        """
        x = self.embedding_layer(x)
        return x

class Model(torch.nn.Module):
    """
    Model class
    """
    def __init__(self,
                 vocab_size: int,
                 token_embedding_size: int = 300,
                 image_network_type: str = "identity",
                 caption_network_type: str = "word2vec",
                 joint_embedding_size: int = 400,
                 intent_dims: int = 7,
                 semiotic_dims: int = 3,
                 contextual_dims: int = 3):
        """
        Constructor

        Args:
            vocab_size (int) - Size of the dictionary
            token_embedding_size (int) - Dimension of the token embedding
            image_network_type
            caption_network_type
            joint_embedding_size
            intent_dims
            semiotic_dims
            contextual_dims
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_size = token_embedding_size
        self.image_network_type = image_network_type
        self.caption_network_type = caption_network_type
        self.joint_embedding_size = joint_embedding_size
        self.intent_dims = intent_dims
        self.semiotic_dims = semiotic_dims
        self.contextual_dims = contextual_dims

        if self.image_network_type == "resnet18":
            self.image_network = models.resnet18(pretrained=True)
        elif self.image_network_type == "identity":
            self.image_network = torch.nn.Identity()
        else:
            raise NotImplementedError("image_network_type = {} not available".format(self.image_network_type))

        if self.caption_network_type == "word2vec":
            self.caption_network = word2vec(self.vocab_size, self.token_embedding_size)
        elif self.caption_network_type == "ELMo":
            raise NotImplementedError
        else:
            raise NotImplementedError("caption_network_type = {} not available".format(self.caption_network_type))

        self.image_joint_embedding_layer = torch.nn.Linear(512, self.joint_embedding_size)
        self.caption_hidden_layer = torch.nn.Linear(2*self.token_embedding_size, self.token_embedding_size)
        self.caption_joint_embedding_layer = torch.nn.Linear(self.token_embedding_size, self.joint_embedding_size)

        self.intent_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.intent_dims)
        self.semiotic_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.semiotic_dims)
        self.contextual_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.contextual_dims)

    def forward(self, x_img: Optional[torch.Tensor], x_caption: Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # pylint: disable=arguments-differ
        """
        Forward pass

        Args:
            x_img (Optional[torch.Tensor]) - Input image tensor
            x_caption (Optional[torch.Tensor]) - Input caption tensor
        """
        x_img = self.image_network(x_img) # (batch_size, img_embedding_dim)
        x_caption = self.caption_network(x_caption) # (batch_size, caption_length, word_embedding_dim)

        # Iterate over the caption
        caption_length = x_caption.size(1)
        hidden = torch.zeros(x_caption.size(0), self.token_embedding_size)
        for t in range(caption_length):
            x_caption_combined = torch.cat((hidden, x_caption[:, t, :]), dim=1)
            hidden = self.caption_hidden_layer(x_caption_combined)

        # Get each embedding
        x_img_joint_embedding = self.image_joint_embedding_layer(x_img)
        x_caption_joint_embedding = self.caption_joint_embedding_layer(hidden)

        # Fusion
        x_fusion = x_img_joint_embedding + x_caption_joint_embedding

        # Prediction
        pred_intent = self.intent_prediction_layer(x_fusion)
        pred_semiotic = self.semiotic_prediction_layer(x_fusion)
        pred_contextual = self.contextual_prediction_layer(x_fusion)

        output = {
            'intent': pred_intent,
            'semiotic': pred_semiotic,
            'contextual': pred_contextual,
        }
        return output

if __name__ == '__main__':
    model = Model(vocab_size=100)
