import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()
    

class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class SEBlock(nn.Module):
    def __init__(self, num_classes, reduction=16):
        """
        Squeeze-and-Excitation Block for 1D tensors (e.g., class scores).

        Args:
            num_classes (int): Number of classes (input and output dimensions).
            reduction (int): Reduction ratio for the bottleneck.
        """
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(num_classes, num_classes // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_classes // reduction, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SE block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_classes).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Squeeze: Global Information Embedding (since already global, directly apply FC layers)
        se = self.fc1(x)          # Shape: (batch_size, num_classes // reduction)
        se = self.relu(se)        # Shape: (batch_size, num_classes // reduction)
        se = self.fc2(se)         # Shape: (batch_size, num_classes)
        se = self.sigmoid(se)     # Shape: (batch_size, num_classes)
        # Excitation: Scale the input by the SE weights
        return x * se             # Shape: (batch_size, num_classes)

class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        # return F.linear(x, weight) * self.scale
        return F.linear(x, weight, self.bias) * self.scale
        # if 
        #     return F.linear(x, weight, self.bias) * self.scale
        # else:
            # return F.linear(x, weight) * self.scale
    
    
    # def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, reduction=16, **kwargs):
    #     """
    #     Cosine-based Classifier with Residual Connection using SE Block.

    #     Args:
    #         feat_dim (int): Dimension of the input features.
    #         num_classes (int): Number of classes.
    #         dtype (torch.dtype): Data type of the parameters.
    #         scale (float): Scaling factor for the cosine similarity.
    #         reduction (int): Reduction ratio for the SE block.
    #         **kwargs: Additional keyword arguments.
    #     """
    #     super().__init__(feat_dim, num_classes, dtype)
    #     self.scale = scale
    #     self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))
    #     self.se_block = SEBlock(num_classes, reduction)

    # def forward(self, x):
    #     """
    #     Forward pass of the CosineClassifier with Residual SE Connection.

    #     Args:
    #         x (Tensor): Input features of shape (batch_size, feat_dim).

    #     Returns:
    #         Tensor: Output logits of shape (batch_size, num_classes).
    #     """
    #     # Normalize input features
    #     x_norm = F.normalize(x, dim=-1)               # Shape: (batch_size, feat_dim)
    #     # Normalize weights
    #     weight_norm = F.normalize(self.weight, dim=-1)  # Shape: (num_classes, feat_dim)
    #     # Main path: Compute cosine similarity and scale
    #     main_output = F.linear(x_norm, weight_norm) * self.scale  # Shape: (batch_size, num_classes)
    #     # Residual path: Apply SE Block to main output
    #     residual = self.se_block(main_output)       # Shape: (batch_size, num_classes)
    #     # Combine main output with residual
    #     combined_output = main_output + residual    # Shape: (batch_size, num_classes)
    #     return combined_output


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)
