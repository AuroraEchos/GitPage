import torch
import torchvision
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)

        return torchvision.ops.sigmoid_focal_loss(
            inputs, targets.float(),
            alpha=self.alpha, gamma=self.gamma,
            reduction=self.reduction
        )
    
if __name__ == "__main__":
    inputs = torch.randn(3, 5, requires_grad=True)
    targets = torch.empty(3, dtype=torch.long).random_(5)

    criterion = FocalLoss(num_classes=5)
    loss = criterion(inputs, targets)
    print(loss)