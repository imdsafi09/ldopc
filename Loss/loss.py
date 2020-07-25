import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, device, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2

        x_t = x * (2 * y - 1) + (1 - y) 

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * x_t.log()

        return loss.sum()

    def cross_entropy(self, x, y):
        return F.binary_cross_entropy(input=x, target=y, reduction='sum' )


    def forward(self, preds, targets):

        batch_size = targets.size(0)
        image_size = targets.size(1) * targets.size(2)
        cls_preds = preds[..., 0]
        loc_preds = preds[..., 1:]

        cls_targets = targets[..., 0]
        loc_targets = targets[..., 1:]
        cls_loss = self.focal_loss(cls_preds, cls_targets)


        loc_loss = torch.tensor(0.0).to(self.device)
        pos_items = cls_targets.nonzero().size(0)
        if pos_items != 0:
            for i in range(loc_targets.shape[-1]):
                loc_preds_filtered = cls_targets * loc_preds[..., i].float()
                loc_loss += F.smooth_l1_loss(loc_preds_filtered, loc_targets[..., i], reduction='sum')

            loc_loss = loc_loss / (batch_size * image_size)

        cls_loss = cls_loss / (batch_size * image_size)
        print('loc_loss: %.5f | cls_loss: %.5f' % (loc_loss.data, cls_loss.data))
        return cls_loss + loc_loss


def test():
    loss = CustomLoss()
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    loss = loss(pred, label)
    print(loss)


if __name__ == "__main__":
    test()
