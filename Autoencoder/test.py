import torch

def test_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images,_ = images.to(device), _
            output = model(images)
            loss=loss_fn(output,images)
            valid_loss+=loss.item()*images.size(0)

    return valid_loss