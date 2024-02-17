import torch

def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss = 0.0
    model.cuda()
    model.train()
    #print(device)
    for images, _ in dataloader:
        images = images.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    return train_loss
