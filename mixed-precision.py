# outside training loop
scaler = torch.cuda.amp.GradScaler()

# forward
with torch.cuda.amp.autocast():
    scores = model(data)
    loss = criterion(scores, targets)

# backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
