from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

if __name__ == '__main__':
    log_dir = 'runs/name/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    var = np.random.random()
    for i in range(50):
        writer.add_scalar("curve", var, i)

    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('name_images', img_grid)

    # need to close the writer after training
    writer.close()
