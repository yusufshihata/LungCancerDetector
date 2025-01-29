import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Show images in the dataloader
def show_imgs(dataloader: DataLoader) -> None:
    for images, labels in dataloader:
        f, axarr = plt.subplots(8, 4)
        idx = 0
        for row in range(8):
            for col in range(4):
                image = images[idx].permute(1, 2, 0).numpy()
                label = labels[idx]

                axarr[row, col].imshow(image)
                axarr[row, col].set_title(int(label))
                idx += 1
        plt.show()

        break