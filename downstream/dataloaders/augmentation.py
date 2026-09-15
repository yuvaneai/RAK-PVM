from torchvision import transforms

def train_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        # transforms.Pad(4),
        # transforms.RandomCrop(img_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # set normalize
    return transform

def sam_train_augmentation(sam_img_size: int, mean: tuple, std: tuple, normalize: str = None):
    sam_transform = transforms.Compose([
        transforms.Resize((sam_img_size,sam_img_size)),
        # transforms.Pad(4),
        # transforms.RandomCrop(img_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # set normalize
    return sam_transform



def test_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    # set normalize
    return transform


def sam_test_augmentation(sam_img_size: int, mean: tuple, std: tuple, normalize: str = None):
    sam_transform = transforms.Compose([
        transforms.Resize((sam_img_size,sam_img_size)),
        transforms.ToTensor()
    ])

    # set normalize
    return sam_transform