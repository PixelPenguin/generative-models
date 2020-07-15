from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 本当はcfgに書くべき
batch_size = 64
# Datasetを作成
mean = (0.5,)
std = (0.5,)


def dataloader():
    dataloader_dict = dict()
    for phase in ['train', 'test']:
        dataloader_dict[phase] = DataLoader(
            GANDataset(
                file_list=make_datapath_list(phase),
                transform=ImageTransform(mean, std)
            ),
            batch_size=batch_size,
            shuffle=True
        )

    return dataloader_dict


def make_datapath_list(phase='train'):
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    img_path_list = list()  # 画像ファイルパスを格納

    if phase == 'train':
        for img_idx in range(200):
            img_path = f"/datadrive/EfficientGAN/img_78_28size/img_7_{img_idx}.jpg"
            img_path_list.append(img_path)
            img_path = f"/datadrive/EfficientGAN/img_78_28size/img_8_{img_idx}.jpg"
            img_path_list.append(img_path)

    elif phase == 'test':
        for img_idx in range(200):
            # 2とか
            pass

    return img_path_list


class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GANDataset(Dataset):
    """画像のDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''前処理をした画像のTensor形式のデータを取得'''

        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed
