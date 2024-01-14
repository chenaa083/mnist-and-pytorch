from PIL import Image
import torchvision.transforms as transforms


def translate(img_path):
    # 1. 读取图像文件
    img = Image.open(img_path)
    # 2. 将图像数据转换为 PyTorch 张量
    transform = transforms.ToTensor()
    img = transform(img)
    # 3. 进行预处理（根据模型的输入要求）
    # 例如，如果模型要求输入在 [0, 1] 范围内，可以进行归一化
    return img

