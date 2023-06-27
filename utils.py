import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from jamo import j2hcj, h2j

class FontDataset(Dataset):
    def __init__(self, font_source, font_target, image_list=[chr(i) for i in range(44032, 55204)], image_transform=transforms.Compose([transforms.ToTensor()])):
        super(FontDataset).__init__()
        self.image_transform=image_transform
        self.images=image_list
        self.image_path = "dataset/"+font_source
        self.target_path = "dataset/"+font_target
        token = set(["None"])
        self.left_set = set(['ㅣ', 'ㅐ', 'ㅔ', 'ㅏ', 'ㅕ', 'ㅖ', 'ㅑ', 'ㅒ', 'ㅓ'])
        self.top_set = set(['ㅛ', 'ㅜ', 'ㅡ', 'ㅗ', 'ㅠ'])
        self.tl_set = set(['ㅚ', 'ㅙ', 'ㅞ', 'ㅟ', 'ㅝ', 'ㅢ', 'ㅘ'])
        for idx in range(44032, 55204):
            a = list(j2hcj(h2j(chr(idx))))
            if a[1] in self.left_set:
                a[0] += 'l'
            elif a[1] in self.top_set:
                a[0] += 't'
            elif a[1] in self.tl_set:
                a[0] += 'tl'
            if len(a) == 3:
                a[0] += '3'
                a[1] += '3'
            token.update(a)
        self.token = {j:i for i, j in enumerate(token)}

    def __getitem__(self, idx):
        image = Image.open(self.image_path+'/'+self.images[idx]+'.png')
        image_tensor = self.image_transform(image)
        image.close()
        target = Image.open(self.target_path+'/'+self.images[idx]+'.png')
        target_tensor = self.image_transform(target)
        target.close()
        image_name = list(j2hcj(h2j(self.images[idx])))
        if image_name[1] in self.left_set:
            image_name[0] += 'l'
        elif image_name[1] in self.top_set:
            image_name[0] += 't'
        elif image_name[1] in self.tl_set:
            image_name[0] += 'tl'
        if len(image_name) != 3:
            image_name.append("None")
        else:
            image_name[0] += '3'
            image_name[1] += '3'
        return image_tensor, target_tensor, torch.tensor([self.token[image_name[0]], self.token[image_name[1]], self.token[image_name[2]]]), self.images[idx]
    
    def __len__(self):
        return len(self.images)
    