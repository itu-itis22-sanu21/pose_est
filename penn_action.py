import os
import cv2
import torch
import scipy.io
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PennActionDataset(Dataset):
    def __init__(self, annotation_dir, image_root, transform=None):
        self.annotation_dir = annotation_dir
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        for mat_file in os.listdir(annotation_dir):
            if not mat_file.endswith('.mat'):
                continue

            mat_path = os.path.join(annotation_dir, mat_file)
            mat_data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

            try:
                x = mat_data['x']
                y = mat_data['y']
                visibility = mat_data['visibility']
                nframes = int(mat_data['nframes'])
            except KeyError as e:
                print(f"Warning: Missing key {e} in {mat_file}, skipping.")
                continue

            video_name = mat_file[:-4]

            for i in range(nframes):
                img_path = os.path.join(image_root, video_name, f'{i+1:06d}.jpg')
                if not os.path.isfile(img_path):
                    continue

                joints = np.stack([x[i], y[i]], axis=1)
                vis = visibility[i].astype(np.float32)
                kps = np.concatenate((joints, vis[:, None]), axis=1)
                self.samples.append((img_path, kps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, kps = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = {'image': image, 'keypoints': kps}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PennTransform:
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
    
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = self.transform(image)

        # image = image.astype(np.float32) / 255.0
        # image = cv2.resize(image, (256, 256))
        # image = torch.from_numpy(image.transpose((2, 0, 1)))

        h, w = 256, 256
        norm_kps = keypoints.copy()
        norm_kps[:, 0] /= w
        norm_kps[:, 1] /= h

        return {
            'image': image,
            'keypoints': torch.from_numpy(norm_kps).float()
        }
