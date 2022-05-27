from multiprocessing.sharedctypes import Value
from PIL import Image
import os
from sys import platform
import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio

import signatory

from tqdm import tqdm
import sklearn.metrics as metrics


def listdir(main_path):
    if platform in ['darwin', 'linux']:
        return [path for path in os.listdir(main_path) if not path.startswith('.')]
    else:
        return os.listdir(main_path)


class FolderDataset(Dataset):
    def __init__(self, 
                 path, 
                 N,
                 image_sig_type='row',
                 data_transforms={'.jpg': None, '.mp4': None, '.wav': None},
                 data_augmentation={'.jpg': None, '.mp4': None, '.wav': None},
                 augment_by={'.jpg': None, '.mp4': None, '.wav': None},
                 data_types = None,
                 **kwargs):
        """
        :param path: where files are stored in class folders
        :param data_transforms: dictionary of transformations to apply to each data type 
               (video transforms are applied frame-wise, audio transforms are applied to the spectrogram)
        :param data_augmentation: dictionary of augmentations to apply to each data type
        :param augment_by: dictionary of augmentation size to apply to each data type
        """
        self.path = path
        self.path_classes = listdir(path)
        self.path_classes.sort()
        self.data_transforms = data_transforms
        self.data_augmentation = data_augmentation
        self.augment_by = augment_by
        if data_types is None:
            data_types = ('.jpg', '.mp4', '.wav')
        self.data_types = data_types

        self.labels = []
        self.data_type = []
        self.tensors = []

        self.sig_tensors = []
        # loop on classes
        for i, class_ in enumerate(tqdm(self.path_classes)):
            for file in tqdm(listdir(os.path.join(self.path, class_))):
                path_to_file = os.path.join(self.path, class_, file)
                _, ext = os.path.splitext(path_to_file)
                if ext in self.data_types:
                    if ext == '.jpg':
                        image = Image.open(path_to_file)
                        if self.data_transforms['.jpg'] is not None:
                            image = self.data_transforms['.jpg'](image)
                        if self.data_augmentation['.jpg'] is not None:
                            aug_images = [image] + [self.data_augmentation['.jpg'](image) for _ in range(augment_by['.jpg'])]
                        else:
                            aug_images = [image]
                        for image in aug_images:
                            image = transforms.ToTensor()(image)
                            self.tensors.append(image)
                            self.labels.append(i)
                            self.data_type.append(ext)

                            im_signature = image_signature(image, N=N, image_sig_type=image_sig_type, kwargs=kwargs)
                            im_signature = torch.unsqueeze(im_signature, 0)
                            self.sig_tensors.append(im_signature)


                    elif ext == '.mp4':
                        vidcap = cv2.VideoCapture(path_to_file)
                        success, image = vidcap.read()
                        frames = []
                        while success:
                            image = Image.fromarray(image)
                            # apply transform to image
                            if self.data_transforms['.mp4'] is not None:
                                image = self.data_transforms['.mp4'](image)
                            image = transforms.ToTensor()(image)
                            frames.append(image)
                            success, image = vidcap.read()
                        # create a tensor of size [frames, channels, height, width]
                        video = torch.stack(frames)
                        if self.data_augmentation['.mp4'] is not None:
                            raise ValueError('Video augmentation not yet supported')
                        else:
                            aug_videos = [video]
                        
                        for video in aug_videos:
                            self.tensors.append(video)
                            self.labels.append(i)
                            self.data_type.append(ext)

                            video_signatures = []
                            for j in range(len(video)):
                                frame = video[j, ...]
                                frame_signature = image_signature(frame, N=N, image_sig_type=image_sig_type, kwargs=kwargs)
                                video_signatures.append(frame_signature)
                            video_signature = torch.stack(video_signatures)
                            self.sig_tensors.append(video_signature)
                        
                    elif ext == '.wav':
                        audio, sample_rate = torchaudio.load(path_to_file)

                        audio = rechannel(audio)
                        audio = resample(audio, sample_rate)
                        audio = mel_spectrogram(audio)

                        if self.data_transforms['.wav'] is not None:
                            audio = self.data_transforms['.wav'](audio)

                        if self.data_augmentation['.wav'] is not None:
                            aug_audios = [audio] + [self.data_augmentation['.wav'](audio) for _ in range(augment_by['.wav'])]
                        else:
                            aug_audios = [audio]
                        
                        for audio in aug_audios:
                            self.tensors.append(audio)
                            self.labels.append(i)
                            self.data_type.append(ext)

                            audio_signature = image_signature(audio, N=N, image_sig_type=image_sig_type, kwargs=kwargs)
                            audio_signature = torch.unsqueeze(audio_signature, 0)
                            self.sig_tensors.append(audio_signature)
                    else:
                        raise ValueError(ext, 'not supported')

        self.data_type_labels = [self.data_types.index(ext) for ext in self.data_type]        
    
    def __len__(self):
        return len(self.sig_tensors)

    def __getitem__(self, idx):
        return self.sig_tensors[idx], (self.labels[idx], self.data_type_labels[idx])


####################################### Auxiliary functions ######################################

def rechannel(audio, n_channels=3, triaural = 'average'):
    """
    Augment the audio data source with the desired number of channels.
    """
    # If audio is correct shape do nothing
    if audio.shape[0] == n_channels:
        return audio
    # Turn Mono Audio into stereo by repeating the data channel
    elif audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    # Add third channel
    if triaural == 'zeros':
        return torch.nn.functional.pad(
            audio, pad=(0, 0, 0, 1), mode='constant', value=0
        )
    elif triaural == 'average':
        return torch.cat(
            [audio, torch.mean(audio, dim=0).view(1, audio.size()[1])]
        )
    elif triaural == 'first':
        return torch.cat(
            [audio, audio[0, :].view(1, audio.size()[1])]
        )
    else:
        raise ValueError('triaural must be either zeros, average or first')

def resample(audio, old_sr, sr=44100):
    """
    Resample audio to a goal sample rate.
    """
    if old_sr == sr:
        return audio
    else:
        return torchaudio.transforms.Resample(old_sr, sr)(audio)

def mel_spectrogram(audio, sr=44100, n_mels=64, n_fft=1024, hop_length=None):
    """
    Compute spectrogram for an audio sample. Power is converted to dB
    """
    transformer = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, normalized=True
        ),
        torchaudio.transforms.AmplitudeToDB()
    )
    return transformer(audio)


def image_signature(image, N, image_sig_type="row", kwargs=None):
    if image_sig_type == "row":
        # from [channels, height, width] to [height, width, channels]
        paths = torch.permute(image, (1, 2, 0))
        # for each row in frames*height compute the signature of the path of length width in R^channels
        row_signatures = signatory.signature(paths, N)
        # average out the signatures
        signature_output = torch.mean(row_signatures, 0)

    elif image_sig_type == "frame_rows":
        (C, H, W) = image.shape
        image = image.numpy()
        flattened_image = np.array(image[:, 0, :])
        for j in range(1, H):
            if j % 2 == 0:
                flattened_image = np.append(flattened_image, image[:, j, :], axis=1)
            else:
                flattened_image = np.append(flattened_image, image[:, j, ::-1], axis=1)
        path = torch.tensor(flattened_image.transpose())
        signature_output = signatory.signature(path.unsqueeze(0), N).squeeze()

    elif image_sig_type == "frame_columns":
        (C, H, W) = image.shape
        image = image.numpy()
        flattened_image = np.array(image[:, :, 0])
        for j in range(1, W):
            if j % 2 == 0:
                flattened_image = np.append(flattened_image, image[:, :, j], axis=1)
            else:
                flattened_image = np.append(flattened_image, image[:, ::-1, j], axis=1)
        path = torch.tensor(flattened_image.transpose())
        signature_output = signatory.signature(path.unsqueeze(0), N).squeeze()

    elif image_sig_type == "spiral":
        # Would be better if pure pytorch but c'est la vie
        (C, H, W) = image.shape
        image = image.numpy()
        out = []
        while image.size:
            out.append(image[:, 0, :])
            image = np.rot90(image[:, 1:, :], axes=(1, 2))
        path = np.concatenate(out, axis=1).transpose()
        signature_output = signatory.signature(torch.Tensor(path).unsqueeze(0),N).squeeze()

    elif image_sig_type == "patch_spiral":
        (C, H, W) = image.shape
        stride = kwargs["stride"]
        patch_size = kwargs["patch_size"]
        patch_signatures = []
        for h in range(0, H - patch_size[0], stride):
            for w in range(0, W - patch_size[0], stride):
                patch = image[:, h : (h + patch_size[0]), w : (w + patch_size[1])]
                patch_signatures.append(
                    image_signature(patch, N=N, image_sig_type="spiral")
                )
        signature_output = torch.stack(patch_signatures).mean(dim=0)
    
    elif image_sig_type == "patch_snake":
        (C, H, W) = image.shape
        stride = kwargs["stride"]
        patch_size = kwargs["patch_size"]
        patch_signatures = []
        for h in range(0, H - patch_size[0], stride):
            for w in range(0, W - patch_size[0], stride):
                patch = image[:, h : (h + patch_size[0]), w : (w + patch_size[1])]
                patch_signatures.append(
                    image_signature(patch, N=N, image_sig_type="frame_rows")
                )
        signature_output = torch.stack(patch_signatures).mean(dim=0)
    
    elif image_sig_type == "patch_row":
        (C, H, W) = image.shape
        stride = kwargs["stride"]
        patch_size = kwargs["patch_size"]
        patch_signatures = []
        for h in range(0, H - patch_size[0], stride):
            for w in range(0, W - patch_size[0], stride):
                patch = image[:, h : (h + patch_size[0]), w : (w + patch_size[1])]
                patch_signatures.append(
                    image_signature(patch, N=N, 
                    image_sig_type="row")
                )
        signature_output = torch.stack(patch_signatures).mean(dim=0)

    else:
        raise ValueError(
            "image_sig_type must be either row, frame_rows, frame_columns, spiral, patch_spiral or patch_snake"
        )

    return signature_output


#####################################################################################################
#################################### Multi-task networks ############################################
#####################################################################################################

class ClassificationBase_multitask(nn.Module):
    def __init__(self, device, weight_1, weight_2, loss_ratio):
        super().__init__()
        self.device = device
        self.task1_weights = torch.tensor(weight_1).float().to(device)
        self.task2_weights = torch.tensor(weight_2).float().to(device)
        self.loss_ratio = loss_ratio

    def training_step(self, batch):
        samples, labels_1, labels_2, samples_len = batch[0].to(self.device), batch[1][:, 0].to(self.device), batch[1][:, 1].to(self.device), batch[2]
        out_1, out_2 = self(samples.float(), samples_len)
        loss_1 = nn.NLLLoss(weight=self.task1_weights)(out_1, labels_1)
        loss_2 = nn.NLLLoss(weight=self.task2_weights)(out_2, labels_2)
        loss = loss_1 + self.loss_ratio*loss_2
        acc_1 = accuracy(out_1, labels_1)
        acc_2 = accuracy(out_2, labels_2)
        return loss, (acc_1, acc_2)
    
    def validation_step(self, batch):
        samples, labels_1, labels_2, samples_len = batch[0].to(self.device), batch[1][:, 0].to(self.device), batch[1][:, 1].to(self.device), batch[2]
        out_1, out_2 = self(samples.float(), samples_len)
        loss_1 = nn.NLLLoss(weight=self.task1_weights)(out_1, labels_1)
        loss_2 = nn.NLLLoss(weight=self.task2_weights)(out_2, labels_2)
        loss = loss_1 + self.loss_ratio*loss_2
        acc_1 = accuracy(out_1, labels_1)
        acc_2 = accuracy(out_2, labels_2)
        return {'Loss': loss.detach(), 'Accuracy Task 1': acc_1, 'Accuracy Task 2': acc_2}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['Loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs_1 = [x['Accuracy Task 1'] for x in outputs]
        batch_accs_2 = [x['Accuracy Task 2'] for x in outputs]
        epoch_acc_1 = torch.stack(batch_accs_1).mean()
        epoch_acc_2 = torch.stack(batch_accs_2).mean()
        return {'Validation Loss': epoch_loss.item(), 'Validation Accuracy Task 1': epoch_acc_1.item(), 'Validation Accuracy Task 2': epoch_acc_2.item()}
    
    def training_epoch_end(self, outputs):
        batch_losses = [x['Loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs_1 = [x['Accuracy Task 1'] for x in outputs]
        batch_accs_2 = [x['Accuracy Task 2'] for x in outputs]
        epoch_acc_1 = torch.stack(batch_accs_1).mean()
        epoch_acc_2 = torch.stack(batch_accs_2).mean()
        return {'Training Loss': epoch_loss.item(), 'Training Accuracy Task 1': epoch_acc_1.item(), 'Training Accuracy Task 2': epoch_acc_2.item()}

    def epoch_end(self, epoch, result):
        print("Epoch :", epoch + 1)
        print(f'Training Accuracy Task 1:{result["Training Accuracy Task 1"]*100:.2f}% Training Accuracy Task 2:{result["Training Accuracy Task 2"]*100:.2f}% Validation Accuracy Task 1:{result["Validation Accuracy Task 1"]*100:.2f}% Validation Accuracy Task 2:{result["Validation Accuracy Task 2"]*100:.2f}%')
        print(f'Training Loss:{result["Training Loss"]:.4f} Validation Loss:{result["Validation Loss"]:.4f}')


class Net_multitask(ClassificationBase_multitask):
    def __init__(self, device, N, input_dim = 120, output_dim1 = 3, output_dim2 = 3, weight_1=[0, 0, 0], weight_2=[1, 1, 1], loss_ratio=0.5, encoder_type='FC'):
        super().__init__(device, weight_1, weight_2, loss_ratio)
        self.N = N
        self.encoder_type = encoder_type
        if self.encoder_type == 'FC':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Linear(50, 3),
                nn.ReLU()
            )
        elif self.encoder_type == 'Conv':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 256, 3),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.MaxPool1d(3),
                nn.Conv1d(256, 128, 3),
                nn.BatchNorm1d(128),
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.MaxPool1d(3),
                nn.Conv1d(128, 64, 3),
                nn.BatchNorm1d(64),
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.MaxPool1d(3),
                nn.Flatten(),
                nn.Linear(64*int((((input_dim-2)/3-2)/3-2)/3), 50),
                nn.LeakyReLU(),
                nn.Linear(50, 3)
            )
        else:
            raise ValueError('encoder_type must be either FC or Conv.')
        # self.decoder1 = nn.Linear(3, output_dim1)
        # self.decoder2 = nn.Linear(3, output_dim2)
        self.decoder1 = nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim1),
                nn.ReLU()
            )
        self.decoder2 = nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(),
                nn.Linear(50, output_dim2),
                nn.ReLU()
            )

    def forward(self, batch, samples_len):
        # batch has dimensions [batch_size, max_length_video, sig_dim]
        output = torch.zeros_like(batch[:, 0, :])
        # videos has dimension [n_videos, max_length_video, sig_dim] 
        videos = batch[samples_len > 1, :, :]
        if videos.shape[0] > 0:
            # videos has dimension [n_videos * max_length_video, sig_dim] 
            videos = torch.flatten(videos, end_dim=1)
            # videos has dimension [length_video_1 + ... + length_video_n, sig_dim] 
            videos = videos[~torch.any(videos.isnan(), dim=1)]
            # videos_latent has dimension [length_video_1 + ... + length_video_n, 3] 
            if self.encoder_type == 'FC':
                videos_latent = self.encoder(videos)
            elif self.encoder_type == 'Conv':
                videos_latent = self.encoder(videos.unsqueeze(1))
            tot = 0 
            signatures = []
            for length_video in samples_len[samples_len>1]:
                # computing signature of [1, length_video, 3] path, has dim [sig_dim]
                signature = signatory.signature(videos_latent[tot:tot+length_video].unsqueeze(0), depth=self.N)
                signatures.append(signature)
                tot += length_video
            # videos_sig has dimension [n_videos, sig_dim]
            videos_sig = torch.stack(signatures)
            # output has shape [batch_size, sig_dim]
            output[samples_len == 1] = batch[samples_len==1, 0, :]
            output[samples_len > 1] = videos_sig.squeeze()
        else:
            # output has shape [batch_size, sig_dim]
            output = batch[:, 0, :]
        # send to latent space
        if self.encoder_type == 'FC':
            output = self.encoder(output)
        elif self.encoder_type == 'Conv':
            output = self.encoder(output.unsqueeze(1))
        # add a FC layer to perform the two tasks
        output1 = self.decoder1(output)
        output2 = self.decoder2(output)
        # send to output
        output1 = F.log_softmax(output1, dim=-1)
        output2 = F.log_softmax(output2, dim=-1)
        return output1, output2
        

def fit_multitask(model, train_loader, val_loader, epochs=10, learning_rate=0.001, early_stopping=5, name = False):
    if not name:
        print('Warning: weights not being saved. Add name')
    model.to(model.device)
    best_valid = None
    count = 0
    history = []
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.0005)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_accuracy_1 = []
        train_accuracy_2 = []
        for batch in tqdm(train_loader):
            loss, (acc_1, acc_2) = model.training_step(batch)
            train_losses.append(loss)
            train_accuracy_1.append(acc_1)
            train_accuracy_2.append(acc_2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['Training Loss'] = torch.stack(train_losses).mean().item()
        result['Training Accuracy Task 1'] = torch.stack(train_accuracy_1).mean().item()
        result['Training Accuracy Task 2'] = torch.stack(train_accuracy_2).mean().item()
        model.epoch_end(epoch, result)
        os.makedirs('./model_weights', exist_ok = True)
        if(best_valid == None or best_valid>result['Validation Loss']):
            count = 0
            best_valid=result['Validation Loss']
            if name:
                torch.save(model.state_dict(), './model_weights/' + name + '.pth')
        else:
            count += 1
        if count == early_stopping:
            break
        history.append(result)
    return history


def prediction_accuracy_multitask(model, test_loader, device):
    correct_1 = 0
    total_1 = 0
    correct_2 = 0
    total_2 = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            # send to GPU if available
            samples, labels_1, labels_2, samples_len = batch[0].to(device), batch[1][:, 0].to(device), batch[1][:, 1].to(device), batch[2]
            # calculate outputs by running images through the network
            out_1, out_2 = model(samples.float(), samples_len)
            # the class with the highest energy is what we choose as prediction
            _, pred_1 = torch.max(out_1.data, 1)
            _, pred_2 = torch.max(out_2.data, 1)
            total_1 += labels_1.size(0)
            total_2 += labels_2.size(0)
            correct_1 += (pred_1 == labels_1).sum().item()
            correct_2 += (pred_2 == labels_2).sum().item()
    return {"Accuracy Task 1": correct_1 / total_1, "Accuracy Task 2": correct_2 / total_2}


def confusion_matrix_multitask(model, test_loader, device):
    # Initialize the prediction and label lists(tensors)
    pred_list_1 = torch.zeros(0, dtype=torch.long, device='cpu')
    label_list_1 = torch.zeros(0, dtype=torch.long, device='cpu')
    pred_list_2 = torch.zeros(0, dtype=torch.long, device='cpu')
    label_list_2 = torch.zeros(0, dtype=torch.long, device='cpu')

    # again no gradients needed
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            # send to GPU if available
            samples, labels_1, labels_2, samples_len = batch[0].to(device), batch[1][:, 0].to(device), batch[1][:, 1].to(device), batch[2]
            # calculate outputs by running images through the network
            out_1, out_2 = model(samples.float(), samples_len)
            # the class with the highest energy is what we choose as prediction
            _, pred_1 = torch.max(out_1.data, 1)
            _, pred_2 = torch.max(out_2.data, 1)

            # collect the correct predictions for each class
            # Append batch prediction results
            pred_list_1 = torch.cat([pred_list_1, pred_1.view(-1).cpu()])
            label_list_1 = torch.cat([label_list_1, labels_1.view(-1).cpu()])
            pred_list_2 = torch.cat([pred_list_2, pred_2.view(-1).cpu()])
            label_list_2 = torch.cat([label_list_2, labels_2.view(-1).cpu()])
        
    conf_mat_1 = metrics.confusion_matrix(label_list_1.numpy(), pred_list_1.numpy())
    conf_mat_2 = metrics.confusion_matrix(label_list_2.numpy(), pred_list_2.numpy())
    return {"Confusion Matrix Task 1": conf_mat_1, "Confusion Matrix Task 2": conf_mat_2}


@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in data_loader]
    return model.validation_epoch_end(outputs)
