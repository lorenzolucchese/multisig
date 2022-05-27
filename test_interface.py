from collections import Counter
import os
if "STY" in os.environ:
    run_name = os.environ['STY']
else:
    run_name = "non_screen"
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/full_architecture_progress_{run_name}.log"),
        logging.StreamHandler()
    ])

from pathlib import Path
import pickle
import time
import argparse
import unicodedata
import re
import warnings
warnings.filterwarnings("error") # Don't let warnings pass silently

from full_architecture import *
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchvision
import nvidia_smi

# Setup argparse
parser = argparse.ArgumentParser(description="Run Full Architecture Training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", "--N", action="append", type=int, help="Sig Depth")
parser.add_argument("-E", "--encoder-type", action="append", help="Encoder Type")
parser.add_argument("-I", "--image-sig-type", action="append", help="Image Sig Type")
parser.add_argument("-D", "--data-types", action="append", help="Data Types")
parser.add_argument("-L", "--learning-rate", action="append", type=float, help="Learning Rate")
parser.add_argument("-B", "--batch-size", action="append", type=int, help="Batch Size")
args = parser.parse_args()
config = vars(args)

path_train_val = 'data_raw/training_set'
path_test = 'data_raw/test_set'

def n_to_dim(d, n):
    """ Convert from input dimension to signature dimension. 
    
    Args:
        d: Input dimension
        n: Signature depth
    """
    return (d**(n+1)-1)/(d-1)-1

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def run_training_procedure(image_sig_type, encoder_type, N, device, learning_rate, batch_size, data_types=None):
    """
    Run the training procedure.

    Args:
        image_sig_type: Type of Signature Transform
        encoder_type: Neural Network encoder type, FC or Conv
        N: Signature Depth
    """
    patch_size = (8,8)
    stride = 4

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.CenterCrop(32)
    ])

    data_transforms = {'.jpg': image_transforms, '.mp4': image_transforms, '.wav': None}

    logging.info("Loading Dataset...")
    root_filename = slugify(f"{image_sig_type}_{N}_{data_types or ''}")
    logging.info(root_filename)
    try:
        with open(f"pickles/{root_filename}_train_dataset.pickle","rb") as handle:
            train_dataset = pickle.load(handle)
        with open(f"pickles/{root_filename}_val_dataset.pickle","rb") as handle:
            val_dataset = pickle.load(handle)
        with open(f"pickles/{root_filename}_test_dataset.pickle","rb") as handle:
            test_dataset = pickle.load(handle)
        logging.info("Dataset loaded from pickle.")
    except IOError:
        logging.info("Dataset not yet pickled, calculating.")

        train_val_dataset = FolderDataset(
            path_train_val, N=N, image_sig_type=image_sig_type, 
            patch_size=patch_size, stride=stride, 
            data_transforms=data_transforms, 
            data_types=data_types)

        test_dataset = FolderDataset(
            path_test, N=N, image_sig_type=image_sig_type, 
            patch_size=patch_size, stride=stride,
            data_transforms=data_transforms, 
            data_types=data_types)
        torch.manual_seed(123)

        train_dataset, val_dataset = random_split(train_val_dataset, [int(len(train_val_dataset)*0.8), int(len(train_val_dataset) - int(len(train_val_dataset)*0.8))])

        with open(f"pickles/{root_filename}_train_dataset.pickle","wb") as handle:
            pickle.dump(train_dataset, handle)
        with open(f"pickles/{root_filename}_val_dataset.pickle","wb") as handle:
            pickle.dump(val_dataset, handle)
        with open(f"pickles/{root_filename}_test_dataset.pickle","wb") as handle:
            pickle.dump(test_dataset, handle)
        logging.info("Dataset loaded and pickled.")

    def pad_collate(batch):
        (xx, y) = zip(*batch)
        y = torch.tensor(y)
        x_lens = np.array([len(x) for x in xx])

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=np.NAN)

        return xx_pad, y, x_lens

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=2)

    tot_samp = len(train_dataset) + len(val_dataset)
    lab_1_cnts = Counter(train_dataset.dataset.labels)
    lab_2_cnts = Counter(train_dataset.dataset.data_type_labels)
    weight_1 = []
    for k, v in lab_1_cnts.items():
        weight_1.append(tot_samp/v)
    weight_2 = []
    for k, v in lab_2_cnts.items():
        weight_2.append(tot_samp/v)
    input_dim = train_dataset[0][0].shape[1]

    logging.info("Beginning training...")
    model = Net_multitask(device, N=N, input_dim = input_dim, weight_1=weight_1, weight_2=weight_2, loss_ratio=0.5, encoder_type=encoder_type)
    history = fit_multitask(model, train_loader, val_loader, epochs=100, learning_rate=learning_rate, early_stopping=25, name = f'net_{encoder_type}_{image_sig_type}_{N}_{data_types or ""}_{learning_rate}_{batch_size}')

    pred_accuracy = prediction_accuracy_multitask(model, test_loader, device)
    logging.info("Training complete.")

    logging.info(f'Accuracy of the network on test images: Task 1 {100*pred_accuracy["Accuracy Task 1"]:.1f} %, Task 2 {100*pred_accuracy["Accuracy Task 2"]:.1f} %')

    conf_matrix = confusion_matrix_multitask(model, test_loader, device)

    logging.info(f"Confusion Matrix Task 1:\n {conf_matrix['Confusion Matrix Task 1']}")
    logging.info(f"Confusion Matrix Task 2:\n {conf_matrix['Confusion Matrix Task 2']}")

def main():
    # Ensure we're running from scripts directory
    if Path.cwd().name == "cdt_datasig_mini":
        os.chdir("./scripts")
    
    # Get Sig Dimension from User
    if config["N"] is not None:
        N = config["N"]
        logging.info(f"Selected Sig Depth(s): {N}")
    else:
        N = []
        while True:
            tmp = int(input("Enter Signature Depth (N) (Or blank to continue): "))
            if tmp == "":
                break
            else:
                N.append(tmp)
    if not N:
        N = [4]

    # Get Sig Type from User
    if config["image_sig_type"] is not None:
        image_sig_types = config["image_sig_type"]
        logging.info(f"Selected Image Sig Types: {image_sig_types}")
    else:
        image_sig_options = ["row", "frame_rows","frame_columns","spiral", "patch_spiral","patch_snake"]
        print(f"Image Sig Options: {image_sig_options}")
        image_sig_types = []
        while True:
            tmp = input("Enter Image Sig Type (Or blank to continue): ")
            if tmp == "":
                break
            else:
                image_sig_types.append(tmp)
    if not image_sig_types:
        image_sig_types = ["row"]
    
    if config["encoder_type"] is not None:
        encoder_type = config["encoder_type"]
        logging.info(f"Selected Encoder(s): {encoder_type}")
    else:
        encoder_type = []
        while True:
            tmp = input("Enter Encoder Type (Conv, FC) (or blank to continue): ")
            if tmp == "":
                break
            else:
                encoder_type.append(tmp)
    if not encoder_type:
        encoder_type = ["FC"]
    
    if config["learning_rate"] is not None:
        learning_rate = config["learning_rate"]
        logging.info(f"Selected Learning Rate(s): {learning_rate}")
    else:
        learning_rate = []
        while True:
            tmp = input("Enter Learning Rate(s) (or blank to continue): ")
            if tmp == "":
                break
            else:
                learning_rate.append(float(tmp))
    if not learning_rate:
        learning_rate = [0.001]

    if config["batch_size"] is not None:
        batch_size = config["batch_size"]
        logging.info(f"Selected batch size(s): {batch_size}")
    else:
        batch_size = []
        while True:
            tmp = input("Enter Batch Size(s) (or blank to continue): ")
            if tmp == "":
                break
            else:
                batch_size.append(int(tmp))
    if not batch_size:
        batch_size = [256]
    
    if config["data_types"] is not None:
        data_types_l = []
        for d_t in config["data_types"]:
            data_types_l.append(tuple(d_t.split("_")))
        logging.info(f"Selected Data Type(s): {data_types_l}")
    else:
        data_types_l = []
        while True:
            tmp = input("Enter Data Types (Or blank to continue): ")
            if tmp == "":
                break
            else:
                data_types_l.append(tmp)
        if data_types_l:
            data_types_l = [tuple(data_types_l)]
        else:
            data_types_l = [None]

    nvidia_smi.nvmlInit()

    if not torch.cuda.is_available():
        raise Exception("Cuda not available.")
    else:
        logging.info("CUDA OK.")
    logging.info("Checking a device is free...")
    while True:
        best_device = -1
        best_free = -1
        logging.info("Current Device Status:")
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            pct_free = info.free/info.total
            
            logging.info(f"Device {i}: {pct_free:.2%} free.")
            if pct_free > best_free:
                best_free = pct_free
                best_device = i
        if best_free > 0.7:
            logging.info(f"Selected device {best_device} with {best_free:.2%} free.")
            break
        else:
            logging.info(f"No devices > 70% free, waiting 5mins.")
            time.sleep(60*5)

    device = torch.device(f'cuda:{best_device}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    for data_types in data_types_l:
        for n in N:
            for image_sig_type in image_sig_types:
                for encoder_t in encoder_type:
                    for l_r in learning_rate:
                        for b_s in batch_size:
                            logging.info(f"Starting Training Procedure with: N = {n} | Image Sig Type = {image_sig_type} | Encoder Type = {encoder_t} | Learning Rate = {l_r} | Data Types = {data_types} | Batch Size = {b_s}")
                            run_training_procedure(image_sig_type, encoder_t, n, device, l_r, b_s, data_types=data_types)
                            logging.info(f"Completed Training Procedure with: N = {n} | Image Sig Type = {image_sig_type} | Encoder Type = {encoder_t} | Learning Rate = {l_r} | Data Types = {data_types} | Batch Size = {b_s}")
    nvidia_smi.nvmlShutdown()

if __name__ == "__main__":
    main()
