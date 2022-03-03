# The transfer learning example in pytorch website.

import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

# Function to remove certain parameters from optimizing
def set_param_required_grad (model, extracting):
    if (extracting):
        for param in model.parameters():
            param.requires_grad = False
        # end for
    # end if
# end set_...

# Function to train the model
def train_model (model, data_loaders, criterion, optimizer, epochs = 25):
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    # Setup tensorboard
    writer = SummaryWriter(log_dir='runs/slow_wave_detecter_res50')

    model = model.to(device)

    for epoch in range(epochs):
        print('='*20)
        print('Epoch: %d / %d' %(epoch, epochs - 1))

        for phase in ['train', 'val']:
            print('-'*20)
            
            if (phase == 'train'):
                model.train()
            else:
                model.eval()
            # end if

            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with T.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = T.max(outputs, 1)

                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()
                    # end if
                # end with

                current_loss = loss.item()
                current_correct = T.sum(preds == labels.data)
                print('--- [{} {:d}] Loss: {:.4f} Acc: {:.4f}'.format(phase, i, current_loss, current_correct.double() / inputs.size(0)))
                writer.add_scalar('{}_loss'.format(phase), current_loss, epoch * len(data_loaders[phase]) + i)
                writer.add_scalar('{}_acc'.format(phase), current_correct.double() / inputs.size(0), epoch * len(data_loaders[phase]) + i)

                # Calculate confusion matrix
                conf_matrix = T.zeros(2, 2)
                for t, p in zip(labels.data, preds):
                    conf_matrix[t, p] += 1
                # end for

                # Calculate specificity and sensitivity
                TP = conf_matrix[1,1]
                FN = conf_matrix[1,0]
                FP = conf_matrix[0,1]
                sensitivity = TP.double() / (TP + FN)
                specificity = TP.double() / (TP + FP)
                Aroc = sensitivity * specificity
                writer.add_scalar('{}_sensitivity'.format(phase), sensitivity, epoch * len(data_loader[phase]) + i)
                writer.add_scalar('{}_specificity'.format(phase), specificity, epoch * len(data_loader[phase]) + i)
                writer.add_scalar('{}_Aroc'.format(phase), Aroc, epoch * len(data_loader[phase]) + i)
            # end for
        # end for
    # end for

    writer.close()

    T.save(model, 'slowwave_classifier_res50.pth')
# end train_model

if __name__ == '__main__':
    root_dir = 'data/'

    image_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val':   transforms.Compose([transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    }

    data_generator = {k: datasets.ImageFolder(os.path.join(root_dir, k),
                        image_transforms[k]) for k in ['train', 'val']}
    
    data_loader = {k: T.utils.data.DataLoader(data_generator[k], batch_size=32, shuffle=True,
                        num_workers=4) for k in ['train', 'val']}

    print('Number of training batches = %d' %(len(data_loader['train'])))
    print('Number of validation batches = %d' %(len(data_loader['val'])))

    model = models.resnet50(pretrained=True)
    set_param_required_grad(model, True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2),
        nn.Softmax(dim=1)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, data_loader, criterion, optimizer)
    print('Done!')
# end main