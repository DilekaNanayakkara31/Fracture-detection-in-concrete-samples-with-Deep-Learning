# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from matplotlib.pyplot import imshow
torch.manual_seed(0)

#________________________________________________________________________________________________________________________________________________________________

url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip"
r = requests.get(url, allow_redirects = True)
open("concrete_crack_images_for_classification.zip", 'wb').write(r.content)

file = "concrete_crack_images_for_classification.zip"
with ZipFile(file, 'r') as zip:
    print('Extracting all files now.....')
    zip.extractall()
    print('Done!')
#________________________________________________________________________________________________________________________________________________________________


def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

directory= ""

negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
#print(negative_files[0:3])

positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
#print(positive_files[0:3])

#________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# Create dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory=""
        positive="Positive_tensors"
        negative='Negative_tensors'

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        # torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")

#_______________________________________________________________________________________________________________________________________________________________

# create two dataset objects, one for the training data and one for the validation data
train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("done")   

#_______________________________________________________________________________________________________________________________________________________________

# Load the pre-trained model resnet18
model = models.resnet18(pretrained = True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

composed = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

# Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the output layer model.fc of the neural network with a nn.Linear object, to classify 2 different classes. For the parameters in_features  remember the last hidden layer has 512 neurons
model.fc = nn.Linear(512, 2)

print(model)

#_______________________________________________________________________________________________________________________________________________________________

# Create the loss function
criterion = nn.CrossEntropyLoss()

# Create a training loader and validation loader object, the batch size should have 100 samples each.
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = 100)
validation_loader = torch.utils.data.DataLoader(dataset= validation_dataset, batch_size = 100)

# optimizer to minimize the loss
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)

# calculate the accuracy on the validation data for one epoch
n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:

        model.train() 
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z = z = model(x)   
        # calculate loss 
        loss = criterion(z, y)
        # calculate gradients of parameters 
        loss.backward()
        optimizer.step()
        # update parameters 
        loss_list.append(loss.data)
    
    correct=0
    for x_test, y_test in validation_loader:
        # set model to eval 
        model.eval()
        
        #make a prediction 
        z = model(x_test)
        
        #find max 
        _, yhat = torch.max(z.data, 1)
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        correct += (yhat == y_test).sum().item()
   
    accuracy=correct/N_test
    loss_list.append(loss.data)
    accuracy_list.append(accuracy)

#_______________________________________________________________________________________________________________________________________________________________
# Print out the Accuracy and plot the loss stored in the list loss_list for every iteration
print(accuracy)
plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

#_______________________________________________________________________________________________________________________________________________________________
# Find the misclassified samples

train_loader_new = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = 1)
validation_loader_new = torch.utils.data.DataLoader(dataset= validation_dataset, batch_size = 1)

n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()

i = 0

for epoch in range(n_epochs):
    for x, y in train_loader_new:

        model.train() 
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z = z = model(x)   
        # calculate loss 
        loss = criterion(z, y)
        # calculate gradients of parameters 
        loss.backward()
        optimizer.step()
        # update parameters 
        loss_list.append(loss.data)
    
    correct=0
    for x_test, y_test in validation_loader_new:
        # set model to eval 
        model.eval()
        
        #make a prediction 
        z = model(x_test)
        
        #find max 
        _, yhat = torch.max(z.data, 1)
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        if yhat != y_test :
            print("sample {}  predicted value: {}  actual value: {}".format(i, yhat, y_test))
        
        i += 1
    

