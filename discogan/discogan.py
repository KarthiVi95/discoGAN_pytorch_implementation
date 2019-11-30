'''
REFERENCES

1) https://arxiv.org/pdf/1703.05192.pdf
2) https://arxiv.org/pdf/1808.04325.pdf
3) https://github.com/SKTBrain/DiscoGAN (1) Paper Author's Implementation)
4) https://github.com/carpedm20/DiscoGAN-pytorch (Few References)

'''

# required imports
import os
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import imageio
import numpy as np
import csv
import random
import cv2

'''
The Generator network is defined close to the orginal paper. Considering the complexity of the
dataset, I have added another extra convultional layer in the middle for better performance
'''
class Generator(nn.Module):
    # conv2d parameters reference
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,64*2,4,2,1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*2, 64*4,4,2,1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*8,100,4,1,0, bias=False),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(100,64*8,4,1,0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2,64,4,2,1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*1,3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main( input )

'''
The Discriminator network. The layers are defined as per the original paper.
Discriminator is very similar to the generator but differs by the last layers which is a sigmoid layer

'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64*8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(64*8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3( bn3 )

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.relu4(bn4)

        conv5 = self.conv5(relu4)

        return torch.sigmoid(conv5), [relu2, relu3, relu4]

'''
Function reads all images and puts them in a np array
Images are dilated and cropped in case of edges2shoes
All Images are resized, pixel values are squeezed and transposed
'''
def read_images_as_np(filenames, preprocess=None):
    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue
        
        if preprocess == 'dilate':
            # https://arxiv.org/pdf/1808.04325.pdf - contains detailed information as to why we need to dilate
            # dilating the image in domain A for easier shape transfer
            dilation_kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, dilation_kernel, iterations=1 )
            image = 255. - image
        elif preprocess == 'crop':
            # cropping the image in domain B
            image = image[:, 256:, :]
            
        image = cv2.resize(image, (64,64))
        # normalizing the inputs for faster calculations
        image = image.astype(np.float32) / 255.
        # reshaping (64,64,3) image to (3,64,64) image
        image = image.transpose(2,0,1)
        images.append( image )
    # converting list to np matrix, i.e. adding a row to the matrix
    images = np.stack(images)
    return images

'''
Function reads train and validation images from appropriate
folder depending upon the choice of dataset
'''
def get_image_paths(dataset='edges2shoes',test=False):
    imagePath = None
    # horse2zebra paths
    if dataset == 'horse2zebra':
        imagePath= './../datasets/horse2zebra/'
        if test==False:
            imagePath_horses= os.path.join(imagePath,'trainA')
            imagePath_zebras= os.path.join(imagePath,'trainB')
        else:
            imagePath_horses= os.path.join(imagePath,'testA')
            imagePath_zebras= os.path.join(imagePath,'testB')
        return [list(map(lambda x: os.path.join(imagePath_horses,x), os.listdir(imagePath_horses))),list(map(lambda x: os.path.join(imagePath_zebras,x), os.listdir(imagePath_zebras)))]
    # edges2shoes path
    else:
        imagePath = './../datasets/edges2shoes/'
        image_paths=[]
        if test == True:
            imagePath = os.path.join( imagePath, 'val' )
        else:
            imagePath = os.path.join( imagePath, 'train' )
            
        image_paths = os.listdir(imagePath)
        for i in range(len(image_paths)):
            image_paths[i] = os.path.join(imagePath,image_paths[i])
            image_paths[i] = image_paths[i].replace("\\",'/')
        
        if test == True:
            return [image_paths, image_paths]
        else:
            n_images = len( image_paths )
            return [image_paths[:n_images//2], image_paths[n_images//2:]]
    
'''
Function converts GPU tensor into a numpy array
'''
def to_numpy(data):
    return data.cpu().data.numpy()

'''
Function calculates feature matching loss for better shape transformation as suggested in the paper
https://arxiv.org/pdf/1808.04325.pdf 
'''
def calculate_featureMatch_loss(real_features, fake_features, criterion):
    # introduced in a later paper with https://arxiv.org/pdf/1808.04325.pdf 
    # for improving state deformation
    losses = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
        losses += loss

    return losses

'''
Function calculates BCE loss for generator and discriminator
'''

def calculate_gan_loss(dis_real, dis_fake, criterion):
    # creating tensor of 1s for loss calculation
    # code adapted from paper author's code
    labels_dis_real = Variable(torch.ones([dis_real.size()[0], 1])).cuda()
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1])).cuda()
    labels_gen = Variable(torch.ones([dis_fake.size()[0], 1])).cuda()
    
    # BCE losses are calculated here
    dis_loss = criterion(dis_real, labels_dis_real) * 0.5 + criterion(dis_fake, labels_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, labels_gen)

    return dis_loss, gen_loss

'''
Function updates the losses to a csv file for plotting graphs
'''
def log_to_csv(val_dict):
    # writing values to a CSV file for plotting
    losses = ["epoch","gen_loss_A","gen_loss_B","dis_loss_A","dis_loss_B","recon_loss_A","recon_loss_B","fm_loss_A","fm_loss_B"]
    if os.path.exists('./logger.csv'):
        with open('./logger.csv', mode='a') as logger:
            logger_writer = csv.DictWriter(logger,fieldnames=losses)
            logger_writer.writerow(val_dict)
    else:
        with open('./logger.csv', mode='w') as logger:
            logger_writer = csv.DictWriter(logger,fieldnames=losses)
            logger_writer.writeheader()
            logger_writer.writerow(val_dict)
            
'''
Function takes generators and tests them using images from validation
set to form a grid of images
'''
def predict_and_form_images(generator_A,generator_B,task):
    # takes the two generators and makes predictions on the validation set
    val_A,val_B=get_image_paths(dataset=task,test=True)
    val_A = read_images_as_np( val_A, 'dilate')
    val_B = read_images_as_np( val_B, 'crop')
    val_A = Variable( torch.FloatTensor( val_A ), volatile=True).cuda()
    val_B = Variable( torch.FloatTensor( val_B ), volatile=True).cuda()
    AB = generator_B(val_A)
    BA = generator_A(val_B)
    ABA = generator_A(AB)
    BAB = generator_B(BA)
    
    # no of images to be tested on the model
    n_testset = 25

    white_space_ver = np.ones((10,445,3),dtype=int)*254
    white_space_hor = np.ones((64,10,3),dtype=int)*254
    result_img = np.ones((10,445,3),dtype=int)*254
    
    # forming a grid of images with 25 rows and 6 columns
    for im_idx in range(n_testset):
        hor_buff = np.ones((64,1,3),dtype=int)*254
        A_val = to_numpy(val_A[im_idx]).transpose(1,2,0) * 255.
        B_val = to_numpy(val_B[im_idx]).transpose(1,2,0) * 255.
        BA_val = to_numpy(BA[im_idx]).transpose(1,2,0)* 255.
        ABA_val = to_numpy(ABA[im_idx]).transpose(1,2,0)* 255.
        AB_val = to_numpy(AB[im_idx]).transpose(1,2,0)* 255.
        BAB_val = to_numpy(BAB[im_idx]).transpose(1,2,0)* 255.

        # horizontally stacking images with color correction 
        hor_buff = np.hstack((hor_buff,A_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.hstack((hor_buff,AB_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.hstack((hor_buff,ABA_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.hstack((hor_buff,B_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.hstack((hor_buff,BA_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.hstack((hor_buff,BAB_val.astype(np.uint8)[:,:,::-1]))
        hor_buff = np.hstack((hor_buff,white_space_hor))
        hor_buff = np.vstack((hor_buff,white_space_ver))
        result_img = np.vstack((result_img,hor_buff))
    
    return result_img

def main():

    # Tweakable parameters to play around 
    epoch_size = 120
    batch_size = 64
    learning_rate = 0.0002
    
    # Parameters to customize the run 
    torch.cuda.set_device(0)
    task_name = 'edges2shoes'
    result_path = './results/'
    model_path = './models/'
    image_save_interval = 2
    model_save_interval = 10
    
    
    result_path = os.path.join( result_path, task_name )
    model_path = os.path.join( model_path, task_name )
    # reading all file names
    data_domainA, data_domainB = get_image_paths(dataset=task_name,test=False)
    test_domainA, test_domainB = get_image_paths(dataset=task_name,test=True)

    if task_name == 'edges2shoes':
        test_A = read_images_as_np( test_domainA, 'dilate')
        test_B = read_images_as_np( test_domainB, 'crop')
    else:
        # loading test images as matrix in the shape (no of images,no of channels,64,64)
        test_A = read_images_as_np( test_domainA, None)
        test_B = read_images_as_np( test_domainB, None)

    # loading the test_images to the GPU
    test_A = Variable(torch.FloatTensor(test_A), volatile=True).cuda()
    test_B = Variable(torch.FloatTensor(test_B), volatile=True).cuda()

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator_A = Generator().cuda()
    generator_B = Generator().cuda()
    discriminator_A = Discriminator().cuda()
    discriminator_B = Discriminator().cuda()

    # minimum of the no of train images in domain A and domain B
    data_size = min( len(data_domainA), len(data_domainB) )
    num_batches = ( data_size // batch_size )

    # Combining the two iterators into one which contains the parameters 
    # of both the generator
    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    # betas and weight decays followed as per paper suggested standards
    optim_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    for epoch in range(epoch_size):
        # 0 to 120
        # Shuffling the training data for avoiding biases
        print("Epoch "+str(epoch)+" in progress .....")
        random.shuffle(data_domainA)
        random.shuffle(data_domainB)

        for i in range(num_batches):

            # Setting the gradients to zeros before back propagation
            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            # Seperating batches of training data
            A_path = data_domainA[ i * batch_size: (i+1) * batch_size ]
            B_path = data_domainB[ i * batch_size: (i+1) * batch_size ]

            if task_name == 'edges2shoes':
                A = read_images_as_np( A_path, 'dilate')
                B = read_images_as_np( B_path, 'crop')
            else:
                A = read_images_as_np( A_path, None)
                B = read_images_as_np( B_path, None)
            
            A = Variable( torch.FloatTensor( A ) ).cuda()
            B = Variable( torch.FloatTensor( B ) ).cuda()

            # Parallel GANs model definition 
            
            # AB generates images in the form Gab - Horse that looks like corresponding zebra
            AB = generator_B(A)
            # Horse that is converted to zerbra and then converted back to original horse
            ABA = generator_A(AB)
            
            BA = generator_A(B)
            BAB = generator_B(BA)

            # l2 norm loss or cycle reconstruction loss. Cycle GANS use L1 loss for this
            mse_loss = nn.MSELoss()
            # Reconstruction Loss - MSE Loss as suggested in the Paper
            # Loss concurred while changing horse -> zebra -> horse
            recon_loss_A = mse_loss(ABA, A)
            # Loss concurred while changing zebra -> horse -> zebra
            recon_loss_B = mse_loss(BAB, B)

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A(A)
            A_dis_fake, A_feats_fake = discriminator_A(BA)
            
            bce_loss = nn.BCELoss()
            hinge_loss = nn.HingeEmbeddingLoss()

            dis_loss_A, gen_loss_A = calculate_gan_loss(A_dis_real, A_dis_fake, bce_loss)
            fm_loss_A = calculate_featureMatch_loss(A_feats_real, A_feats_fake, hinge_loss)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B( B )
            B_dis_fake, B_feats_fake = discriminator_B( AB )

            dis_loss_B, gen_loss_B = calculate_gan_loss( B_dis_real, B_dis_fake, bce_loss)
            fm_loss_B = calculate_featureMatch_loss( B_feats_real, B_feats_fake, hinge_loss )
            
            # modified loss function with shape transform (feature matching loss) included
            gen_loss_total_A = (gen_loss_B*0.1 + fm_loss_B*0.7) * 0.5 + recon_loss_A * 0.5
            gen_loss_total_B = (gen_loss_A*0.1 + fm_loss_A*0.7) * 0.5 + recon_loss_B * 0.5

            gen_loss = gen_loss_total_A + gen_loss_total_B
            dis_loss = dis_loss_A + dis_loss_B
        
            if iters % 3 == 0:
                # training two generators and one discriminator in order
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters%(num_batches-1) == 0:
                val_dic = {
                        "epoch": epoch,
                        "gen_loss_A":to_numpy(gen_loss_A.mean()),
                        "gen_loss_B":to_numpy(gen_loss_B.mean()),
                        "dis_loss_A":to_numpy(dis_loss_A.mean()),
                        "dis_loss_B":to_numpy(dis_loss_B.mean()),
                        "recon_loss_A":to_numpy(recon_loss_A.mean()),
                        "recon_loss_B":to_numpy(recon_loss_B.mean()),
                        "fm_loss_A":to_numpy(fm_loss_A.mean()),
                        "fm_loss_B":to_numpy(fm_loss_B.mean())
                        }
                log_to_csv(val_dic)
            
            if iters % (num_batches-1)//2 == 0:
                print("Generator Losses        "+ "A |"+ str(to_numpy(gen_loss_A.mean()))+"B |"+str(to_numpy(gen_loss_B.mean())))
                print("Discriminator Losses    "+ "A |"+ str(to_numpy(dis_loss_A.mean()))+"B |"+str(to_numpy(dis_loss_B.mean())))
                print("Reconstruction Losses   "+ "A |"+ str(to_numpy(recon_loss_A.mean()))+"B |"+str(to_numpy(recon_loss_B.mean())))
                print("Feature Matching Losses "+ "A |"+ str(to_numpy(fm_loss_A.mean()))+"B |"+str(to_numpy(fm_loss_B.mean())))
                
            if epoch %image_save_interval == 0:
                imageio.imwrite((result_path +'/'+ str(epoch)+'.jpg'),predict_and_form_images(generator_A,generator_B,task_name))
                    
            if epoch % model_save_interval == 0:
                torch.save( generator_A, os.path.join(model_path,('model_gen_A-' + str(epoch)+'.pth')))
                torch.save( generator_B, os.path.join(model_path,('model_gen_B-' + str(epoch)+'.pth')))
                torch.save( discriminator_A, os.path.join(model_path,('model_dis_A-' + str(epoch)+'.pth')))
                torch.save( discriminator_B, os.path.join(model_path,('model_dis_B-' + str(epoch)+'.pth')))

            iters += 1

if __name__=="__main__":
    main()