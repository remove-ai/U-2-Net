import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
from PIL import Image as Img
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'


    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images' + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name+'_human_seg', model_name + '_human_seg.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7


def process_image_named(name, threshold_cutoff=0.90, use_transparency=False):
    result_img = load_img('test_data/test_human_images_results/' + name + '.png')
    # convert result-image to numpy array and rescale(255 for RBG images)
    RESCALE = 255
    out_img = img_to_array(result_img)
    out_img /= RESCALE
    # define the cutoff threshold below which, background will be removed.
    THRESHOLD = threshold_cutoff

    # refine the output
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0

    if use_transparency:
        # convert the rbg image to an rgba image and set the zero values to transparent
        shape = out_img.shape
        a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
        mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(out_img, a_layer, axis=2)
        mask_img = Img.fromarray((rgba_out * RESCALE).astype('uint8'), 'RGBA')
    else:
        mask_img = Img.fromarray((out_img * RESCALE).astype('uint8'), 'RGB')

    # load and convert input to numpy array and rescale(255 for RBG images)
    input = load_img('test_data/test_human_images/' + name + '.jpg')
    inp_img = img_to_array(input)
    inp_img /= RESCALE

    if use_transparency:
        # since the output image is rgba, convert this also to rgba, but with no transparency
        a_layer = np.ones(shape=(shape[0], shape[1], 1))
        rgba_inp = np.append(inp_img, a_layer, axis=2)

        # simply multiply the 2 rgba images to remove the backgound
        rem_back = (rgba_inp * rgba_out)
        rem_back_scaled = Img.fromarray((rem_back * RESCALE).astype('uint8'), 'RGBA')
    else:
        rem_back = (inp_img * out_img)
        rem_back_scaled = Img.fromarray((rem_back * RESCALE).astype('uint8'), 'RGB')

    # select a layer(can be 0,1 or 2) for bounding box creation and salient map
    LAYER = 2
    out_layer = out_img[:, :, LAYER]

    # find the list of points where saliency starts and ends for both axes
    x_starts = [
        np.where(out_layer[i] == 1)[0][0] if len(np.where(out_layer[i] == 1)[0]) != 0 else out_layer.shape[0] + 1 for i
        in range(out_layer.shape[0])]
    x_ends = [np.where(out_layer[i] == 1)[0][-1] if len(np.where(out_layer[i] == 1)[0]) != 0 else 0 for i in
              range(out_layer.shape[0])]
    y_starts = [
        np.where(out_layer.T[i] == 1)[0][0] if len(np.where(out_layer.T[i] == 1)[0]) != 0 else out_layer.T.shape[0] + 1
        for i in range(out_layer.T.shape[0])]
    y_ends = [np.where(out_layer.T[i] == 1)[0][-1] if len(np.where(out_layer.T[i] == 1)[0]) != 0 else 0 for i in
              range(out_layer.T.shape[0])]

    # get the starting and ending coordinated for the box
    startx = min(x_starts)
    endx = max(x_ends)
    starty = min(y_starts)
    endy = max(y_ends)

    # show the resulting coordinates
    start = (startx, starty)
    end = (endx, endy)
    print(start, end)

    cropped_rem_back_scaled = rem_back_scaled.crop((startx, starty, endx, endy))
    if use_transparency:
        cropped_rem_back_scaled.save('test_data/test_humandata_cropped_results/' + name + '_cropped_no-bg.png')
    else:
        cropped_rem_back_scaled.save('test_data/test_humandata_cropped_results/' + name + '_cropped_no-bg.jpg')

    cropped_mask_img = mask_img.crop((startx, starty, endx, endy))

    if use_transparency:
        cropped_mask_img.save('test_data/test_humandata_cropped_results/' + name + '_cropped_no-bg_mask.png')
    else:
        cropped_mask_img.save('test_data/test_humandata_cropped_results/' + name + '_cropped_no-bg_mask.jpg')


if __name__ == "__main__":
    main()
    image_dir = os.path.join(os.getcwd(), 'test_data/test_human_images_results/*.png')
    print(image_dir)
    file_names = glob(image_dir)
    # print(file_names)
    names = [os.path.basename(name[:-4]) for name in file_names]
    print(names)

    for name in names:
        process_image_named(name)
