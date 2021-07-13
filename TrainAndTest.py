import math
import numpy as np
from PIL import Image
import torch
import random
import CacheDictUtils as c_utils
import GenericDataloader as g_dl
import PolygonFactory as pf
import os
import CNNPolyPredictor as cnn_poly
import MLPPolyPredictor as mlp_poly

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def generateTestData(num_images, max_num_sides, min_number_sides, smaller_ims):
    if smaller_ims:
        im_width = 300
        fp = "./test_data_small/im"
    else:
        im_width = 600
        fp = "./test_data_large/im"

    ensure_dir(fp)

    for image_num in range(1, num_images+1):
        im_fp = fp+str(image_num)+".png"
        info_fp = fp+str(image_num)
        orig_num_sides = random.randint(min_number_sides, max_num_sides)
        radius = random.randint(50, int(im_width / 2 - 1))
        area = pf.getPolygonArea(orig_num_sides, radius)
        poly = pf.getColoredPolygon(orig_num_sides, radius, im_width)
        image = Image.fromarray(poly)
        image.save(im_fp)
        test_data_dict = {}
        test_data_dict["num_sides"] = orig_num_sides
        test_data_dict["radius"] = radius
        test_data_dict["area"] = area
        c_utils.writeReadableCachedDict(info_fp, test_data_dict)

#generateTestData(500, 14, 3, smaller_ims=True)
#generateTestData(500, 13, 3, smaller_ims=False)
#debug = "debug"

def readTestData(smaller_ims):
    if smaller_ims:
        fp = "./test_data_small/im"
    else:
        fp = "./test_data_large/im"

    batch_size = 5
    # 500 test images, will process in batches of 5
    for i in range(1, 100):
        poly_ims = []
        side_nums = []
        areas = []
        batch_dict = {}

        for j in range(batch_size):
            image_num = i*batch_size + j
            im_fp = fp + str(image_num) + ".png"
            info_fp = fp + str(image_num)
            poly = np.array(Image.open(im_fp))
            poly = np.moveaxis(poly, -1, 0)  # channels last to channels first
            poly = torch.tensor([poly], dtype=torch.float)
            poly_ims.append(poly)
            info_dict = c_utils.readReadableCachedDict(info_fp)
            orig_num_sides = info_dict["num_sides"]
            side_nums.append(orig_num_sides)
            area = info_dict["area"]
            areas.append(area)

        side_nums = np.array(side_nums)
        side_nums = torch.tensor(side_nums, dtype=torch.long).cuda()

        areas = np.array(areas)
        areas = torch.unsqueeze(torch.tensor(areas, dtype=torch.float), dim=1).cuda()

        poly_ims = torch.cat([im for im in poly_ims], dim=0).cuda()

        batch_dict["poly_ims"] = poly_ims
        batch_dict["num_sides"] = side_nums
        batch_dict["areas"] = areas
        yield batch_dict


def getBatch(start_value, skip_value, batch_size, im_width, max_num_sides, min_number_sides, num_images):

    for current_index in range(num_images):
        batch_dict = {}

        if not g_dl.skipFunction(current_index, start_value, batch_size, skip_value):
            continue

        orig_num_sides = random.randint(min_number_sides, max_num_sides)
        radius = random.randint(50, int((im_width/2)-1))

        poly = pf.getColoredPolygon(orig_num_sides, radius, im_width)
        poly = np.moveaxis(poly, -1, 0) # channels last to channels first
        poly = torch.tensor([poly], dtype=torch.float)

        side_num = torch.tensor(orig_num_sides-min_number_sides, dtype=torch.long)

        area = pf.getPolygonArea(orig_num_sides, radius)
        area = torch.tensor(area, dtype=torch.float)

        batch_dict["poly_im"] = poly
        batch_dict["side_num"] = side_num
        batch_dict["area"] = area

        yield batch_dict


def collate_func(batch):
    poly_ims = []
    num_sides = []
    areas = []

    for batch_elem in batch:
        areas.append(torch.unsqueeze(batch_elem["area"], dim=0))
        num_sides.append(torch.unsqueeze(batch_elem["side_num"], dim=0))
        poly_ims.append(batch_elem["poly_im"])

    poly_ims = torch.cat([poly_im for poly_im in poly_ims], dim=0)
    #poly_ims = poly_ims /255.0
    #poly_ims = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(poly_ims)
    num_sides = torch.cat([num_side for num_side in num_sides], dim=0)
    areas = torch.cat([area for area in areas], dim=0)
    areas = torch.unsqueeze(areas, dim=1)

    return poly_ims, num_sides, areas


def trainAndTest(model, optimizer, lr_scheduler, im_width):
    if __name__ == "__main__":
        sub_batch_loss = 0.0
        sub_batch_counter = 1
        num_training_ims = 400000
        iterator = getBatch
        # args are of the form im_width, max_num_sides, min_number_sides, num_images
        args = [im_width, 14, 3, num_training_ims]

        dl = g_dl.getSkipIterableDataLoader(bs=8, iterator=iterator, args=args, num_workers=8, collate_func=collate_func, pinned_memory=True, persistent_workers=True, prefetch_factor=2)
        for batch_num, batch in enumerate(dl):
            optimizer.zero_grad()
            batch_dict = {}
            batch_dict["poly_ims"] = batch[0].cuda(non_blocking=True)
            batch_dict["num_sides"] = batch[1].cuda(non_blocking=True)
            batch_dict["areas"] = batch[2].cuda(non_blocking=True)
            loss_dict, _ = model(batch_dict)
            loss = loss_dict["side_pred_loss"] + loss_dict["area_loss"]
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
            sub_batch_loss += loss_val

            if batch_num % 25 == 0 and batch_num != 0:
                sub_batch_loss = round(sub_batch_loss, 2)
                print("The accumulated loss for sub-batch " + str(sub_batch_counter) + " is " + str(sub_batch_loss))
                print("The average loss per batch is " + str(round(sub_batch_loss/25, 2)) + "\n")
                if sub_batch_loss/25 < 0.05: # early stopping for very low training loss values
                    break
                sub_batch_loss = 0.0
                sub_batch_counter += 1
                if lr_scheduler is not None:
                    lr_scheduler.step()

        print("Training complete!")

        # eval
        model.eval()
        correct = torch.tensor(0, dtype=torch.long).cuda()
        error = torch.tensor(0.0, dtype=torch.float).cuda()
        smaller_ims = True
        if im_width == 600:
            smaller_ims = False
        with torch.no_grad():
            for index, batch in enumerate(readTestData(smaller_ims)):
                _, pred_dict = model(batch)
                side_predictions = pred_dict["side_predictions"]
                area_predictions = pred_dict["areas"]
                gt_num_sides = batch["num_sides"]
                gt_areas = batch["areas"]
                correct += torch.sum(side_predictions == gt_num_sides)
                error += torch.sum((area_predictions-gt_areas)**2)

        print("Evaluation complete!")

        correct = int(correct.item())
        error = float(error.item())
        accuracy = round((correct/500), 2)
        rmse = round(math.sqrt(error/500), 2)
        print("The model accuracy for predicting the number of sides is " + str(accuracy))
        print("The root mean-squared error for area predictions is " + str(rmse))


def trainAndTestCNNPolyPredictor(im_width):
    model = cnn_poly.getCNNPolyPredictor(im_width)
    optimizer, lr_scheduler = cnn_poly.getOptimizerAndLRScheduler(model)
    trainAndTest(model, optimizer, lr_scheduler, im_width)

def trainAndTestMLPPredictor(im_width):
    model = mlp_poly.createPolyMLP(im_width)
    optimizer, lr_scheduler = mlp_poly.getOptimizer(model)
    trainAndTest(model, optimizer, lr_scheduler, im_width)

#trainAndTestCNNPolyPredictor(600)
trainAndTestCNNPolyPredictor(300)

#trainAndTestMLPPredictor(300)
#trainAndTestMLPredictor(600)
