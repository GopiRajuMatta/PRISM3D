import torch
import math
import numpy as np
import cv2
import random
import os

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))
    #rounding = 1e8
    #y = torch.round(y*rounding)/rounding
    return y.float()

def event_loss_call_linlog(all_rgb, event_data, data_factor, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1

        small_val = 1e-10
        #print(all_rgb.shape)
        all_rgb_end= 0.299*all_rgb[end,:,:,0]+0.587*all_rgb[end,:,:,1]+0.114*all_rgb[end,:,:,2]
        all_rgb_start=  0.299*all_rgb[start,:,:,0]+0.587*all_rgb[start,:,:,1]+0.114*all_rgb[start,:,:,2]
        #all_rgb_start= all_rgb_start[::data_factor,::data_factor]
        #all_rgb_end= all_rgb_end[::data_factor,::data_factor]
        #thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.35
        #thres = (torch.log(all_rgb[end] * 255) - torch.log(all_rgb[start]* 255)) / 0.35
        #print(thres)
        event_cur = event_data[start]
        #print(event_cur.shape, data_factor)
        event_cur = event_cur[::data_factor,::data_factor]
        #print(all_rgb_end.shape, all_rgb_start.shape,event_cur.shape)
        thres_pos = (lin_log((all_rgb_end * 255) + small_val) - lin_log((all_rgb_start * 255) + small_val)) / 0.3
        thres_neg = (lin_log((all_rgb_end * 255) + small_val) - lin_log((all_rgb_start * 255) + small_val)) / 0.2
        thres= (lin_log((all_rgb_end * 255) + small_val) - lin_log((all_rgb_start * 255) + small_val)) / 0.25
        pos = event_cur > 0
        neg = event_cur < 0
        noe = event_cur == 0
        #print(thres)
        loss_pos = ((thres * pos) - ((event_cur + 0.5) * pos)) ** 2
        loss_neg = ((thres * neg) - ((event_cur - 0.5) * neg)) ** 2
        loss_noe = ((thres * noe) - ((event_cur) * noe)) ** 2
        #print(pos)
        loss.append(torch.mean(loss_pos + loss_neg + loss_noe))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    #print(event_loss)
    return event_loss

def event_loss_call_linlog_normalized_pos_neg_noe(all_rgb, event_data, data_factor, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1

        small_val = 1e-10
        #print(all_rgb.shape)
        all_rgb_end= 0.299*all_rgb[end,:,:,0]+0.587*all_rgb[end,:,:,1]+0.114*all_rgb[end,:,:,2]
        all_rgb_start=  0.299*all_rgb[start,:,:,0]+0.587*all_rgb[start,:,:,1]+0.114*all_rgb[start,:,:,2]
        #all_rgb_start= all_rgb_start[::data_factor,::data_factor]
        #all_rgb_end= all_rgb_end[::data_factor,::data_factor]
        #thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.35
        #thres = (torch.log(all_rgb[end] * 255) - torch.log(all_rgb[start]* 255)) / 0.35
        #print(thres)
        event_cur = event_data[start]
        #print(event_cur.shape, data_factor)
        event_cur = event_cur[::data_factor,::data_factor]
        #print(all_rgb_end.shape, all_rgb_start.shape,event_cur.shape)
        thres = (lin_log((all_rgb_end * 255) + small_val) - lin_log((all_rgb_start * 255) + small_val)) / 0.35
        thres = thres / (torch.linalg.norm(thres, dim=0, keepdim=True) + 1e-9)
        event_cur = event_cur / (torch.linalg.norm(event_cur, dim=0, keepdim=True) + 1e-9)
        pos = event_cur > 0
        neg = event_cur < 0
        noe = event_cur == 0
        #print(thres)
        loss_pos = ((thres * pos) - ((event_cur + 0.5) * pos)) ** 2
        loss_neg = ((thres * neg) - ((event_cur - 0.5) * neg)) ** 2
        loss_noe = ((thres * noe) - ((event_cur) * noe)) ** 2
        #print(pos)
        loss.append(torch.mean(loss_pos + loss_neg + loss_noe))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    #print(event_loss)
    return event_loss

def event_loss_call_linlog_normalized(all_rgb, event_data, data_factor, bin_num):
    loss = []
    small_val = 1e-10
    for its in range(bin_num):
        start = its
        end = its + 1

        # Convert RGB to grayscale
        #gray_start = 0.299 * all_rgb[start, :, :, 0] + 0.587 * all_rgb[start, :, :, 1] + 0.114 * all_rgb[start, :, :, 2]
        #gray_end = 0.299 * all_rgb[end, :, :, 0] + 0.587 * all_rgb[end, :, :, 1] + 0.114 * all_rgb[end, :, :, 2]

        gray_start = 1/3 * all_rgb[start, :, :, 0] + 1/3 * all_rgb[start, :, :, 1] + 1/3 * all_rgb[start, :, :, 2]
        gray_end = 1/3 * all_rgb[end, :, :, 0] + 1/3 * all_rgb[end, :, :, 1] + 1/3 * all_rgb[end, :, :, 2]

        # Linear + logarithmic mapping
        bright_start = lin_log((gray_start * 255) + small_val)
        bright_end = lin_log((gray_end * 255) + small_val)

        # Brightness change
        thres = (bright_end - bright_start) / 0.35

        # Downsample for comparison with event resolution (if needed)
        event_cur = event_data[start][::data_factor, ::data_factor]
        # bright_diff = thres[::data_factor, ::data_factor]  # Uncomment if needed
        # event_cur = event_cur[::data_factor, ::data_factor]

        # Normalize both brightness difference and event frame
        norm_thres = thres# / (torch.linalg.norm(thres, dim=0, keepdim=True) + 1e-9)
        norm_event = event_cur# / (torch.linalg.norm(event_cur, dim=0, keepdim=True) + 1e-9)

        # Squared L2 loss between normalized frames
        loss.append(torch.mean((norm_thres - norm_event) ** 2))

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss

import torch.nn.functional as F

def event_loss_call_linlog_cosine_normalized(all_rgb, event_data, data_factor, bin_num):
    loss = []
    small_val = 1e-10

    for its in range(bin_num):
        start = its
        end = its + 1

        # Convert RGB to grayscale (already downsampled)
        gray_start = 0.299 * all_rgb[start, :, :, 0] + 0.587 * all_rgb[start, :, :, 1] + 0.114 * all_rgb[start, :, :, 2]
        gray_end   = 0.299 * all_rgb[end, :, :, 0]   + 0.587 * all_rgb[end, :, :, 1]   + 0.114 * all_rgb[end, :, :, 2]

        # Apply lin-log mapping
        bright_start = lin_log((gray_start * 255) + small_val)
        bright_end   = lin_log((gray_end   * 255) + small_val)

        # Brightness difference (threshold surface)
        thres = bright_end - bright_start

        # Downsample event data to match RGB
        event_cur = event_data[start][::data_factor, ::data_factor]

        # Normalize both to unit vectors (pixel-wise cosine sim)
        norm_thres = thres / (torch.norm(thres, p=2, dim=0, keepdim=True) + 1e-9)
        norm_event = event_cur / (torch.norm(event_cur, p=2, dim=0, keepdim=True) + 1e-9)

        # Cosine similarity loss = 1 - cosine_similarity
        cosine_sim = torch.sum(norm_thres * norm_event, dim=0)
        loss.append(torch.mean(1.0 - cosine_sim))

    event_loss = torch.mean(torch.stack(loss), dim=0)
    return event_loss

def event_loss_call_linlog_normalized_random_consecutive(all_rgb, event_data, data_factor, bin_num):
    """
    Event loss using a randomly selected consecutive bin pair.
    Normalizes both brightness diff and event map before computing L2 loss.

    Args:
        all_rgb: [bin_num, H, W, 3] - downsampled RGB images.
        event_data: [bin_num, H_full, W_full] - full-res event frames.
        data_factor: int - downsampling factor for event data.
        bin_num: int - number of bins (must be >= 2).

    Returns:
        Scalar event loss.
    """
    small_val = 1e-10

    # Randomly pick a valid start frame index (to ensure start+1 exists)
    start = torch.randint(0, bin_num - 1, (1,)).item()
    end = start + 1

    # Convert RGB to grayscale
    #gray_start = 0.299 * all_rgb[start, :, :, 0] + 0.587 * all_rgb[start, :, :, 1] + 0.114 * all_rgb[start, :, :, 2]
    #gray_end   = 0.299 * all_rgb[end, :, :, 0]   + 0.587 * all_rgb[end, :, :, 1]   + 0.114 * all_rgb[end, :, :, 2]


    gray_start = 1/3.0 * all_rgb[start, :, :, 0] + 1/3.0 * all_rgb[start, :, :, 1] + 1/3.0 * all_rgb[start, :, :, 2]
    gray_end   = 1/3.0 * all_rgb[end, :, :, 0]   + 1/3.0 * all_rgb[end, :, :, 1]   + 1/3.0 * all_rgb[end, :, :, 2]

    # Apply lin-log mapping
    bright_start = lin_log((gray_start * 255) + small_val)
    bright_end   = lin_log((gray_end   * 255) + small_val)

    # Brightness difference
    thres = bright_end - bright_start

    # Downsample event frame
    event_cur = event_data[start][::data_factor, ::data_factor]

    # Normalize both threshold surface and event surface
    norm_thres = thres / (torch.linalg.norm(thres, dim=0, keepdim=True) + 1e-9)
    norm_event = event_cur / (torch.linalg.norm(event_cur, dim=0, keepdim=True) + 1e-9)

    # Final normalized L2 loss
    event_loss = torch.mean((norm_thres - norm_event) ** 2)
    return event_loss



def event_loss_call_log(all_rgb, event_data, bin_num, data_factor):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1
        #print(start,end)
        #print("data_factor", data_factor)
        small_val = 1e-10
        #print(all_rgb.shape)
        all_rgb_end= 0.299*all_rgb[end,:,:,0]+0.587*all_rgb[end,:,:,1]+0.114*all_rgb[end,:,:,2]
        all_rgb_start=  0.299*all_rgb[start,:,:,0]+0.587*all_rgb[start,:,:,1]+0.114*all_rgb[start,:,:,2]
        #all_rgb_start= all_rgb_start[::data_factor,::data_factor]
        #all_rgb_end= all_rgb_end[::data_factor,::data_factor]
        #thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.35
        #thres = (torch.log(all_rgb[end] * 255) - torch.log(all_rgb[start]* 255)) / 0.35
        #print(thres)
        event_cur = event_data[start]
        event_cur = event_cur[::data_factor,::data_factor]
        #print(all_rgb_end.shape, all_rgb_start.shape,event_cur.shape)
        #thres = (torch.log((all_rgb_end * 255) + small_val) - torch.log((all_rgb_start * 255) + small_val))
        render_brightness_diff = all_rgb_end - all_rgb_start
        render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9)
        pos = event_cur > 0
        neg = event_cur < 0
        noe = event_cur == 0
        target_s = event_cur
        target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9)
        thres = render_norm
        event_cur = target_s_norm
        #print(pos)
        loss_pos = ((thres * pos) - ((event_cur) * pos)) ** 2
        loss_neg = ((thres * neg) - ((event_cur) * neg)) ** 2
        loss_noe = ((thres * noe) - ((event_cur) * noe)) ** 2

        loss.append(torch.mean(loss_pos + loss_neg + loss_noe))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss


