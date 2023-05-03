import torch
import json
from models.Video_sota_models import slow_r50
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict

####################
# SlowFast transform
####################

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 1
frames_per_second = 30

alpha= 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second

if __name__=='__main__':
    # Device on which to run the model
    # Set to cuda to load on GPU
    device = "cpu"

    # Pick a pretrained model and load the pretrained weights
    '''models that work directly with 25 frames
        model_name= slow_r50, c2d_r50, i3d_r50, csn_r101
    '''
    # model_name = "slow_r50"
    # model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=False)
    # print(model)
    # # exit(0)
    # # Set to eval mode and move to desired device
    # model = model.to(device)
    # model = model.eval()
    # # Download the example video file
    #
    #
    # # Load the example video
    # video_path = "archery.mp4"
    #
    # # Select the duration of the clip to load by specifying the start and end duration
    # # The start_sec should correspond to where the action occurs in the video
    # start_sec = 0
    # end_sec = 1
    #
    # # Initialize an EncodedVideo helper class
    # video = EncodedVideo.from_path(video_path)
    #
    # # Load the desired clip
    # video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    #
    # # Apply a transform to normalize the video input
    # video_data = transform(video_data)
    #
    # # Move the inputs to the desired device
    # inputs = video_data["video"]
    # inputs = [i.to(device)[None, ...] for i in inputs]
    # [print(i.shape) for i in inputs]
    # print(len(inputs))
    # data1 = torch.randn((2,3,8,200,200))
    # data2 = torch.randn((2,3,25,200,200))
    #
    # data = [data1, data2]
    # # [print(i.shape) for i in data]
    # #
    # out = model(data2)
    # print(out.shape)
    ############## testing ready to use models ###############
    model = slow_r50()
    data = torch.randn((2,25,3,224,224))
    out = model(data)
    print(out.shape)