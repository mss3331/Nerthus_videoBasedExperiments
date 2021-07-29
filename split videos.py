from glob import glob as glob
import shutil, os

class_dir = r"E:\Databases\Nerthus\SubSubVideoBased_not_splitted_into_trainVal\3"
output ="n"

all_subvideos = glob(class_dir+'\\*')
all_subvideos.sort()
sub_video_name = 3
for sub_video in all_subvideos:
    frames = glob(sub_video+"\\*.jpg")
    frames.sort()
    folder_name = sub_video.split('\\')[-1]

    if len(frames)>80: #split this video if it exceed 80 frames "we have 25,52,100 and 125"
        # new_sub_video_name = "_".join(folder_name.split('_')[:-1]+[str(sub_video_name)]) #--> 1_0_3, 1_0_4
        new_sub_video_name = folder_name+"_"+str(sub_video_name) #--> 1_0_1_3, 1_0_2_4
        print(new_sub_video_name)
        frames_subvidoe_files = frames[1::2]
        os.mkdir(class_dir + '\\' + new_sub_video_name)
        for frame in frames_subvidoe_files:
            shutil.copy(frame,class_dir+'\\'+new_sub_video_name)
            os.remove(frame)
        sub_video_name+=1

    # print(frames)
    # exit(0)