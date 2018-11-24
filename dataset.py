#coding: utf-8
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row,modality):
        self._data = row
        self.modality = modality

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if self.modality =="Flow":
            return int(self._data[1])-1
        else:
            return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])-1


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='im{}.jpg', flow_tmpl ='flow_{}{}.jpg',transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False,save_scores=False):

        self.root_path = root_path
        self.list_file = list_file
        #split 3 ,here we divide video 3 segmentation
        self.num_segments = num_segments
        # you must remeber the new length is using to get input imformation ,for example ,
        #if you use the img diff you must use 2 length
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.flow_tmpl=flow_tmpl
        self.transform = transform
        self.random_shift = random_shift
        #for here test_mode is False,but if you want to true
        self.test_mode = test_mode
        self.save_scores = save_scores

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff
        #For here we get the video list
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.flow_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.flow_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        #u must be know the list format eg. YoYo/v_YoYo_g11_c02.avi 101
        self.video_list = [VideoRecord(x.strip().split(' '),modality = self.modality) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        #if here we devide for 3 segmen
        average_duration = (record.num_frames - self.new_length+1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        #验证的时候取得是中间片段，这个有点意思
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        #suppose here we have 13220 vedio ,so we need to choice to deal with this problem
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        #[x1,x2,x3] for each x1 ,call us will the off set in each segment

        return self.get(record, segment_indices) if self.save_scores==False else self.get_and_record(record,segment_indices)

    def get(self, record, indices):

        #here we must remember for here is the rgbdiff we need 6 img,but flow we only need 5 img
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.transform != None:
            process_data = self.transform(images)
        else:
            process_data = images
        return process_data, record.label
    def get_and_record(self,record,indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        if self.transform != None:
            process_data = self.transform(images)
        else:
            process_data = images
        return record.path,process_data, record.label

    def __len__(self):
        return len(self.video_list)
if __name__ == '__main__':
    t = TSNDataSet("", "object.txt", num_segments=3,
               new_length=5,
               modality="Flow",
               image_tmpl="im{}.jpg",
               transform=None
               )
    img,label = t[0]

