from struct import unpack
from tqdm import tqdm
import os


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []


def remove_bad_photos(folder):
    bad_data = []
    for dirs in os.listdir(folder):
        for file in os.listdir(os.path.join(folder,dirs)):
            try:
                im = JPEG(os.path.join(folder,dirs,file))
                im.decode()
            except Exception as e:
                print(e)
                print(f'Bad file: {os.path.join(folder,dirs,file)}')
                bad_data.append(os.path.join(folder,dirs,file))
    print(len(bad_data))
    for img in bad_data:
        os.remove(img)

if __name__ == '__main__':
    remove_bad_photos('cleanup_data/dataset/yoga_set1/train')
    remove_bad_photos('cleanup_data/dataset/yoga_set1/test')
    remove_bad_photos('cleanup_data/dataset/yoga_set2/train')
    remove_bad_photos('cleanup_data/dataset/yoga_set2/test')