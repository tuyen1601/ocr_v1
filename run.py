import os
import sys
import time
import math
from tools.infer import predict_system
from ppocr.utils import logging
from ppocr.utils import utility
from numpy import median
from pdf2image import convert_from_path
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from support.Get_Angle import getAngleBetweenPoints, getPoint_Center, getPointRotate, getIntersection
from support.Get_Distance import getDistance
from support.Rotate import rotate
from support.Draw_Image import draw
from support.Download_Image import download_with_progressbar
from support.Crop_Image import Crop_Image
from support.Init import Init
from support.Convert_CV2PIL import opencv2pil

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

logger = logging.get_logger()

# __all__ = ['PaddleOCR']
#
# SUPPORT_DET_MODEL = ['DB']
# VERSION = 2.0

def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help)
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=False)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str, default='./test/2.jpg')
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str, default='./detection_model')
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)

        parser.add_argument("--det", type=str2bool, default=True)
        return parser.parse_args()
    else:
        return argparse.Namespace(
            use_gpu=True,
            ir_optim=True,
            use_tensorrt=False,
            gpu_mem=8000,
            image_dir='',
            det_algorithm='DB',
            det_model_dir=None,
            det_limit_side_len=960,
            det_limit_type='max',
            max_text_length=25,
            rec_char_dict_path=None,
            use_space_char=True,
            drop_score=0.5,
            cls_model_dir='./cls_model',
            cls_image_shape="3, 48, 192",
            label_list=['0', '180'],
            cls_batch_num=30,
            cls_thresh=0.9,
            enable_mkldnn=False,
            use_zero_copy_run=False,
            use_pdserving=False,
            lang='ch',
            det=True,
            rec=True,
            use_angle_cls=False)


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):

        postprocess_params = parse_args(mMain=False)
        postprocess_params.__dict__.update(**kwargs)
        self.use_angle_cls = postprocess_params.use_angle_cls

        # if postprocess_params.det_algorithm not in SUPPORT_DET_MODEL:
        #     logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
        #     sys.exit(0)

        # init text_detection_model
        super().__init__(postprocess_params)

    def ocr(self, img, det=True):
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det:
            logger.error('When input a list of images, det must be false')
            exit(0)

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = utility.check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        dt_boxes, elapse = self.text_detector(img)
        if dt_boxes is None:
            return None
        return [box.tolist() for box in dt_boxes]

config = Cfg.load_config_from_file('config.yml')
detector = Predictor(config)
args = parse_args(mMain=True)
ocr_engine = PaddleOCR(**(args.__dict__))



def sorter(item):
    return item[0][0]

def run():
    result_ocr = ''
    Init('./Line_cut/', './pdf_image/', './Result/')
    # for cmd
    # args = parse_args(mMain=True)

    image_dir = args.image_dir
    if '.pdf' in image_dir:
        print('pdf----')
        # images = convert_from_path(image_dir, poppler_path='/home/tuyen/Desktop/Project/OCR/OCR/poppler-0.68.0/bin') #windows
        images = convert_from_path(image_dir)
        for i in range(len(images)):
            images[i].save('./pdf_image/page_' + str(i+1) + '.jpg', 'JPEG')
        image_file_list = utility.get_image_file_list('./pdf_image')

    else:
        if image_dir.startswith('http'):
            download_with_progressbar(image_dir, 'tmp.jpg')
            image_file_list = ['tmp.jpg']
        else:
            image_file_list = utility.get_image_file_list(image_dir)
        if len(image_file_list) == 0:
            logger.error('no images find in {}'.format(image_dir))

    start = time.time()
    count1 = 1
    count2 = 1
    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        f = open('./Result/{}.txt'.format(img_name), 'w', encoding='utf-8')
        img = cv2.imread(img_path)
        w_0, h_0 = img.shape[:2]
        #print(img.shape[:2])
        # deskew
        result_0 = ocr_engine.ocr(img_path, det=True)
        list_angle = []
        if result_0 is not None:
            for line in result_0:
                angle = getAngleBetweenPoints(line)
                list_angle.append(int(angle))
            list_angle.sort()
            angle_skew = median(list_angle)
            img = rotate(img, angle_skew, (0, 0, 0))
            cv2.imwrite('./Result/deskew.jpg', img)
            
            # print(img.shape[:2])
            print("Deskew Done!")
        
        #print(result_0)
        #crop ảnh theo angle_skew

        w, h = img.shape[:2]
        result_1  = []
        center_0 = [int(w_0/2),int(h_0/2)]
        center = [int(w/2),int(h/2)]
        #print(center)
        angle_skew = math.pi*angle_skew/180
       
        if result_0 is not None:
            for line in result_0:
                line_1 = []
                for point in line:
                    point = getPoint_Center(point,center_0)
                    point = getPointRotate(point,angle_skew,center)
                    line_1.append(point)
                result_1.append(line_1)
        # print("Ket qua sau khi xoay")
        
        #kết thúc kết quả xoay
      
        #print(result_1)
        # for line in result_1:
        #     img  = draw(img, line)
        # cv2.imwrite('.\\Result\\img.jpg', img)

        # sắp xếp thành các line
        img1 = img.copy()
        result_1.reverse()
        list_point = []
        list_flag = []
        for line in result_1:
            i_point = getIntersection([line[0],line[2]],[line[1],line[3]])
            list_point.append(i_point)
            list_flag.append(0)
        temp = []
        temp.append(result_1[0])
        list_flag[0] = 1
        i=0
        end_=False
        while i<len(result_1):
            if list_flag[i] == 1:
                if i+1<len(result_1):
                    h = (getDistance(result_1[i][0],result_1[i][3])+getDistance(result_1[i][1],result_1[i][2]))/4
                    if list_point[i+1][1] <= list_point[i][1]+1.2*h and list_point[i+1][1] >= list_point[i][1]-1.2*h:
                        list_flag[i+1] = 1
                        temp.append(result_1[i+1])
            else:

                temp = sorted(temp, key=sorter)

                for line in temp:
                    img1 = draw(img1, line)
                    angle = getAngleBetweenPoints(line)
                    crop, crop_h, crop_w = Crop_Image(img, line, angle)
                    cv2.imwrite('./Line_cut/' + str(count1) + '.jpg', crop)
                    gray = opencv2pil(crop)

                    try:
                        # start1 = time.time()
                        text, prob = detector.predict(gray, return_prob=True)  # muốn trả về xác suất của câu dự đoán thì đổi return_prob=True
                        # end1 = time.time()
                        if str(prob) != 'nan':
                            if float(prob) >= 0.4:
                                f.write(text+'\t')
                                result_ocr += text+'\t'
                                print(text)
                    except:
                        print('None')
                    count1 += 1
                f.write('\n')
                result_ocr+='\n'
                temp.clear()
                if end_:
                    break
                if i!=len(result_1)-1:
                    list_flag[i] = 1
                else:
                    end_ = True
                temp.append(result_1[i])
                i-=1
            i+=1
        cv2.imwrite('./Result/{}.jpg'.format(img_name), img1)
        count2 += 1
        f.write('\n')
        result_ocr += '\n'
    f.close()
    f.close()
    folder_line_cut = './Line_cut/'
    for image in os.listdir(folder_line_cut):
        os.remove(folder_line_cut + image)
    os.remove('./Result/deskew.jpg')
    end = time.time()
    # print("thoi gian 2: "+ str(end - start))
    # end = time.time()
    print('Time: %5.3f' % (end - start), ' s')
    return result_ocr.strip()

if __name__ == '__main__':

    run()