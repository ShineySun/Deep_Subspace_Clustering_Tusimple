import os
from tqdm import tqdm
import json
import argparse
import numpy as np
import cv2

class Data_Builder_Tusimple:
    def __init__(self, K = 4, data_path = "tusimple"):
        self._K = K
        self._data_path = data_path
        # train : label_data_0313.json, label_data_0601.json
        # validation : label_data_0531.json
        # test : test_label.json
        self._train_json_file_path = os.path.join(self._data_path, r'label_data_0313.json')

    def quick_inspector(self):
        data_list = []

        with open(self._train_json_file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)

        color_map_mat = np.zeros((6,3), dtype=np.uint8)

        for i in range(0,6):
            color_map_mat[i] = np.random.randint(0,255,dtype=np.uint8, size=3)

        for data_entry in tqdm(data_list):
            clip_dir = os.path.dirname(data_entry['raw_file'])

            twenty_frames = [os.path.join(clip_dir, '%d.jpg' %x) for x in range(1,21)]

            for frame in twenty_frames:
                #print("Frame : {}".format(os.path.join(self._data_path, frame)))
                bgr_image = cv2.imread(os.path.join(self._data_path, frame))

                seg_image = np.zeros_like(bgr_image)

                print("len(data_entry['lanes']) : {}".format(len(data_entry['lanes'])))

                for lane_idx, lane_x_points in enumerate(data_entry['lanes'],0):
                    lane_label_color = color_map_mat[lane_idx]

                    curve_vertices = list(filter(lambda xy_pair: xy_pair[0]>0, zip(lane_x_points, data_entry['h_samples'])))

                    for vertex_1, vertex_2 in zip(curve_vertices[:-1], curve_vertices[1:]):
                        color = tuple(np.random.randint(0, 255, dtype=np.uint8, size=3))
                        cv2.line(seg_image, tuple(vertex_1), tuple(vertex_2), (int(color_map_mat[5][0]), int(color_map_mat[5][1]), int(color_map_mat[5][2])),2)

            res = cv2.addWeighted(bgr_image, 1, seg_image, 0.5, 0.4)

            cv2.imshow("Image BGR TUSIMPLE", bgr_image)
            cv2.imshow("Image BGR TUSIMPLE with Lane", res)
            cv2.waitKey(1)

    def K_Parser(self):
        data_list = []

        with open(self._train_json_file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)

        f_w =  open('test.txt','w')
        data_name = 0

        for data_entry in tqdm(data_list):
            clip_dir = os.path.dirname(data_entry['raw_file'])

            number_of_class = len(data_entry['lanes'])

            image_list = list()

            if self._K == number_of_class:
                bgr_image = cv2.imread(os.path.join(self._data_path, clip_dir, '20.jpg'))
                #print("os.path.join(self._data_path, clip_dir) : {}".format(os.path.join(self._data_path, clip_dir)))

                seg_image_0 = np.zeros_like(bgr_image)
                seg_image_1 = np.zeros_like(bgr_image)
                seg_image_2 = np.zeros_like(bgr_image)
                seg_image_3 = np.zeros_like(bgr_image)

                lane_counter = 0


                for lane_idx, lane_x_points in enumerate(data_entry['lanes'],0):
                    print("Lane_Index : {}".format(lane_idx))
                    print("Lane X Points : {}".format(len(np.unique(lane_x_points))))

                    if len(np.unique(lane_x_points)) > 1:
                        lane_counter += 1

                    curve_vertices = list(filter(lambda xy_pair : xy_pair[0] > 0, zip(lane_x_points, data_entry['h_samples'])))

                    for vertex_1, vertex_2 in zip(curve_vertices[:-1], curve_vertices[1:]):
                        if lane_idx == 0:
                            cv2.line(seg_image_0, tuple(vertex_1), tuple(vertex_2), (255,255,255), 5)

                        elif lane_idx == 1:
                            cv2.line(seg_image_1, tuple(vertex_1), tuple(vertex_2), (255,255,255), 5)

                        elif lane_idx == 2:
                            cv2.line(seg_image_2, tuple(vertex_1), tuple(vertex_2), (255,255,255), 5)

                        elif lane_idx == 3:
                            cv2.line(seg_image_3, tuple(vertex_1), tuple(vertex_2), (255,255,255), 5)

                    if lane_idx == 0:
                        image_list.append(seg_image_0)
                    if lane_idx == 1:
                        image_list.append(seg_image_1)
                    if lane_idx == 2:
                        image_list.append(seg_image_2)
                    if lane_idx == 3:
                        image_list.append(seg_image_3)

                if lane_counter == 4:
                    for image in image_list:
                        tmp_name = 'data/class4/' + str(data_name).zfill(4) + '.jpg'
                        cv2.imwrite(tmp_name, image)
                            #cv2.imshow("seg_image!!", image)
                            #cv2.waitKey(-1)

                        tmp_content = str(data_name%4) + ' ' + str(data_name).zfill(4) + '.jpg\n'
                        f_w.write(tmp_content)
                        data_name += 1


                    print("Lane Counter : {}".format(lane_counter))

                    # cv2.imshow("seg_image_0",seg_image_0)
                    #
                    # cv2.imshow("seg_image_1",seg_image_1)
                    #
                    # cv2.imshow("seg_image_2",seg_image_2)
                    #
                    # cv2.imshow("seg_image_3",seg_image_3)
                        #cv2.waitKey(1)



                # cv2.imshow("K CLASS", bgr_image)
                # cv2.waitKey(-1)





def parse_args():
    parser = argparse.ArgumentParser(description='Tusimple Dataset for K class')

    parser.add_argument('--rootDir', type=str, default=r'tusimple')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--K', type=int, default=4)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_builder = Data_Builder_Tusimple(K=args.K, data_path = args.rootDir)

    if args.mode == 0:
        data_builder.quick_inspector()
    elif args.mode == 1:
        data_builder.K_Parser()
