import argparse
import os

class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        self.initialized = True
        parser.add_argument('--landmark_num', type=int, default=68, help='landmark number')
        parser.add_argument('--vertex_num', type=int, default=6144, help='vertex number of 3D mesh')
        parser.add_argument('--device_num', type=int, default=0, help='gpu id')
        parser.add_argument('--data_path', type=str, default="data/", help='path of related data')
        if_train_parser = parser.add_mutually_exclusive_group(required=False)
        if_train_parser.add_argument('--train', dest='if_train', action='store_true') # train mode
        if_train_parser.add_argument('--no_train', dest='if_train', action='store_false') # test mode
        parser.set_defaults(if_train=True)
        parser.add_argument('--train_image_path', type=str, default="exp/train_images.txt", help='train images path')
        parser.add_argument('--train_landmark_path', type=str, default="exp/train_landmarks.txt",
                            help='train landmarks path')
        parser.add_argument('--train_vertex_path', type=str, default="exp/train_vertex.txt",
                            help='train vertex path')
        parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
        parser.add_argument('--num_workers', type=int, default=6, help='threads for loading data')
        parser.add_argument('--test_image_path', type=str, default="exp/test_images.txt", help='test images path')
        parser.add_argument('--test_landmark_path', type=str, default="exp/test_landmarks.txt",
                            help='test landmarks path')
        parser.add_argument('--test_lrecord_path', type=str, default="exp/test_lrecord.txt",
                            help='path to save estimated landmarks')
        parser.add_argument('--test_vrecord_path', type=str, default="exp/test_vrecord.txt",
                            help='path to save estimated coordinates of vertices')
        parser.add_argument('--resnet34_lr', type=float, default=1e-4, help='learning rate of ResNet34')
        parser.add_argument('--mynet1_lr', type=float, default=1e-5,
                            help='learning rate of the first and second FC layers of MyNet')
        parser.add_argument('--mynet2_lr', type=float, default=1e-8,
                            help='learning rate of the last FC layer of MyNet')
        use_premodel_parser = parser.add_mutually_exclusive_group(required=False)
        use_premodel_parser.add_argument('--premodel', dest='use_premodel', action='store_true') # use pretrained model
        use_premodel_parser.add_argument('--no_premodel', dest='use_premodel', action='store_false') # no pretrained model
        parser.set_defaults(use_premodel=True)
        parser.add_argument('--model1_path', type=str, default="model/resnet34_adam.pth",
                            help='the pretrained model of ResNet34 structure')
        parser.add_argument('--model2_path', type=str, default="model/mynet_adam.pth",
                            help='the pretrained model of MyNet structure')
        parser.add_argument('--total_epoch', type=int, default=1000, help='number of total training epoch')
        parser.add_argument('--lambda_land', type=float, default=1, help='weight of landmark loss')
        parser.add_argument('--lambda_srt', type=float, default=1e-1, help='weight of srt loss')
        parser.add_argument('--test_frequency', type=int, default=100, help='frequency for testing')
        parser.add_argument('--save_frequency', type=int, default=200, help='frequency for saving models')
        parser.add_argument('--save_model_path', type=str, default="record/", help='path to save models')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):

        opt = self.gather_options()
        self.opt = opt
        return self.opt