import argparse


def arg_init():
    parser = argparse.ArgumentParser(description='Spy Cam Experiments.')
    parser.add_argument('--gpu', type=int, choices=[0,1,2,3,4], default=0, help='gpu to run the exp.')
    parser.add_argument('-d', '--data', type=int, default=0, help='which dataset to use: 0-office, 1-hotel, 2-airbnb.')
    parser.add_argument('-e', '--epoch', type=int, default=25, help='total number of training epoches.')
    parser.add_argument('-n', '--name', type=str, default='test', help='name of the experiment.')
    parser.add_argument('--multi', action='store_true', help='run multi-class classification instead of binary.')
    parser.add_argument('--ther_off', action='store_false', help='disable using thermal images (will use original image only)')
    parser.add_argument('--orig_off', action='store_false', help='disable using original images (will use thermal image only)')
    parser.add_argument('--mask_off', action='store_false', help='disable masking')
    parser.add_argument('-a', '--attention', type=str, choices=['CBAM', 'BAM', None], default=None, help='Type of attention block to use.')
    args = parser.parse_args()
    return args