from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Image_to_3D')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plot_every_batch', type=int, default=10)
    parser.add_argument('--save_every_epoch', type=int, default=25)
    parser.add_argument('--save_after_epoch', type=int, default=1)
    parser.add_argument('--test_every_epoch', type=int, default=25)
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--skip_train', action='store_true')

    parser.add_argument('--viewnum', type=int, default=36)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--mcube_znum', type=int, default=128)
    parser.add_argument('--test_pointnum', type=int, default=65536)
    parser.add_argument('--chunk_s', type=int, default=0)
    parser.add_argument('--chunk_l', type=int, default=217)

    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=4)

    # Required. Model & Dataset identifier.
    parser.add_argument('--model', type=str,
                        help='Full path of the model')
    parser.add_argument('--dataset', type=str,
                        help='Full path of the dataset')

    # Data augmentation
    parser.add_argument('--random_h_flip', action='store_true')
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--normalize', action='store_true')

    # Model componenets
    parser.add_argument('--point_decoder', action='store_true')
    parser.add_argument('--warm_start', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--cam_batch_size', type=int, default=16)
    parser.add_argument('--cam_lr', type=float, default=0.00005)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--sampling_mode', type=str, default='weighted')
    parser.add_argument('--exp_name', '-e', type=str, default='d2im+tGCN')
    parser.add_argument('--eval_pred', action='store_true')
    parser.add_argument('--supervise_proj', action='store_true')
    parser.add_argument('--coarse_point_density', type=int, default=10000)
    parser.add_argument('--sample_point_density', type=int, default=32768)
    parser.add_argument('--sdf_max_dist', type=float, default=1.0)
    parser.add_argument('--sdf_scale', type=float, default=1.0)

    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.003, 0.01, 0.07])
    parser.add_argument('--sample_distribution', type=float, nargs='+',
                        default=[0.5, 0.49, 0.01])

    parser.add_argument('--point_feat', type=int,
                        default=[128, 128, 256, 256, 256, 128, 128, 3], nargs='+',
                        help='Features for point decoder.')
    parser.add_argument('--point_degree', type=int,
                        default=[2, 2, 2, 2, 2, 2, 64], nargs='+',
                        help='Upsample degrees for point decoder.')
    parser.add_argument('--im_enc_layers', type=int,
                        default=[1, 1, 1, 1, 16, 32, 64, 128, 128], nargs='+',
                        help='Layer dimension for voxnet encoder.')

    parser.add_argument('--n_decoder_pos', type=int, default=2)
    parser.add_argument('--bb_min', type=float, default=-0.5,
                        help='Bounding box minimum.')
    parser.add_argument('--bb_max', type=float, default=0.5,
                        help='Bounding box maximum.')
    parser.add_argument('--vox_res', type=int, default=128,
                        help='Bounding box res.')

    parser.add_argument(
        '--data_dir', default='/work/06035/sami/maverick2/datasets/shapenet/')
    parser.add_argument(
        '--mesh_dir', default='/work/06035/sami/maverick2/datasets/shapenet/im3d/isosurface/')
    parser.add_argument(
        '--h5_dir', default='/work/06035/sami/maverick2/datasets/shapenet/im3d/sampled_points/')
    parser.add_argument(
        '--density_dir', default='/work/06035/sami/maverick2/datasets/shapenet/d2im/SDF_density/')
    parser.add_argument(
        '--cam_dir', default='/work/06035/sami/maverick2/datasets/shapenet/disn/image/')
    parser.add_argument(
        '--image_dir', default='/work/06035/sami/maverick2/datasets/shapenet/disn/image/')
    parser.add_argument(
        '--normal_dir', default='/work/06035/sami/maverick2/datasets/shapenet/d2im/normal_processed/')
    parser.add_argument('--catlist', type=str,
                        default=['03001627', '02691156', '02828884', '02933112', '03211117', '03636649',
                                 '03691459', '04090263', '04256520', '04379243', '04530566', '02958343', '04401088'],
                        nargs='+',
                        help='catagory list.')

    # parser.add_argument('--model_dir', default='/work/06035/sami/maverick2/results/d2im/')
    parser.add_argument(
        '--output_dir', default='/work/06035/sami/maverick2/results/')
    # parser.add_argument('--log', default='log.txt')
    parser.add_argument('--test_cam_id', type=int,
                        default=2,
                        help='Cam id to test with.')
    parser.add_argument('--test_gpu_id', type=int,
                        default=0,
                        help='GPU id to test with.')
    parser.add_argument('--test_checkpoint', default='best_model_test.pt.tar')
    parser.add_argument('--testlist_file',
                        default='./data/DISN_split/testlist_all.lst')

    args = parser.parse_args()
    # some selected chairs with details
    with open(args.testlist_file, 'r') as f:
        lines = f.readlines()

    # print(lines)
    testlist = []
    for l in lines[:30]:
        fn = l.strip()
        if not fn == '':
            fn = fn.split(' ')
            if fn[0] in args.catlist:
                testlist.extend(
                    [{'cat_id': fn[0], 'shape_id':fn[1], 'cam_id':fn[2]}])

    args.testlist = testlist
    print(args.testlist)
    # args.catlist = ['03001627']
    # args.catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']

    args.checkpoint_dir = args.output_dir+args.exp_name+'/checkpoints/'
    args.results_dir = args.output_dir+args.exp_name+'/'
    args.log = args.output_dir+args.exp_name+'/log.txt'

    return args


if __name__ == '__main__':
    args = get_args()
    print(len(args.testlist))
