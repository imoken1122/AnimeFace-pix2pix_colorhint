import argparse
import ast

def arg_as_list(s):                                                            
    v = ast.literal_eval(s)
    if type(v) is not list:                                                    
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v    
def define_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./dataset",type=str)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", default = 4,type=int)
    parser.add_argument("--which_epoch", default=0, type=int)
    parser.add_argument('--niter', type=int, default=100,)
    parser.add_argument('--niter_decay', type=int, default=0, )
    parser.add_argument("--isTrain",action="store_true")
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--continue_train_latest', action='store_true')
    parser.add_argument("--cuda",action="store_true")
    parser.add_argument("--seed",type=int, default=100)
    parser.add_argument("--save_epoch_freq",type=int, default=5)
    parser.add_argument("--save_iter_freq",type=int, default=2)

    parser.add_argument("--aug" ,action='store', dest='aug',
                        type=str, nargs='*', default=['circle'])
    parser.add_argument('--disc_norm_type', type=str, default='instance')
    parser.add_argument('--gen_norm_type', type=str, default='instance')
    parser.add_argument('--no_vgg_loss', action='store_true')
    parser.add_argument('--sle', action='store_true')
    parser.add_argument('--gan_mode', type=str, default='hinge')
    parser.add_argument('--gen_in_ch', type=int, default=4)
    parser.add_argument("--disc_in_ch",type=int, default=7)
    parser.add_argument("--beta1",type=float, default=0.5)
    parser.add_argument("--beta2",type=float, default=0.999)
    parser.add_argument("--lr",type=float, default=0.0002)
    parser.add_argument("--lambda_vgg",type=float, default=10.)
    parser.add_argument('--lr_policy', type=str, default='lambda',)
    parser.add_argument('--init_type', type=str, default='xavier')
    opt = parser.parse_args()

    return opt
