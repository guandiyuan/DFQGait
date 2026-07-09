
import os
import argparse
import torch
import torch.nn as nn
from lib.modeling import models
import torch.multiprocessing as mp
from lib.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr, get_model_complexity
from thop import profile, clever_format


parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='.\DFQGait\config/DFQGaitGait_CasiaB.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
parser.add_argument('--device', default='0,1', type=str, help='device id, 0 or 1 or 0,1')

opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    ################################################
    flops_model = Model(cfgs, training)
    flops_model.cuda()
    flops_model.eval()

    # Dummy inputs
    sils = torch.randn(64,60,64,44).cuda()
    pose = torch.randn(64,60,17,3).cuda()

    with torch.no_grad():
        flops, params = profile(
            flops_model,
            inputs=((sils, pose),)
        )

    flops, params = clever_format([flops, params], format="%.2f")
    msg_mgr.log_info(f"FLOPs: {flops}")
    msg_mgr.log_info(f"Params: {params}")
    #####################################################################

    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))

    ################################################################
    # input_size = (3, 224, 224)
    # flops, params = get_model_complexity(model, input_size)
    # msg_mgr.log_info(f"FLOPs: {flops}")
    # msg_mgr.log_info(f"Total Params (ptflops): {params}")
    ############################################################
    msg_mgr.log_info("Model Initialization Finished!")


    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


# if __name__ == '__main__':
#     # torch.distributed.init_process_group('nccl', init_method='env://')
#     # if torch.distributed.get_world_size() != torch.cuda.device_count():
#     #     raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
#     #         torch.cuda.device_count(), torch.distributed.get_world_size()))
#     import os
#     import torch.distributed as dist
#     import torch
#     import gc
#
#     def force_clear_cuda():
#         # 1. 删除所有引用
#         for var in dir():
#             obj = eval(var)
#             if isinstance(obj, torch.Tensor) and obj.is_cuda:
#                 del obj
#         # 2. Python 垃圾回收
#         gc.collect()
#         # 3. PyTorch 清空缓存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             # 如果上面不够，强制同步（会卡一下，但确保显存释放）
#             torch.cuda.synchronize()
#         print("CUDA Memory Cleared!")
#
#
#     # 在你的代码中调用
#     # force_clear_cuda()
#
#     # 设置环境变量
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '9998'
#     os.environ["USE_LIBUV"] = "0"
#     torch.distributed.init_process_group('gloo', init_method='env://', rank=0, world_size=1)
#     cfgs = config_loader(opt.cfgs)
#     if opt.iter != 0:
#         cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
#         cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
#
#     training = (opt.phase == 'train')
#     initialization(cfgs, training)
#     run_model(cfgs, training)
##############################################################
"上述为测试过程"

# def main(rank, args):
#         """
#         rank表示进程序号，用于进程间通讯，每一个进程对应了一个rank,单机多卡中可以理解为第几个GPU。
#         args为函数传入的参数
#         """
#
#         # Environment settings
#         MASTER_ADDR = "localhost"
#         MASTER_PORT = "17877"
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#         os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#         os.environ["MASTER_ADDR"] = MASTER_ADDR
#         os.environ["MASTER_PORT"] = MASTER_PORT
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
#         print('Distributed init (rank {}): {}'.format(rank, 'env://'), flush=True)
#
#         # Windows does not support nccl backend, use gloo instead of nccl, it is recommended to use nccl on Linux.
#         torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size,
#                                              rank=rank)
#
#         if torch.distributed.get_world_size() != torch.cuda.device_count():
#             raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
#                 torch.cuda.device_count(), torch.distributed.get_world_size()))
#         cfgs = config_loader(opt.cfgs)
#         if opt.iter != 0:
#             cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
#             cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
#
#         training = (opt.phase == 'train')
#         initialization(cfgs, training)
#         run_model(cfgs, training)
#
#
# if __name__ == '__main__':
#     WORK_PATH = "."
#     os.chdir(WORK_PATH)
#     print("WORK_PATH:", os.getcwd())
#     mp.spawn(main,
#              args=(opt,),
#              nprocs=opt.world_size,
#              join=True)


####################################

if __name__ == '__main__':
    import sys
    import os
    from datetime import datetime, timedelta, timezone
    import torch.distributed as dist
    import torch
    import gc
    import time


    class Tee:
        def __init__(self, log_file, mode='w', encoding='utf-8'):
            self.file = open(log_file, mode, encoding=encoding)
            self.stdout = sys.stdout

        def write(self, message):
            self.stdout.write(message)  # 输出到控制台
            self.file.write(message)  # 写入文件

        def flush(self):
            self.stdout.flush()
            self.file.flush()

        def close(self):
            self.file.close()


    # ===== 使用示例 =====
    beijing_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() + 8 * 3600))
    log_file = f"E:/he_gait_recognization/out_{beijing_time}.log"

    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee

    print("这条信息会同时出现在控制台和 log 文件中")
    # ====================== 【自定义】日志保存的文件夹 ======================
    # log_dir = r"E:\he_gait_recognization\DFQGait\logs"  # 你想存哪里改这里
    # # =====================================================================
    #
    # # 创建文件夹
    # os.makedirs(log_dir, exist_ok=True)
    #
    # # 获取 北京时间（UTC+8）
    # beijing_tz = timezone(timedelta(hours=8))
    # beijing_time = datetime.now(beijing_tz).strftime("log_%Y%m%d_%H%M%S")
    # log_file = os.path.join(log_dir, f"{beijing_time}.log")
    #
    # # 重定向控制台输出到日志
    # sys.stdout = open(log_file, 'w', encoding='utf-8')
    # sys.stderr = sys.stdout
    #
    # # 打印你要看到的那句话
    # print(f"Console output is saving to: {log_file}")

    # ============================================================


    def force_clear_cuda():
        for var in dir():
            obj = eval(var)
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("CUDA Memory Cleared!")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '996'
    os.environ["USE_LIBUV"] = "0"
    torch.distributed.init_process_group('gloo', init_method='env://', rank=0, world_size=1)
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)


    # 重新建一个模型（和 run_model 里一模一样）

    run_model(cfgs, training)