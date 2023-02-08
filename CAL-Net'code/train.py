import datetime
import os
from functools import partial
from transunet.model import TransUNet
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam
from nets.unetv4 import Unetv4
from nets.unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss, get_lr_scheduler)
from utils.callbacks import (EvalCallback, ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import UnetDataset
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import Iou_score, f_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":    
    eager           = False
    train_gpu       = [0,]
    num_classes     = 2
    backbone        = "inceptionv4"
    model_path      = ""
    input_shape     = [256, 256]
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    UnFreeze_Epoch      = 120
    Unfreeze_batch_size = 2
    Freeze_Train        = False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 1
    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = True
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 1
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")
        
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    
    if ngpus_per_node > 1:
        with strategy.scope():
            # model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
            model = TransUNet(256)
            if model_path != '':

                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:

        model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
        # model = TransUNet(256)
        if model_path != '':

            model.load_weights(model_path, by_name=True, skip_mismatch=True)


    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )


    if True:
        if Freeze_Train:

            if backbone == "vgg":
                freeze_layers = 17
            elif backbone == "resnet50":
                freeze_layers = 172
            else:
                raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
            for i in range(freeze_layers): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)
        val_dataloader      = UnetDataset(val_lines, input_shape, batch_size, num_classes, False, VOCdevkit_path)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False

            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
            if ngpus_per_node > 1:
                gen     = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)

            for epoch in range(start_epoch, end_epoch):

                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    nbs             = 16
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model.layers)): 
                        model.layers[i].trainable = True

                    epoch_step      = num_train // batch_size
                    epoch_step_val  = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    train_dataloader.batch_size    = batch_size
                    val_dataloader.batch_size      = batch_size

                    gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    if ngpus_per_node > 1:
                        gen     = strategy.experimental_distribute_dataset(gen)
                        gen_val = strategy.experimental_distribute_dataset(gen_val)
                    
                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)
                
                fit_one_epoch(model, loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, f_score(), save_period, save_dir, strategy)

                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()

        else:
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
                
            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(loss = loss,
                            optimizer = optimizer,
                            metrics = [f_score()])
            else:
                model.compile(loss = loss,
                        optimizer = optimizer,
                        metrics = [f_score()])

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = False, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = False, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = False, save_best_only = True, period = 1)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)
            callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )

            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]
                    
                for i in range(len(model.layers)): 
                    model.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(loss = loss,
                                optimizer = optimizer,
                                metrics = [f_score()])
                else:
                    model.compile(loss = loss,
                            optimizer = optimizer,
                            metrics = [f_score()])

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_dataloader.batch_size    = Unfreeze_batch_size
                val_dataloader.batch_size      = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
