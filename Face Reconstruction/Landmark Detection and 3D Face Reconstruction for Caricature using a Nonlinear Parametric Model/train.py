import cariface
from train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()

    model = cariface.CariFace()
    model.init_numbers(opt.landmark_num, opt.vertex_num, opt.device_num)
    model.init_data(opt.data_path)
    if opt.if_train == True:
        model.load_train_data(opt.train_image_path, opt.train_landmark_path, opt.train_vertex_path,
                                opt.batch_size, opt.num_workers)
        model.load_test_data(opt.test_image_path, opt.test_landmark_path,
                                opt.test_lrecord_path, opt.test_vrecord_path, opt.num_workers)
        model.load_model(opt.resnet34_lr, opt.mynet1_lr, opt.mynet2_lr, opt.use_premodel,
                            opt.model1_path, opt.model2_path)
        model.test()
        for epoch in range(1, opt.total_epoch+1):
            model.train(epoch, opt.lambda_land, opt.lambda_srt)
            if epoch % opt.test_frequency == 0:
                model.test()
            if epoch % opt.save_frequency == 0:
                model.save_model(epoch, opt.save_model_path)
        model.save_model(opt.total_epoch, opt.save_model_path)
    else:
        model.load_test_data(opt.test_image_path, opt.test_landmark_path,
                                opt.test_lrecord_path, opt.test_vrecord_path, opt.num_workers)
        model.load_model(opt.resnet34_lr, opt.mynet1_lr, opt.mynet2_lr, opt.use_premodel,
                            opt.model1_path, opt.model2_path)
        model.test()