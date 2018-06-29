#!/usr/bin/env python3

class image_properties(object):
    depth = 3
    height = 512
    width = 512
    compressed_dims = [1, 32, 32, 8]

class config_train(object):
    mode = 'gan-train'
    num_epochs = 10
    batch_size = 5
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 1

    # WGAN
    gradient_penalty = True
    lambda_gp = 10
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 20

    # Compression
    lambda_X = 12
    channel_bottleneck = 8
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

class config_test(object):
    mode = 'gan-test'
    num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256

    # WGAN
    gradient_penalty = True
    lambda_gp = 10
    weight_clipping = False
    max_c = 1e-2
    n_critic_iterations = 5

    # Compression
    lambda_X = 12
    channel_bottleneck = 8
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

class directories(object):
    train = 'data/faces_paths_train.d5'
    test = 'data/faces_paths_test.d5'
    val = 'data/faces_paths_validation.d5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    samples = 'samples/'

