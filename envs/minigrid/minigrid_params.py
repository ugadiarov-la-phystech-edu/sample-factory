def minigrid_override_default_params_and_args(params, args):
    params.obs_subtract_mean = 0.0
    params.obs_scale = 1.0

    params.conv_filters = [
        [3, 16, 2, 1],
        'maxpool_2x2',
        [16, 32, 2, 1],
        [32, 64, 2, 1],
    ]

    params.hidden_size = 256

    if 'render_action_repeat' in args:
        if args.render_action_repeat is None:
            args.render_action_repeat = 1
