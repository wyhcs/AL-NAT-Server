# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select an encoder network."""


def build_encoder(args):

    if 'conv' in args.enc_type:
        assert args.n_stacks == 1 and args.n_splices == 1
        from models.seq2seq.encoders.conv import ConvEncoder
        conv = ConvEncoder(args.input_dim,
                           in_channel=args.conv_in_channel,
                           channels=args.conv_channels,
                           kernel_sizes=args.conv_kernel_sizes,
                           strides=args.conv_strides,
                           poolings=args.conv_poolings,
                           dropout=0.,
                           normalization=args.conv_normalization,
                           residual=False,
                           bottleneck_dim=args.transformer_enc_d_model if 'former' in args.enc_type else args.conv_bottleneck_dim,
                           param_init=args.param_init)
    else:
        conv = None

    # safeguard
    if not hasattr(args, 'transformer_enc_d_model') and hasattr(args, 'transformer_d_model'):
        args.transformer_enc_d_model = args.transformer_d_model
        args.transformer_dec_d_model = args.transformer_d_model
    if not hasattr(args, 'transformer_enc_d_ff') and hasattr(args, 'transformer_d_ff'):
        args.transformer_enc_d_ff = args.transformer_d_ff
    if not hasattr(args, 'transformer_enc_n_heads') and hasattr(args, 'transformer_n_heads'):
        args.transformer_enc_n_heads = args.transformer_n_heads

    if 'transformer' in args.enc_type:
        from models.seq2seq.encoders.transformer import TransformerEncoder
        encoder = TransformerEncoder(
            input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
            enc_type=args.enc_type,
            n_heads=args.transformer_enc_n_heads,
            n_layers=args.enc_n_layers,
            n_layers_sub1=args.enc_n_layers_sub1,
            n_layers_sub2=args.enc_n_layers_sub2,
            d_model=args.transformer_enc_d_model,
            d_ff=args.transformer_enc_d_ff,
            ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim,
            ffn_activation=args.transformer_ffn_activation,
            pe_type=args.transformer_enc_pe_type,
            layer_norm_eps=args.transformer_layer_norm_eps,
            last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else 0,
            dropout_in=args.dropout_in,
            dropout=args.dropout_enc,
            dropout_att=args.dropout_att,
            dropout_layer=args.dropout_enc_layer,
            subsample=args.subsample,
            subsample_type=args.subsample_type,
            n_stacks=args.n_stacks,
            n_splices=args.n_splices,
            frontend_conv=conv,
            task_specific_layer=args.task_specific_layer,
            param_init=args.transformer_param_init,
            clamp_len=args.transformer_enc_clamp_len,
            lookahead=args.transformer_enc_lookaheads,
            chunk_size_left=args.lc_chunk_size_left,
            chunk_size_current=args.lc_chunk_size_current,
            chunk_size_right=args.lc_chunk_size_right,
            streaming_type=args.lc_type)

    elif 'conformer' in args.enc_type:
        from models.seq2seq.encoders.conformer import ConformerEncoder
        encoder = ConformerEncoder(
            input_dim=args.input_dim if args.input_type == 'speech' else args.emb_dim,
            enc_type=args.enc_type,
            n_heads=args.transformer_enc_n_heads,
            kernel_size=args.conformer_kernel_size,
            normalization=args.conformer_normalization,
            n_layers=args.enc_n_layers,
            n_layers_sub1=args.enc_n_layers_sub1,
            n_layers_sub2=args.enc_n_layers_sub2,
            d_model=args.transformer_enc_d_model,
            d_ff=args.transformer_enc_d_ff,
            ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim,
            ffn_activation='swish',
            pe_type=args.transformer_enc_pe_type,
            layer_norm_eps=args.transformer_layer_norm_eps,
            last_proj_dim=args.transformer_dec_d_model if 'transformer' in args.dec_type else 0,
            dropout_in=args.dropout_in,
            dropout=args.dropout_enc,
            dropout_att=args.dropout_att,
            dropout_layer=args.dropout_enc_layer,
            subsample=args.subsample,
            subsample_type=args.subsample_type,
            n_stacks=args.n_stacks,
            n_splices=args.n_splices,
            frontend_conv=conv,
            task_specific_layer=args.task_specific_layer,
            param_init=args.transformer_param_init,
            clamp_len=args.transformer_enc_clamp_len,
            lookahead=args.transformer_enc_lookaheads,
            chunk_size_left=args.lc_chunk_size_left,
            chunk_size_current=args.lc_chunk_size_current,
            chunk_size_right=args.lc_chunk_size_right,
            streaming_type=args.lc_type)

    return encoder
