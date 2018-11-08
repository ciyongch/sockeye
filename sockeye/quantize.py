# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for quantization in inference
"""
import mxnet as mx
import logging
import os
import time
from typing import Optional
from . import utils

logger = logging.getLogger(__name__)


g_encoder_sym_time = []
g_decoder_sym_time = []

class Quantize(object):
    def __init__(self,
                 params,
                 aux_params,
                 excluded_encoder_sym_names : Optional[list] = None,
                 excluded_decoder_sym_names : Optional[list] = None,
                 dtype : str = 'int8') -> None:
        self.params = params
        self.aux_params = aux_params
        self.excluded_encoder_sym_names = excluded_encoder_sym_names
        self.excluded_decoder_sym_names = excluded_decoder_sym_names
        self.dtype = dtype
        self.logger = logger

    def save_symbol(self, fname, sym, logger=None) -> None:
        if logger is not None:
            logger.info('Saving symbol into file at %s' % fname)
        sym.save(fname)

    def save_params(self, fname, params, aux_params, logger=None) -> None:
        if logger is not None:
            logger.info('Saving params into file at %s' % fname)
        save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in params.items()}
        save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.nd.save(fname, save_dict)

    def quantize_symbol(self, sym, module='encoder', bucket_key=None):
        excluded_sym_names = None
        sym_file_name = '%s' % (module + '_quantized')

        timing_list = None

        if module is 'encoder':
            if self.excluded_encoder_sym_names is None:
                self.excluded_encoder_sym_name = []

            if not isinstance(self.excluded_encoder_sym_names, list):
                raise ValueError('excluded_encoder_sym_names must be a list of strings representing'
                                 ' the names of the symbols that will not be quantized,'
                                 ' while received type %s' % str(type(self.excluded_encoder_sym_names)))
            excluded_sym_names = self.excluded_encoder_sym_names

            sym_file_name = '%s_symbol.json' % (sym_file_name + ('_' + str(bucket_key) if
                                                    isinstance(bucket_key, int) else ''))
            global g_encoder_sym_time
            timing_list = g_encoder_sym_time

        if module is 'decoder':
            if self.excluded_decoder_sym_names is None:
                self.excluded_decoder_sym_name = []

            if not isinstance(self.excluded_decoder_sym_names, list):
                raise ValueError('excluded_decoder_sym_names must be a list of strings representing'
                                 ' the names of the symbols that will not be quantized,'
                                 ' while received type %s' % str(type(self.excluded_decoder_sym_names)))
            excluded_sym_names = self.excluded_decoder_sym_names

            sym_file_name = '%s_symbol.json' % (sym_file_name + ('_' + '-'.join(str(i) for i in bucket_key)
                                                    if isinstance(bucket_key, tuple) else ''))
            global g_decoder_sym_time
            timing_list = g_decoder_sym_time

        if self.dtype != 'int8' and self.dtype != 'uint8':
            raise ValueError('unknown quantized_dtype %s received,'
                             ' expected `int8` or `uint8`' % self.dtype)
        tic = time.time()
        qsym = mx.contrib.quant._quantize_symbol(sym, excluded_symbols=excluded_sym_names,
                                                 offline_params=list(self.params.keys()),
                                                 quantized_dtype=self.dtype)
        timing_list.append(time.time() - tic)

        return qsym

    def quantize_params(self, qsym, params):
        return mx.contrib.quant._quantize_params(qsym, params)


def initialize_quantizer(params, aux_params, dtype='int8', logger=None):
    excluded_encoder_sym_names = ["encoder_birnn_forward_l0_t0_h2h",
                                  "encoder_birnn_reverse_l0_t0_h2h",
                                  "encoder_rnn_l0_t0_h2h",
                                  "encoder_rnn_l1_t0_h2h",
                                  "encoder_rnn_l2_t0_h2h",
                                  "encoder_rnn_l3_t0_h2h",
                                  "encoder_rnn_l4_t0_h2h",
                                  "encoder_rnn_l5_t0_h2h",
                                  "encoder_rnn_l6_t0_h2h",
                                  ]
    excluded_decoder_sym_names = ["logits",
                                  ]
    quant = Quantize(params,
                     aux_params,
                     excluded_encoder_sym_names,
                     excluded_decoder_sym_names,
                     dtype)

    return quant


def load_quantized_symbols(path):
    utils.check_condition(os.path.isdir(path), "%s is not a folder or is not existing" % path)

    tic = time.time()
    files = os.listdir(path)
    encoder_syms = {}
    decoder_syms = {}
    encoder_sym_count = 0
    decoder_sym_count = 0
    for f in files:
        f = os.path.join(path, f)
        if not os.path.isdir(f):
            sym_name, _ = os.path.splitext(f)
            if sym_name.find('encoder') != -1:
                encoder_syms[sym_name] = mx.symbol.load(f)
                encoder_sym_count += 1
            if sym_name.find('decoder') != -1:
                decoder_syms[sym_name] = mx.symbol.load(f)
                decoder_sym_count += 1

    load_time = time.time() - tic
    logger.info('Success to load %d calib_encoder_syms and %d calib_decoder syms in %.4fs',
                encoder_sym_count, decoder_sym_count, load_time)
    return encoder_syms, decoder_syms
