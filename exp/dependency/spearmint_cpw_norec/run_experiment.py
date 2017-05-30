'''
Created on Apr 29, 2016

@author: jch
'''

import os
import logging
import pprint
import subprocess

from lvsr.config import Configuration
from lvsr.main import train_multistage


logger = logging.getLogger(__name__)
# DEBUG from pykwalify is a way too verbose
logging.getLogger('pykwalify').setLevel(logging.INFO)
logging.getLogger('blocks.monitoring').setLevel(logging.INFO)


def prepare_config(params):
    # Experiment configuration
    opts = [(k.replace('/', '.'), str(v[0])) for (k, v) in params.items()]
    config = Configuration(
        'parser_config.yaml',
        '$LVSR/lvsr/configs/schema.yaml',
        opts
    )
    logger.info("Config:\n" + pprint.pformat(config, width=120))
    return config


def main(job_id, params):
    print "HOSTNAME %s" % (subprocess.check_output("hostname",
                                                   shell=True))
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    #
    # Tak mozna obsluzyc nietypowe parametry
    #
    num_enc_layers = params.pop('num_enc_layers')[0]
    dim_enc = params.pop('dim_enc')[0]
    params['net.dims_bidir'] = [[dim_enc] * num_enc_layers]
    params['net.subsample'] = [[1] * num_enc_layers]

    dim_pm = params.pop('post_merge_dims')[0]
    params['net.post_merge_dims'] = [[dim_pm]]

    base_cpw_filter_cnt = params.pop('base_cpw_filter_cnt')[0]
    params['net.bottom.char_to_word_conf.filters'] = [[
        [i, base_cpw_filter_cnt*i] for i in range(1, 7)
        ]]

    params['training.min_performance_stops'] = [[
        [1.0,  10],
        [0.30, 50],
        [0.15, 150],
        [0.125, 300],
        [0.11, 500],
        ]]

    config = prepare_config(params)
    base_exp_dir = "%s/local_storage/spearmint_chars_per_word_nr" % (os.environ['LVSR'],)
    try:
        os.mkdir(base_exp_dir)
    except:
        pass

    ml = train_multistage(config, save_path="%s/%08d_run" % (
                            base_exp_dir, job_id,),
                          bokeh_name="", params=None, start_stage="",
                          bokeh_server=None, bokeh=0, test_tag=0,
                          use_load_ext=False, override_file=False,
                          load_log=False, fast_start=False)
    return ml.log.status['best_valid_UAS']
