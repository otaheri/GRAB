
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import yaml
import os

class Config(dict):


    def __init__(self,default_cfg_path=None,**kwargs):

        default_cfg = {}
        if default_cfg_path is not None and os.path.exists(default_cfg_path):
            default_cfg = self.load_cfg(default_cfg_path)

        super(Config,self).__init__(**kwargs)

        default_cfg.update(self)
        self.update(default_cfg)
        self.default_cfg = default_cfg

    def load_cfg(self,load_path):
        with open(load_path, 'r') as infile:
            cfg = yaml.safe_load(infile)
        return cfg if cfg is not None else {}

    def write_cfg(self,write_path=None):

        if write_path is None:
            write_path = 'yaml_config.yaml'

        dump_dict = {k:v for k,v in self.items() if k!='default_cfg'}
        makepath(write_path, isfile=True)
        with open(write_path, 'w') as outfile:
            yaml.safe_dump(dump_dict, outfile, default_flow_style=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

if __name__ == '__main__':

    cfg = {
        'intent': 'all',
        'only_contact': True,
        'save_body_verts': False,
        'save_object_verts': False,
        'save_contact': False,
    }

    cfg = Config(**cfg)
    cfg.write_cfg()