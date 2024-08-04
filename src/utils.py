import random
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain


@dataclass
class InteractionArgs:
    has_intro: Optional[bool] = field(default=True)
    do_shuffle: Optional[bool] = field(default=False)
    max_hist_len: Optional[int] = field(default=None)
    first_round_eg: Optional[bool]=field(default=False)
    model_type: Optional[str]=field(default=None)
    img_once: Optional[bool]=field(default=False)
    img_mask: Optional[bool]=field(default=False)
    no_history: Optional[bool]=field(default=False)
    misleading_trials: Optional[list]=field(default=None)
    #TODO think about how to do round name. should be dependnet on hist length


@dataclass
class ModelArgs:
    role: Optional[str] = field(default='spkr')
    model_ckpt: Optional[str] = field(default=None)
    max_output_tokens: Optional[int] = field(default=30)
    img_mode: Optional[str] = field(default=None)
    label_space: Optional[list] = field(default=None)
    intro_text: Optional[str]=field(default=None)
    altnernating_roles: Optional[bool]=field(default=True)
    


def masking_images(lsnr_context_imgs, img_mask, img_mask_url, img_mask_base64):
    for img in lsnr_context_imgs:
        img['URL']=img_mask_url
        img['PIL']=img_mask
        img['base64_string']=img_mask_base64
    
    return lsnr_context_imgs

    
        
