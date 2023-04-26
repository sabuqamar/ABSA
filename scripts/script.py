#!/usr/bin/env python
from enum import Enum
import warnings
import sys

warnings.filterwarnings('ignore')


# Add the parent directory of ABSA to the module search path
sys.path.append('./')

from models.InstructABSA.InstructABSA.utils import T5Generator
from models.InstructABSA.instructions import InstructionsHandler


class Task(Enum):
    ATE = 1
    JOINT = 2

def return_iabsa(task, model_checkpoint, set_instruction_key, sent):
    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set1() if set_instruction_key == 1 else instruct_handler.load_instruction_set2()
    max_token_length = 128

    indomain = 'bos_instruct1' if set_instruction_key == 1 else 'bos_instruct2'
    t5_exp = T5Generator(model_checkpoint)

    if task == Task.ATE:
        bos_instruction_id = instruct_handler.ate[indomain]
    elif task == Task.JOINT:
        bos_instruction_id = instruct_handler.joint[indomain]

    eos_instruction = instruct_handler.ate['eos_instruct']
    model_input = bos_instruction_id + sent + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length = max_token_length)

    model_output = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)

    aspects = model_output.split(", ")
    if task == Task.JOINT:
        asp_list, pol_list = [], []
        for asp in aspects:
            asp_split = asp.split(":")
            asp_list.append(asp_split[0])
            pol_list.append(asp_split[1])
        return asp_list, pol_list

    return aspects
