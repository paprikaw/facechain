
from modelscope_agent.tools import Tool
character_model = 'ly261666/cv_portrait_model'
import enum
import os
import json
import shutil
import slugify
import time
from concurrent.futures import ProcessPoolExecutor
from torch import multiprocessing
import cv2
import gradio as gr
import numpy as np
import torch
from glob import glob
import platform
import subprocess
from facechain.utils import snapshot_download

from facechain.inference import preprocess_pose, GenPortrait
from facechain.inference_inpaint import GenPortrait_inpaint
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt as neg, pos_prompt_with_cloth, pos_prompt_with_style, \
    pose_models, pose_examples, base_models
styles=[]
for base_model in base_models:
    style_in_base = []
    folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r",encoding='utf-8') as f:
            data = json.load(f)
            style_in_base.append(data['name'])
            styles.append(data)
    base_model['style_list'] = style_in_base
atraining_done_count = 0
inference_done_count = 0
character_model = 'ly261666/cv_portrait_model'
BASE_MODEL_MAP = {
    "leosamsMoonfilm_filmGrain20": "写实模型(Realistic model)",
    "MajicmixRealistic_v6": "\N{fire}写真模型(Photorealistic model)",
}

def launch_pipeline(uuid,
                    pos_prompt,
                    neg_prompt=None,
                    base_model_index=None,
                    user_model=None,
                    num_images=1,
                    lora_choice=None,
                    style_model=None,
                    multiplier_style=0.35,
                    multiplier_human=0.95,
                    pose_model=None,
                    pose_image=None
                    ):
    print("-------uuid: ", uuid)
    print("-------pos_prompt: ", pos_prompt)
    print("-------neg_prompt: ", neg_prompt)
    print("-------base_model_index: ", base_model_index)
    print("-------user_model: ", user_model)
    print("-------num_images: ", num_images)
    print("-------lora_choice: ", lora_choice)
    print("-------style_model: ", style_model)
    print("-------multiplier_style: ", multiplier_style)
    print("-------multiplier_human: ", multiplier_human)
    print("-------pose_model: ", pose_model)
    print("-------pose_image: ", pose_image)

    uuid = 'qw'

    # Check base model
    if base_model_index == None:
        raise '请选择基模型(Please select the base model)！'

    # Check character LoRA
    folder_path = f"./{uuid}/{character_model}"
    folder_list = []
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)
    if len(folder_list) == 0:
        raise '没有人物LoRA，请先训练(There is no character LoRA, please train first)!'

    # Check output model
    if user_model == None:
        raise '请选择人物LoRA(Please select the character LoRA)！'
    # Check lora choice
    # if lora_choice != 'preset':
    #     raise '请选择LoRA模型(Please select the LoRA model)!'
    # Check style model
    if style_model == None and lora_choice == 'preset':
        raise '请选择风格模型(Please select the style model)!'

    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']

    before_queue_size = 0
    before_done_count = inference_done_count
    matched = list(filter(lambda item: style_model == item['name'], styles))
    if len(matched) == 0:
        raise ValueError(f'styles not found: {style_model}')
    matched = matched[0]
    style_model = matched['name']

    if lora_choice == 'preset':
        if matched['model_id'] is None:
            style_model_path = None
        else:
            model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
            style_model_path = os.path.join(model_dir, matched['bin_file'])
    else:
        print(f'uuid: {uuid}')
        temp_lora_dir = f"./{uuid}/temp_lora"
        file_name = lora_choice
        print(lora_choice.split('.')[-1], os.path.join(temp_lora_dir, file_name))
        if lora_choice.split('.')[-1] != 'safetensors' or not os.path.exists(os.path.join(temp_lora_dir, file_name)):
            raise ValueError(f'Invalid lora file: {lora_file.name}')
        style_model_path = os.path.join(temp_lora_dir, file_name)

    if pose_image is None or pose_model == 0:
        pose_model_path = None
        use_depth_control = False
        pose_image = None
    else:
        model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
        pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')
        if pose_model == 1:
            use_depth_control = True
        else:
            use_depth_control = False

    print("-------user_model: ", user_model)

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    instance_data_dir = os.path.join('./', uuid, 'training_data', character_model, user_model)
    lora_model_path = f'./{uuid}/{character_model}/{user_model}/'
    print('----------======================')
    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, instance_data_dir,
                                 num_images, base_model, lora_model_path, sub_path, revision)
        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                print("排队等待资源中, 前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                       None)
            else:
                print("生成中, 请耐心等待(Generating)...", None)
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    save_dir = os.path.join('./', uuid, 'inference_result', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # use single to save outputs
    if not os.path.exists(os.path.join(save_dir, 'single')):
        os.makedirs(os.path.join(save_dir, 'single'))
    for img in outputs:
        # count the number of images in the folder
        num = len(os.listdir(os.path.join(save_dir, 'single')))
        cv2.imwrite(os.path.join(save_dir, 'single', str(num) + '.png'), img)

    if len(outputs) > 0:
        result = concatenate_images(outputs)
        if not os.path.exists(os.path.join(save_dir, 'concat')):
            os.makedirs(os.path.join(save_dir, 'concat'))
        num = len(os.listdir(os.path.join(save_dir, 'concat')))
        image_path = os.path.join(save_dir, 'concat', str(num) + '.png')
        cv2.imwrite(image_path, result)

        print("生成完毕(Generation done)!", outputs_RGB)
        return
    else:
        print("生成失败, 请重试(Generation failed, please retry)!", outputs_RGB)
        return
class FaceChainInferenceTool(Tool):
    description = "模型微调推理，根据用户人脸Lora以及输入的期望风格输出图片"
    name = "facechain_Inference_tool"
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True
    }]

    def __init__(self):
        super().__init__()
        # self.base_model_path = 'ly261666/cv_portrait_model'
        # self.revision = 'v2.0'
        # self.sub_path = "film/film"
        # # 这里固定了Lora的名字,重新训练会覆盖原来的
        # self.lora_name = "person1"

    def _remote_call(self, *args, **kwargs):
        pass

    def _local_call(self, *args, **kwargs):
        print(123)
        pos_prompt = 'raw photo, masterpiece, chinese, wearing silver armor, simple background, high-class pure color background, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality'
        neg_prompt = '(nsfw:2), paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, bad hand, tattoo, (username, watermark, signature, time signature, timestamp, artist name, copyright name, copyright),low res, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands))'
        base_model_index = 0
        user_model = 'person1'
        num_images = 1
        lora_choice =  'preset'
        style_model = '盔甲风(Armor)'
        multiplier_style = 0.35
        multiplier_human = 0.95
        pose_model = 0
        pose_image = None

        launch_pipeline(uuid=None,pos_prompt=pos_prompt,
                        neg_prompt=neg_prompt,base_model_index=base_model_index,
                        user_model=user_model,num_images=num_images,
                        style_model=style_model,multiplier_style=multiplier_style,
                        multiplier_human=multiplier_human,pose_model=pose_model,
                        pose_image=pose_image,lora_choice=lora_choice
                        )

if __name__ == "__main__":
    tool = FaceChainInferenceTool()
    tool._local_call()