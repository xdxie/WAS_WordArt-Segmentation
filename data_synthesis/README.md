## (Step 1) Install

First create a new conda environment

    conda env create -f environment.yaml
    conda activate control

The trained ControlNet checkpoint for generating images with artistic text masks can be downloaded from [Hugging Face page](https://huggingface.co/datasets/AlanYeager/WAS). Ensure that the checkpoint is placed in "ControlNet/models" and the [detectors](https://huggingface.co/lllyasviel/ControlNet/tree/main/) are placed in "ControlNet/annotator/ckpts".

## (Step 2) Artistic Text Mask Generation

Then generate text masks with scaling, rotation, and perspective transformation.

1. **Prepare fonts**  
   Put `.ttf` font files into the `fonts4mask/` folder.

2. **Customize**  
   Edit parameters in `__main__` of `mask_render.py`.

3. **Run example**  
   ```bash
   python mask_render.py

## (Step 3) Inference

With masks generated in Step 2, please prepare a JSON file  (`prompts.json`) with one prompt per mask image. Each line must be a JSON object containing a `"prompt"` field.
Before you run the inference, please make sure mask images are stored in the --source_folder (e.g., ./masks) and that the number of lines in prompts.json matches the number of mask images.
**Run**
   ```bash
   python inference.py \
       --ckpt ./models/epoch=999-step=41999.ckpt.ckpt \
       --config ./models/cldm_v15.yaml \
       --source_folder ./masks \
       --target_folder ./images \
       --prompt_file ./prompts.json \
       --image_resolution 648 \
       --ddim_steps 15
