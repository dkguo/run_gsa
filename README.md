### Install Grounded SAM
Follow https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker, install without Docker
```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
git submodule update --init --recursive

conda create -n gsa python=3.8.18
conda activate gsa

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
# cuda should be already installed
# export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:
```
python -m pip install -e segment_anything
```

Install Grounding DINO:
```
python -m pip install -e GroundingDINO
```

Install diffusers:
```
# pip install --upgrade diffusers[torch] # torch version could not be found
pip install --upgrade diffusers
```

Install osx: \
There could be errors about mmcv-full compilation, but it is not used in the code.
```
cd grounded-sam-osx && bash install.sh
```

Install RAM & Tag2Text:
```
cd Tag2Text && pip install -r requirements.txt
```

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

### Download pretrained models
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Try the demo
```
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
```

### Config run_gsa
Before running, you need to change a few things in `/run_gsa`:
1. Change `gsa_path` in `predicor.py` to the path of Grounded-Segment-Anything
2. Change `midman_address` in `midman.py` 
3. Change `server_address` in `request_predictions.py`

### run_gsa
Run `python midman.py` to start the midman server. \
Run `python run_predictors.py X` to load X predictors. \
Use `request_predictions.py` to send requests to the midman server.
