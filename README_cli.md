0. Installation:

```
conda create -n framepack python=3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Then (or if conda environment is already set up):

```
conda activate framepack
```

1. With text prompt:

```
python generate_f1_next_frames.py \
  --image /home/zemskova-ts/FramePack/sample_image.jpg \
  --frames 33 \
  --prompt "The man dances energetically, leaping mid-air with fluid arm swings and quick footwork." \
  --use-teacache \
  --output-dir ./outputs/f1_next_frames \
  --save-mp4
```

2. Without text prompt:


```
python generate_f1_next_frames.py \
  --image /home/zemskova-ts/FramePack/sample_image.jpg \
  --frames 33 \
  --prompt "" \
  --use-teacache \
  --output-dir "/home/zemskova-ts/FramePack/outputs/f1_next_frames_no_text_prompt" \
  --save-mp4
```

3. ActionGenome without text prompt:

```
python generate_f1_next_frames.py \
  --image /home/zemskova-ts/FramePack/00N38_000184.png \
  --frames 33 \
  --prompt "" \
  --use-teacache \
  --output-dir "/home/zemskova-ts/FramePack/outputs/f1_next_frames_AG_no_text_prompt" \
  --save-mp4
```

4. ActionGenome with text prompt: 

```
python generate_f1_next_frames.py \
  --image /home/zemskova-ts/FramePack/00N38_000184.png \
  --frames 33 \
  --prompt "The person is wearing the clothes." \
  --use-teacache \
  --output-dir "/home/zemskova-ts/FramePack/outputs/f1_next_frames_AG_with_text_prompt" \
  --save-mp4
```