docker run --rm -it --gpus '"device=0"'  \
    -v /home/yca/CLIP:/workspace/CLIP \
    clip:1.0 bash
