
## scorer.py
`Scorer` is tool class that leverages metric functions from [COCO-Caption Evaluation](https://github.com/ruotianluo/coco-caption.git). Here are steps to setup the requirements:

1. Clone the repository into `coco_caption`: `git clone https://github.com/ruotianluo/coco-caption.git coco_caption`
2. `cd coco_caption`
3. Convert the folder to a package: `touch __init__.py`
4. Download model files for SPICE: `bash get_stanford_models.sh`
5. Install java1.8 runtime: `sudp apt update; sudo apt install openjdk-8-jdk`
6. Download model files for WMD: `bash get_google_word2vec_model.sh`
7. Install python packages: `pip install gensim pot`
8. Test if everything works: `python scorer.py`

Please refer to this [blog](https://blog.csdn.net/weixin_41848012/article/details/121254472) for more details.