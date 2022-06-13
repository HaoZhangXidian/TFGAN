# TFGAN
This repository is for our TNNLS submission: TFGAN

The dataset used in the paper can be downloaded from https://drive.google.com/file/d/1ICP01QH1UWcivUpTocr4EfWcW91PQz27/view?usp=sharing

In this demo, we provided how to run TFGAN based on COCO and CUB dataset. 
Once paper is receved, we will release all codes for other experiments.

Firstly, you can run "main_COCO_caption_gpt2.py" or "main_cub_caption_gpt2.py" to finetune the gpt2 on COCO and CUB (Optional)

Then you can try "main_COCO_caption_lstmreg_gan.py" and "main_CUB_caption_lstmreg_gan.py" to distill from gpt2 to lstm by our proposed TFGAN.

mainTest_cub_caption.py is a test demo on CUB.
