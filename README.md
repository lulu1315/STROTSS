### Modifications from original code :

<img src='https://raw.githubusercontent.com/lulu1315/STROTSS/master/images/lou1.jpeg?raw=true'>

```
usage : python3 Styletransfer.py content_image style_image result_image content_weight output_resolution max_scale weight_decay max_iterations loss_treshold
```

exemple : python3 Styletransfer.py content.png style.png out.png .2 1280 6 3 500 1e-5

-*content_weight/weight_decay*: content_weight is the weight for the last iteration.content_weight is multiplied by weight decay at each iteration.weight decay = 1 would give you the same weight at each scale.The original code uses 2 for weight decay.I found that using a bigger weight decay gives better results for a low content_weight (<.2)

-*output_resolution/max_scale* : the resolution at each scale is defined by output_resolution (resolution at final iteration) and max_scale.

-*max_iterations/loss_treshold* : moving to the next scale appends if abs(loss) is < loss_treshold or the number of iterations reaching max_iterations.(this is very empiric)

I also linearly interpolate the learning_rate between two min,max values at each scale (I don't think it actually makes a big difference..)

you can see some results here : https://www.youtube.com/watch?v=9174cVe-9Qk

<img src='https://raw.githubusercontent.com/lulu1315/STROTSS/master/images/lou2.jpeg?raw=true'>
<img src='https://raw.githubusercontent.com/lulu1315/STROTSS/master/images/lou3.jpeg?raw=true'>

### original readme

# Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS)
Code for the paper https://arxiv.org/abs/1904.12785, to appear CVPR 2019

Webdemo available at: http://128.135.245.233:8080/ 

## Dependencies:
* python3 >= 3.5
* pytorch >= 1.0
* imageio >= 2.2
* numpy >= 1.1

## Usage:
### Unconstrained Style Transfer:

```
python3 styleTransfer.py {PATH_TO_CONTENT} {PATH_TO_STYLE} {CONTENT_WEIGHT}
```

The default content weight is 1.0 (for the images provided my personal favorite is 0.5, but generally 1.0 works well for most inputs). The content weight is actually multiplied by 16, see section 2.5 of paper for explanation. 

The resolution of the output can be set on line 80 of styleTransfer.py; the current scale is 5, and produces outputs that are 512 pixels on the long side, setting it to 4 or 6 will produce outputs that are 256 or 1024 pixels on the long side respectively, most GPUs will run out of memory for settings of this variable above 6.

The output will appear in the same folder as 'styleTransfer.py' and be named 'output.png'

### Spatially Guided Style Transfer:

```
python3 styleTransfer.py {PATH_TO_CONTENT} {PATH_TO_STYLE} {CONTENT_WEIGHT} -gr {PATH_TO_CONTENT_GUIDANCE} {PATH_TO_STYLE_GUIDANCE}
```

guidance should take the form of two masks such as these:


Content Mask           |  Style Mask
:-------------------------:|:-------------------------:
<img height="200" src='https://github.com/nkolkin13/STROTSS/blob/master/content_guidance.jpg?raw=true'> |  <img height="200" src='https://github.com/nkolkin13/STROTSS/blob/master/style_guidance.jpg?raw=true'>


where regions that you wish to map onto each other have the same color.
