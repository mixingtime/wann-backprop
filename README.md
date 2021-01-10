# WANN in Autograd and PyTorch
This repository contains code for the blog post ["Training Weight Agnostic Neural Networks with Backpropagation"](https://mixingtime.github.io/2019/12/28/training-wanns-with-backprop/). It focuses on WANN experiments on MNIST dataset. 

The code is based on [WANN tool](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANNTool)


## Dependencies
In addition to the dependencies of WANN tool, you will need:

- Autograd 1.3

- PyTorch 1.7.0

The code has been tested with Python 3.8.3.



## Training WANN
To train WANN with Autograd, run

`python train_ag.py mnist256train -f zoo/mnist256.wann.json --opt [adam|sgd]`

To train WANN with PyTorch, run

`python train_th.py mnist256train -f zoo/mnist256.wann.json --opt [adam|sgd]` 

If your machine comes with a GPU, you may speed up training with the `--device gpu` flag, as in

`python train_th.py mnist256train -f zoo/mnist256.wann.json --opt [adam|sgd] --device gpu` 

By default, the above commands train WANN for 1 epoch. To train it for longer, say 10 epochs, use the flag `-e 10`. 

Also, by default, the trained model and training/validation statistics are saved in `sgd_zoo/mnist.wann.[adam|sgd].exp.1.json`. 


## Testing WANN after training

To test the trained model on the test set of MNIST, run

`python test_model.py mnist256test --test_file sgd_zoo/mnist.wann.[adam|sgd].exp.1.json`

---

### Citation
If you find the blog and code useful, please cite them as

```
@article{wannbp2019,
  title  = {Training Weight Agnostic Neural Networks with Backpropagation}, 
  url    = {https://mixingtime.github.io/2019/12/28/training-wanns-with-backprop},  
  note   = "\url{https://mixingtime.github.io/2019/12/28/training-wanns-with-backprop}",  
  year   = {2019}  
}
```


