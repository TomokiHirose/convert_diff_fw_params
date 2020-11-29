# convert_diff_fw_params
This repo's trying to copy TorchVision's Pre-trained parameters to a model created with TensorFlow.

Ideas for transfer

```python
weights = torch_model.state_dict()["xxx.weight"].numpy().transpose((2, 3, 1, 0))
bias = torch_model.state_dict()["xxx.bias"].numpy()
tf_layer.set_weights([weights, bias])
```

## Hot to check transfer

So very easy.
Plz, clone this repository.

Next, build the environment.
This repository is managed by Peotry, so install it if necessary.

```
$ pip install poetry
```

And, Build Environment.

```
$ poetry install
```

After the environment is built, check the operation with the command below.

```
$ poetry run test
```


The source code for operation is located in ./tests/test_models.py
Please refer to here for the mounting method.