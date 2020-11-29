import fw_convert.models as models
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import random
import pathlib


def test_vgg11():
    # TensorFlowのVGGモデルの準備
    tf_model = models.tf_vgg_model()
    tf_layers = [l for l in tf_model.layers if "Conv2D" in l.__class__.__name__ or "Dense" in l.__class__.__name__]

    # PytorchのVGGモデルの準備
    torch_model = models.torch_vgg_model()
    torch_model.eval()
    layer_name = [name for name, _ in torch_model.state_dict().items()]

    for tf_layer, name in zip(tf_layers, np.reshape(layer_name, (-1, 2))):
        if "features" in name[0]:
            tf_layer.set_weights([torch_model.state_dict()[name[0]].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()[name[1]].numpy()])
        elif "classifier" in name[0]:
            tf_layer.set_weights([torch_model.state_dict()[name[0]].numpy().transpose((1, 0)), torch_model.state_dict()[name[1]].numpy()])

    # 画像データの準備
    image = Image.open(str(pathlib.Path("./data/sample.jpg")))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    torch_image = preprocess(image)
    torch_image.unsqueeze_(0)
    tf_image = torch_image.detach().numpy().transpose(0, 2, 3, 1)

    ## 推論
    torch_output = torch_model(torch_image)
    tf_output = tf_model.predict(tf_image)

    # ImageNetラベルファイルを取得
    labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    label = imagenet_labels[np.argmax(tf_output[0], -1)]

    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(image)
    plt.axis("off")
    _ = plt.title("Prediction: " + label.title())
    plt.savefig("figure.png")


def test_get_torch_matrix():
    """Pytorchのレイヤーから行列を取り出す"""
    try:
        torch_conv2d = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3)
        print(torch_conv2d.weight.numpy())
    except Exception as e:
        print(e)
    finally:
        print(torch_conv2d.weight.detach().numpy().shape)


def test_set_tf_matrix():
    """TensorFlowのレイヤーに行列を突っ込む"""
    inputs = tf.keras.Input(
        shape=(28, 28, 5),
    )
    tf_conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=tf_conv2d)
    print(len(model.layers[1].get_weights()))
    print([type(item) for item in model.layers[1].get_weights()])

    print(model.layers[1].get_weights()[0].shape)
    print(model.layers[1].get_weights()[1].shape)


def test_channek_last_and_first():
    """channel lastとchannel first"""
    inputs = tf.keras.Input(
        shape=(28, 28, 5),
    )
    tf_conv2d = tf.keras.layers.Conv2D(filters=64, kernel_size=3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=tf_conv2d)

    torch_conv2d = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3)
    torch_np = torch_conv2d.weight.detach().numpy()

    assert model.layers[1].get_weights()[0].shape == torch_np.transpose((2, 3, 1, 0)).shape


def test_fc_network():
    torch_model = models.TorchSimpleFCNet()
    tf_model = models.TFSimpleFCNet()

    tf_model.layers[1].set_weights([torch_model.state_dict()["fc1.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc1.bias"].numpy()])
    tf_model.layers[2].set_weights([torch_model.state_dict()["fc2.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc2.bias"].numpy()])

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    image = X_train[0].reshape(1, -1).astype(np.float32)
    img_torch = torch.tensor(image)
    # tfの出力
    tf_output = tf_model.predict(image)

    # eval関数を使用後torchのTensorに変換して入力
    torch_model.eval()
    torch_output = torch_model(img_torch)
    # 出力ベクトルの差の最大を取得
    assert np.max(np.abs(tf_output - torch_output.detach().numpy())) < 0.01


def test_simple_network():
    torch_model = models.TorchSimpleNet()
    tf_model = models.TFSimpleNet()

    tf_model.layers[1].set_weights(
        [torch_model.state_dict()["conv1.weight"].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()["conv1.bias"].numpy()]
    )
    tf_model.layers[2].set_weights(
        [torch_model.state_dict()["conv2.weight"].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()["conv2.bias"].numpy()]
    )

    tf_model.layers[5].set_weights([torch_model.state_dict()["fc1.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc1.bias"].numpy()])
    tf_model.layers[6].set_weights([torch_model.state_dict()["fc2.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc2.bias"].numpy()])

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    image = np.expand_dims(X_train[0].reshape(1, 28, 28), 0).astype(np.float32)

    img_torch = torch.autograd.Variable(torch.from_numpy(image.copy()).float())
    image = np.transpose(image.copy(), (0, 2, 3, 1))

    # tfの出力
    tf_output = tf_model.predict(image)

    # eval関数を使用後torchのTensorに変換して入力
    torch_model.eval()
    torch_output = torch_model(img_torch)
    # 出力ベクトルの差の最大を取得
    assert np.max(np.abs(tf_output - torch_output.detach().numpy())) < 0.01


def test_simple_permute_network():
    torch_model = models.TorchSimplePermuteNet()
    tf_model = models.TFSimplePermuteNet()

    tf_model.layers[1].set_weights(
        [torch_model.state_dict()["conv1.weight"].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()["conv1.bias"].numpy()]
    )
    tf_model.layers[2].set_weights(
        [torch_model.state_dict()["conv2.weight"].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()["conv2.bias"].numpy()]
    )

    tf_model.layers[6].set_weights([torch_model.state_dict()["fc1.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc1.bias"].numpy()])
    tf_model.layers[7].set_weights([torch_model.state_dict()["fc2.weight"].numpy().transpose((1, 0)), torch_model.state_dict()["fc2.bias"].numpy()])

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    image = np.expand_dims(X_train[0].reshape(1, 28, 28), 0).astype(np.float32)

    img_torch = torch.autograd.Variable(torch.from_numpy(image.copy()).float())
    image = np.transpose(image.copy(), (0, 2, 3, 1))

    # tfの出力
    tf_output = tf_model.predict(image)

    # eval関数を使用後torchのTensorに変換して入力
    torch_model.eval()
    torch_output = torch_model(img_torch)
    # 出力ベクトルの差の最大を取得
    assert np.max(np.abs(tf_output - torch_output.detach().numpy())) < 0.01


def test_flatten():

    torch_model = models.TorchConvFlattenNet()
    tf_model = models.TFConvFlattenNet()

    array = np.arange(0, 28 * 28).astype(dtype=np.float32)

    torch_output = torch_model(torch.tensor(array.reshape((1, 1, 28, 28)))).data.numpy()
    tf_output = tf_model(array.reshape((1, 28, 28, 1))).numpy()
    assert np.max(np.abs(tf_output - torch_output)) < 0.01


def test_conv_flatten():

    torch_model = models.TorchFlattenNet()
    tf_model = models.TFFlattenNet()

    tf_model.layers[1].set_weights(
        [torch_model.state_dict()["conv1.weight"].numpy().transpose((2, 3, 1, 0)), torch_model.state_dict()["conv1.bias"].numpy()]
    )

    array = np.arange(0, 28 * 28).astype(dtype=np.float32)

    torch_output = torch_model(torch.tensor(array.reshape((1, 1, 28, 28)))).data.numpy()
    tf_output = tf_model(array.reshape((1, 28, 28, 1))).numpy()
    assert np.max(np.abs(tf_output - torch_output)) < 0.01
