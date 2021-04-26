## <span style="font-variant:small-caps;">AutoVC</span>: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This is an unofficial implementation of <span style="font-variant:small-caps;">AutoVC</span> based on the official one.
The D-Vector and vocoder are from [yistlin/dvector](https://github.com/yistLin/dvector) and [yistLin/universal-vocoder](https://github.com/yistLin/universal-vocoder) respectively.

This implementation supports torch.jit, so the full model can be loaded with simply one line:

```python
model = torch.jit.load(model_path)
```

Pre-trained models are available [here](https://drive.google.com/drive/folders/1YbLqrdTAvyRF5SmkHQ829zwWWOv5p8-x?usp=sharing).

### Preprocessing

```bash
python preprocess.py <data_dir> <save_dir> <encoder_path> [--seg_len seg] [--n_workers workers]
```

- **data_dir**: The directory of speakers.
- **save_dir**: The directory to save the processed files.
- **encoder_path**: The path of pre-trained D-Vector.
- **seg**: The length of segments for training.
- **workers**: The number of workers for preprocessing.

### Training

```bash
python train.py <config_path> <data_dir> <save_dir> [--n_steps steps] [--save_steps save] [--log_steps log] [--batch_size batch] [--seg_len seg]
```

- **config_path**: The config file of model hyperparameters.
- **data_dir**: The directory of preprocessed data.
- **save_dir**: The directory to save the model.
- **steps**: The number of training steps.
- **save**: To save the model every <em>save</em> steps.
- **log**: To record training information every <em>log</em> steps.
- **batch**: The batch size.
- **seg**: The length of segments for training.

### Inference

```bash
python inference.py <model_path> <vocoder_path> <source> <target> <output>
```

- **model_path**: The path of the model file.
- **vocoder_path**: The path of the vocoder file.
- **source**: The utterance providing linguistic content.
- **target**: The utterance providing target speaker timbre.
- **output**: The converted utterance.

### Reference

Please cite the paper if you find it useful.

```bib
@InProceedings{pmlr-v97-qian19c,
  title = {{A}uto{VC}: Zero-Shot Voice Style Transfer with Only Autoencoder Loss},
  author = {Qian, Kaizhi and Zhang, Yang and Chang, Shiyu and Yang, Xuesong and Hasegawa-Johnson, Mark},
  pages = {5210--5219},
  year = {2019},
  editor = {Kamalika Chaudhuri and Ruslan Salakhutdinov},
  volume = {97},
  series = {Proceedings of Machine Learning Research},
  address = {Long Beach, California, USA},
  month = {09--15 Jun},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v97/qian19c/qian19c.pdf},
  url = {http://proceedings.mlr.press/v97/qian19c.html}
}
```
