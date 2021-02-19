## <span style="font-variant:small-caps;">AutoVC</span>: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This is an unofficial implementation of <span style="font-variant:small-caps;">AutoVC</span> based on the official one.

The repository is still under construction, so some details may be missing or incomplete.

### Preprocessing

```bash
python preprocess.py <data_path> <save_path> <encoder_path> [--seg_len seg] [--n_workers workers]
```

### Training

```bash
python train.py <config> <data_path> <save_path> [--n_steps steps] [--save_steps save] [--log_steps log] [--batch_size batch] [--seg_len seg]
```

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
