# pre-trained modelyou can find `/home2/pytorch-broad-models/MGANet-DCC2020/model/MGANet_model_AI37.pth`, easy to use local dataset by symlink like:

```
cd PT/MGANet
ln -s /home2/pytorch-broad-models/MGANet-DCC2020/model/MGANet-DCC2020/model/MGANet_model_AI37.pth models/MGANet_model_AI37.pth
```
# Run

```bash
cd PT/MGANet/codes
python MGANet_test_AI37.py  --net_G ../models/MGANet_model_AI37.pth --ipex
```
