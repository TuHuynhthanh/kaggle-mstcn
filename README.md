# MSTCN-SE Kaggle Reproduction

Tái hiện phương pháp **Multi-Scale TCN for Causal Speech Enhancement** theo Zhang & Wang, INTERSPEECH 2020.

## Thành phần

- `model.py`: MSTCN-SE, Forward Stacking, Multi-Scale Dilated ResBlock
- `dataset.py`: dữ liệu dạng clean+noise hoặc clean+noisy pair
- `train.py`: training multi-objective LPS + IRM
- `inference.py`: tăng cường tiếng nói từ một file noisy wav
- `notebook.ipynb`: notebook chạy nhanh trên Kaggle

## Chạy nhanh trên Kaggle

Upload zip này bằng nút **Upload** ở panel bên phải, rồi chạy notebook.

Nếu dùng dữ liệu clean + noise:

```bash
python train.py --mode mix \
  --clean_train /kaggle/input/.../clean_train \
  --clean_val /kaggle/input/.../clean_val \
  --noise_train /kaggle/input/.../noise_train \
  --noise_val /kaggle/input/.../noise_val \
  --epochs 20 --batch 4 --out /kaggle/working/runs/mstcn_se2
```

Nếu dùng dữ liệu clean + noisy có cặp file tương ứng:

```bash
python train.py --mode paired \
  --clean_train /kaggle/input/.../clean_train \
  --clean_val /kaggle/input/.../clean_val \
  --noisy_train /kaggle/input/.../noisy_train \
  --noisy_val /kaggle/input/.../noisy_val \
  --epochs 20 --batch 4 --out /kaggle/working/runs/mstcn_se2
```
