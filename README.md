untuk amannya pake python3.8

tapi 3.10 juga bisa

install semua yang ada di requirements38.txt

dan

```python
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

install semua asset

```bash
bash inferno_apps/FaceReconstruction/download_assets.sh
```

copy ke dalam root folder

copy head_template.obj, texture_data_256.npy ke dalam assets
copy cfg.yml  ke dalam assets/FaceRecognition/models/EMICA-CVT_flame2020_notexture 

download juga BFM_to_FLAME.npz di [sini](https://huggingface.co/datasets/fachrinnk4869/deca_dataset)

kalau belum ada kamera jalankan dulu

```python
export PYTHONPATH="inferno_apps:$PYTHONPATH"
python3 inferno_apps/FaceReconstruction/demo/demo_face_rec_on_images.py
```

kalau ada kamera

```python
python3 main.py
```
documentation

![WhatsApp Image 2025-10-26 at 7 39 15 AM](https://github.com/user-attachments/assets/61ccf2aa-d2ad-4636-8234-0f2ff3e26612)
