# data/ 폴더 규약 (git 에는 올리지 않음)

음성:  data/<이름>/<화자폴더>/*.wav      예) data/vox50/id10001/xxx.wav   (화자 폴더 = 클래스)
이미지: data/<이름>/*.png|jpg            예) data/imagenet_val200/*.jpg

데이터가 아직 없으면:  python scripts/make_synthetic_data.py   (합성 데이터, 스모크 테스트 전용)
