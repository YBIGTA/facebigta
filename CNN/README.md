# CNN for talent

연예인 분류를 위한 CNN 네트워크입니다. 기본적으로 입력은 3 x 128 x 128이라고 가정하며

전체적인 네트워크 구조는 VGG16의 네트워크와 동일합니다.

그러나 이때 VGG16은 224 x 224의 이미지를 처리하므로 이 부분에서 약간 차이가 존재하며

마지막 Fully-Connected layer부분도 원래 VGG에 비해 간단하게 구성하였습니다.

# Dependency

pytorch를 깔아주세요

# Dataset

데이터셋은 슬랙에 올려뒀습니다. 깃헙에 푸시할지말지는 미정입니다.

# Usage

사용법은

```
python vgg.py
```

로 그냥 사용하시면 됩니다

데이터 폴더는 vgg.py와 같은 폴더에 preprocessed라는 폴더가 있다고 가정합니다.
