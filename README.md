# Deep Learning Handbook

딥러닝의 이론과 PyTorch 구현을 연결하는 학습 교재입니다.

- [문서 사이트](https://minwoo02.github.io/deep-learning-handbook/)
- [학습 로드맵](https://minwoo02.github.io/deep-learning-handbook/roadmap/)

## 문서 구조

- `docs/theory/`: 기초 개념, 모델 구조, Weight, Loss, Gradient, Optimizer, Normalization, Regularization, Evaluation
- `docs/practice/`: 단계별 학습 경로와 PyTorch·scikit-learn 실습
- `docs/assets/`: 이미지, 수식 설정, 스타일
- `examples/fragments/`: 문서에 포함된 코드·출력·도표 블록
- `sources/`: 내용 대조와 유지보수를 위한 내부 자료. 사이트에는 표시하지 않습니다.
- `tools/`: 내용 보존·사이트 링크 검사

## 로컬 실행

Python 3.12 이상을 사용합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
python tools/check_sources.py
zensical build --clean
python tools/check_built_links.py
zensical serve
```

## 배포

`main` 변경 시 자동 검사·빌드 후 GitHub Pages로 배포합니다.

제3자 이미지와 인용의 권리는 원저작자에게 있습니다. 별도의 포괄적 라이선스를 부여하지 않습니다.

## 파일 이름

파일명은 소문자 영어와 하이픈으로 주제를 표현합니다. 학습 순서는 파일 번호 대신 `mkdocs.yml`과 로드맵에서 관리합니다.

| 내용 | 파일 |
| --- | --- |
| 활성화 함수 | `docs/theory/architectures/activation-functions.md` |
| 역전파와 MLP | `docs/theory/architectures/perceptron-mlp-backpropagation.md` |
| 선형회귀부터 ANN | `docs/practice/regression/linear-to-ann-regression.md` |
| Iris 분류 | `docs/practice/classification/iris-classification-pipeline.md` |
| 가중치 초기화 | `docs/practice/tech-weight/weight-initialization.md` |

각 폴더의 `index.md`는 해당 분류의 개요입니다. 관련 수집 자료와 코드 블록도 같은 주제의 파일명을 사용합니다.
