# Deep Learning Handbook

딥러닝의 개념과 수학적 원리를 이해하고, PyTorch로 직접 모델을 구현하며 학습하는 Handbook입니다.

[문서 사이트](https://minwoo02.github.io/deep-learning-handbook/) · [학습 로드맵](https://minwoo02.github.io/deep-learning-handbook/roadmap/) · [배포 상태](https://github.com/minwoo02/deep-learning-handbook/actions)

## 학습 구성

| 분류 | 내용 |
| --- | --- |
| **Theory** | Foundations & Tasks, Model Architecture, Weight & Initialization, Loss, Gradient / Autograd, Optimizer, Normalization, Regularization, Evaluation |
| **Practice** | PyTorch 기초, 회귀·분류 구현, 기술별 심화 실습, scikit-learn 입문 |

Regression·Classification은 **해결할 문제(Task)**, MLP·CNN·RNN은 **모델 구조(Architecture)** 로 구분합니다. 관련 이론과 실습은 문서 사이의 링크로 연결합니다.

### 권장 학습 순서

PyTorch Basics → Regression → Classification → CNN → RNN

- **Regression**: 수동 gradient → Autograd → SGD → nn.Module → Dataset/DataLoader → ANN 회귀
- **Classification**: XOR 모델 구조 → Iris 다중 분류
- **CNN·RNN**: 구현 실습 준비 중

실습 문서 앞의 구성표에서 Task, Model, nn.Module, Autograd, Loss, Optimizer, Activation을 확인할 수 있습니다.

## 폴더와 주요 파일

| 경로 | 역할 |
| --- | --- |
| [`docs/theory/`](docs/theory/) | 개념별 이론 문서 |
| [`docs/practice/`](docs/practice/) | 학습 단계별 안내와 구현 실습 |
| [`docs/roadmap.md`](docs/roadmap.md) | 전체 학습 순서와 마인드맵 |
| [`docs/assets/images/`](docs/assets/images/) | 문서에 사용하는 이미지 |
| [`mkdocs.yml`](mkdocs.yml) | 메뉴, 문서 순서, 테마, 목차, 이전 주소 연결 설정 |
| [`examples/fragments/`](examples/fragments/) | 문서의 코드·출력·텍스트 도표 블록. 일부는 부분 코드입니다. |
| [`sources/`](sources/) | 내용 대조와 유지보수를 위한 자료 |
| [`tools/`](tools/) | 내용·링크·문서 구조 검사 도구 |
| [`.github/workflows/docs.yml`](.github/workflows/docs.yml) | 자동 검사·빌드·배포 설정 |

사이트에는 `docs/`의 문서가 표시됩니다. 각 분류의 `index.md`는 해당 분류의 개요입니다.

## 글 찾기

파일명은 소문자 영어와 하이픈으로 주제를 나타냅니다. 학습 순서는 파일 번호 대신 메뉴와 로드맵에서 관리합니다.

| 주제 | 문서 |
| --- | --- |
| 활성화 함수 | [activation-functions.md](docs/theory/architectures/activation-functions.md) |
| Perceptron·MLP·역전파 | [perceptron-mlp-backpropagation.md](docs/theory/architectures/perceptron-mlp-backpropagation.md) |
| 선형회귀부터 ANN | [linear-to-ann-regression.md](docs/practice/regression/linear-to-ann-regression.md) |
| Iris 분류 | [iris-classification-pipeline.md](docs/practice/classification/iris-classification-pipeline.md) |
| 가중치 초기화 | [weight-initialization.md](docs/practice/tech-weight/weight-initialization.md) |

## 글 수정·추가

### 기존 글 수정

1. `docs/`에서 수정할 문서를 엽니다.
2. GitHub의 **Edit(연필)** 버튼으로 내용을 수정합니다.
3. **Commit changes**로 `main`에 저장합니다.

### 새 글 추가

1. 알맞은 분류 폴더에서 **Add file → Create new file**을 선택합니다.
2. `cnn-basics.md`처럼 내용을 나타내는 이름으로 작성합니다.
3. `mkdocs.yml`의 `nav`에 파일 경로를 추가합니다. 경로는 `docs/` 기준이며, 기존 항목의 들여쓰기를 따릅니다.
4. 필요한 경우 개요 페이지와 로드맵에도 연결합니다.

문서 제목은 `#` 하나, 본문 제목은 `##` → `###` 순서로 작성합니다. 제목 아래에 `[TOC]`를 넣으면 본문 목차가 표시됩니다.

이미지는 `docs/assets/images/`에 추가하고 문서에서 상대 경로로 연결합니다. 파일을 이동하거나 이름을 바꿀 때는 메뉴·관련 링크를 수정하고, `mkdocs.yml`의 `redirect_maps`에 이전 경로를 연결합니다.

## 로컬 미리보기

Python 3.12 이상을 사용합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
zensical serve
```

브라우저에서 `http://127.0.0.1:8000`을 열면 수정 내용을 미리 볼 수 있습니다.

### 검사와 빌드

```sh
python tools/check_sources.py
zensical build --clean
python tools/check_built_links.py
python tools/check_site_structure.py
```

문서의 코드를 수정했다면 관련 내용 대조 기준도 함께 점검합니다.

## 자동 배포

`main`에 변경 사항을 저장하면 GitHub Actions가 **검사 → Zensical 빌드 → GitHub Pages 배포**를 실행합니다. Pull Request에서는 검사와 빌드만 실행합니다.

진행 상태와 오류는 [Actions](https://github.com/minwoo02/deep-learning-handbook/actions)에서 확인할 수 있습니다. 빌드 결과물인 `site/`는 자동으로 생성되므로 직접 편집하지 않습니다.

## 저작권

제3자 이미지와 인용의 권리는 원저작자에게 있습니다. 이 저장소 전체에 별도의 포괄적 라이선스를 부여하지 않습니다.
