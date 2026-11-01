# Deep Learning Handbook

Velog DL_a(Theory)·DL_c(Practice)를 보존하며 계층형 목차와 학습 로드맵으로 연결합니다.

- [문서 사이트](https://minwoo02.github.io/deep-learning-handbook/)
- 원문 목록·수집 근거: `sources/manifest.json`
- 구성표·제약·검증 범위: `docs/sources.md`
- 코드 조각: `examples/fragments/` (부분 코드와 출력 포함, 실행용 통합 파일 아님)

## 로컬 실행

Python 3.12 이상을 사용합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
zensical build --clean
zensical serve
```

## 배포

GitHub Settings → Pages → Source를 GitHub Actions로 설정합니다. main push 시 빌드하고 Pages artifact를 배포합니다. PR은 빌드 검증만 수행합니다.

Velog 원본은 변경하지 않았습니다. 사이트의 이관본은 렌더링된 HTML로부터 변환했습니다. 코드의 실행 결과를 새로 측정한 것으로 해석하지 마세요.

제3자 이미지와 인용의 권리는 원저작자에게 있습니다. 별도의 포괄적 라이선스를 부여하지 않습니다.
