# 구축·검증 결과 · 2026-11-01

- 공개 사이트: https://minwoo02.github.io/deep-learning-handbook/
- 로드맵: https://minwoo02.github.io/deep-learning-handbook/roadmap/
- 배포 실행: https://github.com/minwoo02/deep-learning-handbook/actions/runs/41293988846
- 배포한 소스 커밋: `067669b`

## 변경 사항

DL_a 24편과 DL_c 15편(목차 2편 포함)을 수집했습니다. 렌더링된 본문 HTML·텍스트와 출처·수집 시각을 보존하고 77개 Markdown 페이지로 구성했습니다. Theory와 Practice, Task와 기술을 구분한 계층형 목차, 단계별 실습 경로, 실제 코드 기준 구성표, 이론·실습 상호 링크, Mermaid 마인드맵을 추가했습니다. 기존 scikit-learn 글도 별도로 보존했습니다.

Python 3.12와 Zensical 0.0.74로 로컬·GitHub Actions 빌드를 확인하고 의존 버전을 고정했습니다. 이후 main 변경 시 자동 검사·배포됩니다. Velog에는 쓰기 작업을 하지 않았습니다.

## 검증

| 항목 | 결과 |
| --- | --- |
| 원문 수집 | 39/39편 |
| 코드·텍스트 도표 블록 보존 | 1,364개, 원문과 이관본 내용 일치 |
| 이미지 보존 | 15/15개, 로컬 파일·원본 URL·SHA-256 보존 |
| 빌드한 내부 링크·자산 참조 | 7,456개, 누락 없음 |
| Zensical clean build | 로컬·GitHub Actions 성공 |
| 수식 | 로컬 Loss 페이지에서 MathJax 254개 렌더링 확인 |
| 공개 배포 | GitHub Actions deploy 성공, 공개 Home·로드맵 확인 |

## 원문에서 확인된 제한

- XOR 글은 2→4→1 모델 클래스만 포함합니다. Loss·Optimizer·학습 결과를 추가로 만들어 넣지 않았습니다.
- CNN·RNN 독립 실습은 두 시리즈에서 확인되지 않아 예정으로 표시했습니다.
- 회귀 종합 글에 이미 수동 gradient→autograd→SGD→ANN 단계가 있습니다. 이 내용을 보존하여 단계별 안내에서 연결했습니다.
- Perceptron 예제는 Iris 다중 분류입니다. 이전 대화의 binary 예시로 덮어쓰지 않았습니다.
- 코드 조각은 부분 코드·출력·도표를 포함합니다. 전체 학습 실행과 성능 재현을 완료했다는 뜻은 아닙니다.
- 변환의 입력은 Velog의 렌더링 HTML이며 작성기 원본 Markdown과 동일하지 않습니다. 외부 링크 전체의 가용성은 별도 검증 대상입니다.
