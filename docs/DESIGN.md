# Dataset Doctor — 상세 설계

## 1. 해결하는 문제

YOLO 계열 라벨은 텍스트 파일이라 깨져 있어도 학습이 그냥 돌아가는 경우가 많다.

- 좌표가 1.0을 넘으면 프레임워크가 조용히 clip하거나 라벨을 버린다.
- OBB 꼭짓점 순서가 꼬여 박스가 나비 모양(자기교차)이 돼도 에러가 나지 않는다.
- `data.yaml`의 클래스는 5개인데 라벨에 `class 7`이 섞여 있다.
- 같은 이미지가 train과 val에 동시에 있어 val 점수가 부풀려진다.

이런 문제는 대개 학습이 끝난 뒤 성능이 이상할 때 발견된다. Dataset Doctor는 이를 학습 전에 잡는다.

## 2. MVP 지원 포맷

| 포맷 | 라벨 한 줄 형식 | 비고 |
|---|---|---|
| `yolo-hbb` | `cls cx cy w h` | 0~1 정규화 |
| `yolo-obb` | `cls x1 y1 x2 y2 x3 y3 x4 y4` | 꼭짓점 4개, 정규화 |
| `yolo-seg` | `cls x1 y1 x2 y2 ... xn yn` | polygon, 정규화 |

COCO JSON, LabelMe, CVAT는 v0.2 이후에 추가한다.

## 3. 검사 항목

각 결과에는 severity(ERROR / WARNING / INFO)를 붙인다.

### ① 구조 검사 (파일 단위)
- `image_without_label` (INFO, background일 수 있음)
- `label_without_image` (ERROR)
- `corrupt_image`, `empty_file`
- `duplicate_image`: 파일 해시 기준
- `split_leak`: 같은 해시가 train/val/test 중 여러 split에 존재

### ② 문법 검사 (줄 단위)
- `wrong_column_count`: OBB인데 값 7개, seg인데 좌표 수 홀수 등
- `parse_error`: 숫자 파싱 불가, NaN, inf
- `class_id_out_of_range`: `data.yaml` 범위 밖, 또는 정수가 아님

### ③ 기하 검사 (핵심)
- `coord_out_of_range`: 살짝 넘으면 WARNING, 크게 넘으면 ERROR
- `non_positive_size`: HBB w 또는 h ≤ 0
- `tiny_box`: 예) 2px 미만
- `self_intersecting_obb`: 꼭짓점 순서 꼬임
- `non_rectangular_obb`: 네 각이 90°에서 크게 벗어남 (기본 허용 ±10°)
- `degenerate_obb`: 면적이 거의 0
- `inconsistent_winding`: 데이터셋 내 CW/CCW 혼재 (WARNING)
- `invalid_polygon`: 점 3개 미만, 면적 0, 자기교차 (Shapely `is_valid`)

**주의:** 기하 검사는 반드시 이미지 크기로 역정규화한 픽셀 좌표에서 수행한다. 1920×1080 이미지에서 정규화 좌표로 각도를 재면 실제 직사각형도 평행사변형으로 보인다.

```python
import numpy as np

def check_obb(pts_norm: np.ndarray, img_w: int, img_h: int) -> list[str]:
    """pts_norm: (4, 2) 정규화 좌표"""
    issues = []
    pts = pts_norm * np.array([img_w, img_h])  # 픽셀 좌표로 변환

    # 1) 부호 있는 면적 (shoelace): 0에 가까우면 퇴화, 부호는 회전 방향
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area) < 1.0:
        issues.append("degenerate_obb")

    # 2) 자기교차: 마주보는 변(0-1 vs 2-3, 1-2 vs 3-0)이 교차하는지
    if segments_intersect(pts[0], pts[1], pts[2], pts[3]) or \
       segments_intersect(pts[1], pts[2], pts[3], pts[0]):
        issues.append("self_intersecting_obb")

    # 3) 직사각형 여부
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if abs(cos) > np.cos(np.deg2rad(80)):
            issues.append("non_rectangular_obb")
            break
    return issues
```

### ④ 일관성 및 통계
- `duplicate_label`: 같은 클래스 IoU ≥ 0.95
- `label_conflict`: 같은 위치에 다른 클래스
- 클래스 분포와 불균형 비율, 박스 크기 분포, 이미지당 객체 수, 해상도·종횡비 분포

## 4. CLI

```bash
dataset-doctor scan ./dataset --format yolo-obb
dataset-doctor scan ./dataset --data data.yaml --report html --out report/
dataset-doctor scan ./dataset --fail-on error          # CI용, error 있으면 exit 1
dataset-doctor fix  ./dataset --out ./dataset_fixed --clip-coords --reorder-obb --dedup
```

터미널 출력 예시:

```
Dataset Doctor v0.1.0 — yolo-obb — 4,812 images / 31,077 labels

 ERROR    label_without_image        12
 ERROR    class_id_out_of_range       3   (max=4, found 7)
 ERROR    self_intersecting_obb      41
 WARNING  coord_out_of_range        118   (max 1.032)
 WARNING  duplicate_label            27
 WARNING  split_leak                  9   (train ↔ val)
 INFO     image_without_label        64

→ report/index.html
```

HTML 리포트: 문제 유형별 필터, 이미지 위에 문제 라벨을 빨간색으로 오버레이. README 데모 GIF도 이 화면에서 만든다.

`fix`: 원본은 건드리지 않고 새 폴더에 저장하며 `fix_log.json`을 남긴다. 클래스 충돌처럼 판단이 필요한 것은 리포트만 한다.

## 5. 디렉터리 구조

```
dataset-doctor/
├── src/dataset_doctor/
│   ├── loaders/          # yolo_hbb.py, yolo_obb.py, yolo_seg.py
│   ├── core/
│   │   ├── models.py     # Sample, Annotation, Issue
│   │   └── registry.py   # 검사 플러그인 등록
│   ├── checks/           # structure.py, syntax.py, geometry.py, stats.py
│   ├── fix/
│   ├── report/           # terminal.py, json.py, html/ (jinja2)
│   └── cli.py
├── tests/
│   └── fixtures/         # 일부러 깨뜨린 미니 데이터셋
├── docs/
├── .github/workflows/
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

```python
@dataclass
class Issue:
    code: str            # "self_intersecting_obb"
    severity: Severity   # ERROR / WARNING / INFO
    image: Path
    label_file: Path | None
    line: int | None     # 라벨 파일 줄 번호
    detail: dict

@register_check(formats={"yolo-obb"}, stage="geometry")
def obb_geometry(sample: Sample) -> Iterable[Issue]:
    ...
```

대용량 데이터셋에 대비해 `concurrent.futures`로 병렬 처리한다.

## 6. 테스트 전략

일부러 깨뜨린 데이터셋 생성기(`make_dataset`)를 만들고, 검사 항목마다 테스트를 붙인다.

```python
def test_detects_butterfly_obb(tmp_path):
    ds = make_dataset(tmp_path, fmt="yolo-obb")
    ds.add_label("img_001", "0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5")  # 순서 꼬인 박스
    issues = scan(ds.root)
    assert any(i.code == "self_intersecting_obb" and i.line == 1 for i in issues)
```

공개 예제: Ultralytics `coco8`, `coco8-seg`, `dota8` 미니 샘플. 원본 DOTA는 학술용 라이선스이므로 레포에 넣지 않고 다운로드 스크립트로 제공한다.


## 7. 확장 계획
- pre-commit hook과 GitHub Action 제공 (데이터셋 레포 PR 자동 검사)
- COCO JSON, LabelMe, CVAT 포맷
- 근접 중복 이미지 검출 (perceptual hash)
- 모델 예측과 라벨을 비교해 오라벨 가능성이 높은 샘플 찾기
