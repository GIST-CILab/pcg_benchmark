# Video Rendering 알고리즘

각 게임에 `render_solution`, `score_frames`, `llmscore` 메서드가 추가되었다.  
레퍼런스 맵은 `reference/{game}/` 에 저장되어 있으며, 생성된 프레임은 `videos/{game}/{map_name}/` 에 저장된다.  
testvideo.py에서 실행 가능 

---

## 공통 구조

```python
# 1. info 계산 (게임 플레이 시뮬레이션 포함)
info = env._problem.info(content)

# 2. 영상 프레임 생성 + 저장
frames, is_clear = env._problem.render_solution(info, map_name="my_map")

# 3. LLM 점수 계산 (현재는 -1 고정, 추후 LLM 연동 예정)
score = env._problem.score_frames(frames, is_clear)

# 2+3 한번에
score, is_clear = env._problem.llmscore(info, map_name="my_map")
```

`env.llmscore(info_list, map_names=name_list)` 로 배치 처리도 가능하다.

---

## 게임별 알고리즘

### Sokoban (`sokoban-v0`)
- **알고리즘**: BFS + A* 기반 solver(`SokobanSolver`)로 솔루션 액션 리스트를 구하고, 매 액션마다 박스/플레이어 위치를 업데이트하며 프레임 생성
- **결과**: 박스를 목표 위치에 밀어 넣는 풀이 과정 영상
- **저장 위치**: `videos/sokoban/{map_name}/`

### SMB Tile (`smbtile-v0`)
- **알고리즘**: 내장 시뮬레이터로 마리오의 이동 궤적(`locations`)을 구하고, 매 위치마다 마리오 스프라이트를 동적으로 배치해 프레임 생성. 2-step 시뮬레이션(스텝 폭 2배, 탐색 깊이 8)으로 실행 속도 최적화
- **결과**: 마리오가 맵을 좌→우로 이동하는 플레이 영상
- **저장 위치**: `videos/smbtile/{map_name}/`

### Zelda (`zelda-v0`)
- **알고리즘**: BFS로 플레이어 → 열쇠 경로, 열쇠 획득 후 → 문 경로를 각각 계산. 두 구간을 이어 붙여 전체 풀이 프레임 생성. 열쇠 획득 시점에 열쇠 타일이 맵에서 사라짐
- **결과**: 열쇠를 획득하고 문에 도달하는 2단계 탐험 영상
- **저장 위치**: `videos/zelda/{map_name}/`

### Talakat (`talakat-v0`)
- **알고리즘**: JSON 스크립트를 `runPattern()`으로 시뮬레이션하여 전체 탄막 궤적을 계산. 매 프레임마다 탄막(원)·보스·플레이어를 PIL로 직접 그림. `_renderSampling` 간격으로 샘플링해 프레임 수 조절
- **결과**: 보스가 탄막을 발사하고 플레이어가 생존하는 슈팅 게임 영상
- **저장 위치**: `videos/talakat/{map_name}/`

### Lode Runner Tile (`loderunnertile-v0`)
- **알고리즘**: 사다리 내려가기를 지원하는 확장 State(`ExtState`)로 BFS 탐색. **Greedy 최근접 오브젝트 순회** 방식으로 플레이어 경로 생성
  1. `ExtState` BFS로 시작점에서 도달 가능한 모든 타일 계산 (`ext_reachable`)
  2. 도달 가능한 **Gold(동전)·Enemy(적) 타일만** 목표로 설정
  3. 현재 위치에서 BFS로 가장 가까운 미방문 오브젝트를 찾으면 **즉시 중단** 후 이동 (조기 종료 최적화)
  4. 모든 오브젝트 방문 또는 도달 불가 시 종료
- **추가 이미지**: `_overview.png` — 도달 가능한 타일에 초록 반투명 오버레이를 표시한 정적 이미지
- **결과**: 플레이어가 동전과 적을 효율적으로 수집하는 탐험 영상 + overview 이미지
- **저장 위치**: `videos/loderunnertile/{map_name}/`

### MiniDungeons (`mdungeons-v0`)
- **알고리즘**: **Treasure Collector** 페르소나 기반 A* 탐색
  - 일반 풀이(`_run_game`): 기존 A* + BFS cascade (출구 도달 최단 경로)
  - TC 풀이(`_run_game_treasure_collector`): `TreasureCollectorState`로 A* 실행
    - 보물/적이 남아있으면 **출구 진입 자체를 차단** → 모두 수집 후에만 출구 도달 가능
    - 휴리스틱: `남은 보물 × (W+H) × 3 + 남은 적 × (W+H) × 2 + 출구까지 거리`
    - `getKey()`에 보물·적 잔존 상태를 포함 → 수집 상태가 다르면 같은 위치도 재탐색
  - 탐색 한계: 일반 200,000회 / TC 500,000회
- **UI**: 각 프레임 하단에 HP 바 + 실시간 점수(동전 수집 100점, 적 처치 50점) 표시
  - `render_solution(info, show_ui=True, treasure_score=100, enemy_score=50)` 으로 조절 가능
- **결과**: 보물과 적을 최대한 수집하며 출구에 도달하는 던전 탐험 영상
- **저장 위치**: `videos/mdungeons/{map_name}/`

---

## 레퍼런스 맵

| 게임 | 경로 | 포맷 |
|------|------|------|
| Sokoban | `reference/sokoban/` | 텍스트 (`#`, `@`, `$`, `.` 등) |
| SMB Tile | `reference/smbtile/` | 텍스트 (SMB 타일 문자) |
| Zelda | `reference/zelda/` | 텍스트 (`w`, `.`, `g`, `+`, `A`, `1~3` 등) |
| Talakat | `reference/talakat/` | JSON 스크립트 |
| Lode Runner Tile | `reference/loderunnertile/` | 텍스트 (숫자 포맷) |
| MiniDungeons | `reference/mdungeons/` | 텍스트 (숫자 포맷, `#`→0, `.`→1, `S`→2, `E`→3, `P`→4, `T`→5, `M`→6) |

