from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import DictionarySpace, ArraySpace, IntegerSpace
from pcg_benchmark.probs.loderunner.utils import play_loderunner, read_loderunner, js_dist, State
from pcg_benchmark.probs.utils import get_num_tiles, _get_certain_tiles, get_number_regions, get_vert_histogram, get_horz_histogram, get_range_reward
import numpy as np
import os
import shutil
from PIL import Image

def load_level(filepath):
    """
    loderunnertile 레퍼런스 맵 파일을 읽어 content numpy 배열로 변환.
    각 셀이 숫자(0~6)로 저장된 텍스트 파일.
    0=brick, 1=empty, 2=player, 3=gold, 4=enemy, 5=ladder, 6=rope
    """
    return read_loderunner(filepath)

class LodeRunnerProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._gold = kwargs.get("gold")
        self._enemies = kwargs.get("enemies")

        self._target = kwargs.get("exploration",  0.2)
        self._islands = kwargs.get("islands", 0.1)
        self._decorations = kwargs.get("decorations", 0.75)
        self._used_tiles = kwargs.get("used_tiles", 0.75)
        self._diversity = kwargs.get("diversity", 0.4)
        
        self._walking = np.zeros(self._width)
        self._hanging = np.zeros(self._width)
        self._climbing = np.zeros(self._height + 1)
        self._falling = np.zeros(self._height + 1)
        lvls = [os.path.join(os.path.dirname(__file__) + "/data/", f) for f in os.listdir(os.path.dirname(__file__) + "/data/") if "level" in f]
        for lvl in lvls:
            exp = play_loderunner(read_loderunner(lvl))
            self._walking += get_horz_histogram(exp, [1])
            self._hanging += get_horz_histogram(exp, [3])
            self._climbing += get_vert_histogram(exp, [2])
            self._falling += get_vert_histogram(exp, [4])
        self._walking = np.array(self._walking) / sum(self._walking)
        self._hanging = np.array(self._hanging) / sum(self._hanging)
        self._climbing = np.array(self._climbing) / sum(self._climbing)
        self._falling = np.array(self._falling) / sum(self._falling)

        self._cerror = max(int(0.02 * self._width * self._height), 1)

        self._content_space = ArraySpace((self._height, self._width), IntegerSpace(7))
        self._control_space = DictionarySpace({
            "ladder": IntegerSpace(int(0.2 * self._width * self._height)),
            "rope": IntegerSpace(int(0.2 * self._width * self._height))
        })

    def info(self, content):
        content = np.pad(np.array(content), ((0,1), (0,0)))

        empty = get_num_tiles(content, [1])
        player = get_num_tiles(content, [2])
        gold = get_num_tiles(content, [3])
        enemy = get_num_tiles(content, [4])
        ladder = get_num_tiles(content, [5])
        rope = get_num_tiles(content, [6])
        islands = 0
        for row in content:
            islands += get_number_regions(row.reshape(1,-1), [0])
            islands += get_number_regions(row.reshape(1,-1), [6])
        for col in content.transpose():
            islands += get_number_regions(col.reshape(1,-1), [5])

        collected_gold = 0
        tiles = 0
        used_tiles = 0
        exploration = np.zeros(content.shape)
        if player == 1:
            exploration = play_loderunner(content)
            for y in range(self._height):
                for x in range(self._width):
                    if content[y][x] == 0 and content[y-1][x] != 0 and y > 0:
                        tiles += 1
                        if exploration[y-1][x] > 0:
                            used_tiles += 1
                    if content[y][x] == 5 or content[y][x] == 6:
                        tiles += 1
                        if exploration[y][x] > 0:
                            used_tiles += 1
            locs = _get_certain_tiles(content, [3])
            for x,y in locs:
                if exploration[y][x] > 0:
                    collected_gold += 1
        
        walking = get_horz_histogram(exploration, [1])
        walking = np.array(walking) / max(1, sum(walking))
        hanging = get_horz_histogram(exploration, [3])
        hanging = np.array(hanging) / max(1, sum(hanging))
        climbing = get_vert_histogram(exploration, [2])
        climbing = np.array(climbing) / max(1, sum(climbing))
        falling = get_vert_histogram(exploration, [4])
        falling = np.array(falling) / max(1, sum(falling))

        # 레퍼런스 맵이 기본 크기와 다를 경우 shape 맞추기
        wlen = min(len(walking), len(self._walking))
        clen = min(len(climbing), len(self._climbing))
        flen = min(len(falling), len(self._falling))
        hlen = min(len(hanging), len(self._hanging))

        return {
            "empty": empty,
            "player": player,
            "gold": gold,
            "enemy": enemy,
            "ladder": ladder,
            "rope": rope,
            "islands": islands,
            "exploration": exploration,
            "collected_gold": collected_gold,
            "used_tiles": used_tiles,
            "tiles": tiles,
            "walking": js_dist(walking[:wlen], self._walking[:wlen]),
            "hanging": js_dist(hanging[:hlen], self._hanging[:hlen]),
            "climbing": js_dist(climbing[:clen], self._climbing[:clen]),
            "falling": js_dist(falling[:flen], self._falling[:flen]),
            "content": content,
        }

    def quality(self, info):
        stats = get_range_reward(info["player"], 0, 1, 1, self._width * self._height)
        stats += get_range_reward(info["gold"], 0, self._gold, 2 * self._gold, self._width * self._height)
        stats += get_range_reward(info["enemy"], 0, self._enemies, 2 * self._enemies, self._width * self._height)
        stats /= 3.0

        exploration = 0
        if stats >= 1:
            exploration += get_range_reward(((info["exploration"] > 0).astype(int)).sum(), 0,\
                int(self._target * self._width * self._height), self._width * self._height)
        
        play_stats = 0
        if exploration >= 1:
            play_stats += info["collected_gold"] / info["gold"]
            if info["tiles"] > 0:
                play_stats += get_range_reward(info["used_tiles"] / info["tiles"], 0, self._used_tiles, 1.0)
            play_stats /= 2.0

        decoration = 0
        if play_stats >= 1:
            decoration = 0.9 * (info["walking"] + info["hanging"] + info["climbing"]) + 0.1 * info["falling"]
            decoration = get_range_reward(decoration, 0, self._decorations, 1)
            decoration += get_range_reward(info["islands"], 0, 0, self._islands * self._width * self._height, self._width * self._height / 2)
            decoration /= 2

        return (stats + exploration + play_stats + decoration) / 4
    
    def diversity(self, info1, info2):
        walking = abs((info1["exploration"] == 1).astype(int) - (info2["exploration"] == 1).astype(int)).sum()
        stairs = abs((info1["exploration"] == 2).astype(int) - (info2["exploration"] == 2).astype(int)).sum()
        ropes = abs((info1["exploration"] == 3).astype(int) - (info2["exploration"] == 3).astype(int)).sum()
        falling = abs((info1["exploration"] == 4).astype(int) - (info2["exploration"] == 4).astype(int)).sum()
        return get_range_reward(0.3 * (walking + stairs + ropes) + 0.1 * falling, 0,\
            self._diversity * self._width * self._height, self._width * self._height)
    
    def controlability(self, info, control):
        ladder = get_range_reward(info["ladder"], 0, control["ladder"] - self._cerror, control["ladder"] + self._cerror, 
                                  self._width * self._height)
        rope = get_range_reward(info["rope"], 0, control["rope"] - self._cerror, control["rope"] + self._cerror, 
                                  self._width * self._height)
        return (ladder + rope) / 2
    
    def render(self, content):
        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/brick.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/gold.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/enemy.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/ladder.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/rope.png").convert('RGBA'),
        ]
        lvl = np.pad(np.array(content), ((0,1), (0,0)))
        lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
        for y in range(lvl.shape[0]):
            for x in range(lvl.shape[1]):
                lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
        return lvl_image

    def render_solution(self, info, map_name="map"):
        """Greedy 최근접 미방문 지점 이동 방식으로 플레이어 경로를 생성해 렌더링"""
        content = np.array(info["content"])
        exploration = info["exploration"]
        is_clear = info["collected_gold"] > 0 or (info["player"] == 1 and (exploration > 0).sum() > 0)

        starts = _get_certain_tiles(content, [2])
        if len(starts) == 0:
            return [], is_clear

        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/brick.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/gold.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/enemy.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/ladder.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/rope.png").convert('RGBA'),
        ]

        sx, sy = starts[0]
        sim = content.copy()
        sim[sim == 2] = 1
        sim[sim == 3] = 1
        sim[sim == 4] = 1

        # 사다리 내려가기를 지원하는 확장 State
        # 원본 State는 현재 타일이 사다리(5)일 때만 위아래 이동 가능.
        # 여기서는 추가로: 현재 타일이 빈칸(1)이고 바로 아래가 사다리(5)면 내려갈 수 있게 확장.
        class ExtState(State):
            def update(self, dx, dy):
                if self._falling:
                    self._y += 1
                else:
                    if abs(dy) >= 1:
                        ny = self._y + dy
                        if ny < 0:
                            ny = 0
                        if ny > self._level.shape[0] - 1:
                            ny = self._level.shape[0] - 1
                        if self._level[ny][self._x] != 0:
                            if self._level[self._y][self._x] == 5:
                                # 사다리 위에서는 위아래 모두 이동
                                self._y = ny
                            elif self._level[self._y][self._x] == 6 and dy > 0:
                                # 로프에서는 아래로만
                                self._y = ny
                            elif (dy > 0 and
                                  self._level[self._y][self._x] == 1 and
                                  ny < self._level.shape[0] and
                                  self._level[ny][self._x] == 5):
                                # 빈칸에서 아래가 사다리면 내려가기 가능
                                self._y = ny
                    if abs(dx) >= 1:
                        nx = self._x + dx
                        if nx < 0:
                            nx = 0
                        if nx > self._level.shape[1] - 1:
                            nx = self._level.shape[1] - 1
                        if self._level[self._y][nx] != 0:
                            self._x = nx
                self._falling = 0
                if (self._y < self._level.shape[0] - 1 and
                        self._level[self._y][self._x] == 1 and
                        self._level[self._y+1][self._x] not in [0, 5]):
                    self._falling = 1

            def clone(self):
                s = ExtState(self._level, self._x, self._y, self._falling)
                return s

        # ExtState BFS로 도달 가능한 모든 타일 계산
        def _bfs_reachable(cx, cy):
            init_falling = 0
            if cy < sim.shape[0] - 1 and sim[cy][cx] == 1 and sim[cy+1][cx] not in [0, 5]:
                init_falling = 1
            visited_bfs = set()
            reached = set()
            queue = [ExtState(sim, cx, cy, init_falling)]
            while queue:
                cur = queue.pop(0)
                key = str(cur)
                if key in visited_bfs:
                    continue
                visited_bfs.add(key)
                reached.add((cur._x, cur._y))
                for a in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = cur.clone()
                    nxt.update(a[0], a[1])
                    if str(nxt) not in visited_bfs:
                        queue.append(nxt)
            return reached

        ext_reachable = _bfs_reachable(sx, sy)

        # 도달 가능한 gold(3), enemy(4) 타일만 목표로 설정
        reachable = set(
            (x, y)
            for y in range(content.shape[0])
            for x in range(content.shape[1])
            if content[y][x] in (3, ) and (x, y) in ext_reachable
        )

        def _bfs_nearest(cx, cy, unvisited):
            """현재 위치에서 BFS로 가장 가까운 미방문 오브젝트를 찾으면 즉시 중단."""
            init_falling = 0
            if cy < sim.shape[0] - 1 and sim[cy][cx] == 1 and sim[cy+1][cx] not in [0, 5]:
                init_falling = 1
            prev = {}
            visited_bfs = set()
            queue = [ExtState(sim, cx, cy, init_falling)]
            while queue:
                cur = queue.pop(0)
                key = str(cur)
                if key in visited_bfs:
                    continue
                visited_bfs.add(key)
                pos = (cur._x, cur._y)
                if pos in unvisited:
                    path = []
                    c = pos
                    while c in prev:
                        path.append(c)
                        c = prev[c]
                    path.append((cx, cy))
                    path.reverse()
                    return path
                for a in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = cur.clone()
                    nxt.update(a[0], a[1])
                    nkey = str(nxt)
                    if nkey not in visited_bfs:
                        npos = (nxt._x, nxt._y)
                        if npos not in prev:
                            prev[npos] = (cur._x, cur._y)
                        queue.append(nxt)
            return None

        # Greedy: 현재 위치에서 가장 가까운 미방문 타일로 반복 이동
        base = content.copy()
        base[base == 2] = 1

        def _render_frame(px, py):
            state = base.copy()
            state[py][px] = 2
            img = Image.new("RGBA", (state.shape[1]*scale, state.shape[0]*scale), (0,0,0,255))
            for y in range(state.shape[0]):
                for x in range(state.shape[1]):
                    img.paste(graphics[state[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
            return img

        full_path = [(sx, sy)]
        visited_objects = set()
        cx, cy = sx, sy

        while True:
            unvisited = reachable - visited_objects
            if not unvisited:
                break
            seg = _bfs_nearest(cx, cy, unvisited)
            if seg is None:
                break
            # 경로 전체를 full_path에 추가 (중복 첫 위치 제외)
            for pos in seg[1:]:
                full_path.append(pos)
            # 목표 오브젝트(마지막 지점)만 visited 처리
            cx, cy = seg[-1]
            visited_objects.add((cx, cy))

        frames = [_render_frame(px, py) for px, py in full_path]

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "loderunnertile", map_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(save_dir, f"frame_{i:04d}.png"))

        # --- _overview.png: 도달 가능한 타일에 초록 반투명 오버레이 ---
        overview = _render_frame(sx, sy)
        overlay = Image.new("RGBA", overview.size, (0, 0, 0, 0))
        from PIL import ImageDraw as _IDraw
        draw = _IDraw.Draw(overlay)
        for (ox, oy) in ext_reachable:
            draw.rectangle(
                [ox * scale, oy * scale, (ox + 1) * scale - 1, (oy + 1) * scale - 1],
                fill=(0, 200, 80, 100)
            )
        overview = Image.alpha_composite(overview, overlay)
        overview.save(os.path.join(save_dir, "_overview.png"))

        return frames, is_clear

    def score_frames(self, frames, is_clear):
        """이미지 프레임 리스트를 받아 LLM 점수를 반환 (현재는 -1 고정)"""
        return -1

    def llmscore(self, info, map_name="map"):
        frames, is_clear = self.render_solution(info, map_name=map_name)
        score = self.score_frames(frames, is_clear)
        return score, is_clear

