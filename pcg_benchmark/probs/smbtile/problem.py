from pcg_benchmark.probs import Problem
from pcg_benchmark.probs.utils import get_range_reward, get_num_tiles
from pcg_benchmark.probs.smb.engine import runLevel
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
import numpy as np
import math
from PIL import Image
import os

def load_level(filepath):
    """
    레벨 파일(smbtile 포맷)을 읽어 content numpy 배열(height x width)로 변환.
    symbols = ['-', 'X', '#', 'S', 'Q', 't', 'o', 'g', 'k', 'y']
    알 수 없는 문자: E->g(goomba), ?->Q, B->S(brick), <>[]/b->t(pipe), b->'-'
    """
    symbols = ['-', 'X', '#', 'S', 'Q', 't', 'o', 'g', 'k', 'y']
    sym_map = {s: i for i, s in enumerate(symbols)}
    extra_map = {'E': sym_map['g'], '?': sym_map['Q'], 'B': sym_map['S'],
                 '<': sym_map['t'], '>': sym_map['t'], '[': sym_map['t'], ']': sym_map['t'],
                 'b': sym_map['-']}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if len(l) > 0]
    height = len(lines)
    width = max(len(l) for l in lines)
    arr = np.zeros((height, width), dtype=int)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch in sym_map:
                arr[y][x] = sym_map[ch]
            elif ch in extra_map:
                arr[y][x] = extra_map[ch]
            else:
                arr[y][x] = 0  # '-' (empty)
    return arr

def _convert2str(content, symbols):
    content = np.pad(content, [(0, 0), (3,3)])
    width = len(content[0])
    height = len(content)
    result = ""
    for y in range(height):
        for x in range(width):
            if (x < 3 or x >= width - 3) and y > height - 3:
                result += "X"
            elif x == 1 and y == height - 3:
                result += "M"
            elif x == width - 2 and y == height - 3:
                result += "F"
            else:
                result += symbols[content[y][x]]
        result += '\n'
    return result

def _calculate_hnoise(content, slices):
    lvl = _convert2str(content, slices).split('\n')
    values = []
    for l in lvl:
        l = l.strip()
        if len(l) == 0:
            continue
        temp = 0
        for x in range(1,len(l)):
            if l[x] != l[x-1]:
                temp += 1
        temp /= (len(l) - 1)
        values.append(temp)
    return values

def _caculate_fenemies(content, slices):
    lvl = _convert2str(content, slices).split('\n')
    enemies = set('ykgr')
    solid = set('X#tSQ')
    total = 0
    floating = 0
    for x in range(len(lvl[0].strip())):
        for y in range(len(lvl)):
            lvl[y] = lvl[y].strip()
            if len(lvl[y]) == 0:
                continue
            if lvl[y][x] in enemies:
                total += 1
                if y+1 == len(lvl)-1 or lvl[y+1][x] not in solid:
                    floating += 1
    return floating/total

def _convert_action(action):
    result = 0
    for i in range(len(action)):
        result += int(action[i]) * pow(2, i)
    return result

class MarioProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        self._hnoise = []
        with open(os.path.dirname(__file__) + "/noise.txt") as f:
            self._hnoise = np.array([float(v) for v in f.readlines()[0].strip().split(',')])

        self._symbols = ['-', 'X', '#', 'S', 'Q', 't', 'o', 'g', 'k', 'y']
        
        self._width = kwargs.get("width")
        self._height = kwargs.get("height", 16)
        self._empty = kwargs.get("empty", 0.5)
        self._fenemies = kwargs.get("fenemies", 0.1)
        self._solver = kwargs.get("solver", 100 / (int(self._width < 30) + 1))
        self._solver_iterations = kwargs.get("solver_iterations", 200)
        self._sticky_actions = kwargs.get("sticky_actions", 8)
        self._timer = kwargs.get("timer", math.ceil(self._width / 5))
        self._diversity = kwargs.get("diversity", 0.4)

        self._content_space = ArraySpace((self._height, self._width), IntegerSpace(10))
        self._control_space = DictionarySpace({"enemies": IntegerSpace(0, max(2, int(self._width / 7))), 
                                               "jumps": IntegerSpace(0, max(2, int(self._width / 5))), 
                                               "coins": IntegerSpace(0, max(2, int(self._width / 10)))})
        self._cerror = {"enemies": int(0.1 * self._control_space._value["enemies"]._max_value),
                        "jumps": int(0.1 * self._control_space._value["jumps"]._max_value),
                        "coins": int(0.1 * self._control_space._value["coins"]._max_value)}

    def parameters(self, **kwargs):
        Problem.parameters(self, **kwargs)

        self._diversity = kwargs.get("diversity", 0.4)
        self._solver = kwargs.get("solver", self._solver)
        self._timer = kwargs.get("timer", self._timer)

    def info(self, content):
        lvl = _convert2str(content, self._symbols)
        lvl_lines = lvl.split("\n")
        tube_issue = 0
        for l in lvl_lines:
            test_tube = 0
            for c in l:
                if c == 't':
                    test_tube += 1
                else:
                    if test_tube % 2 > 0:
                        tube_issue += 1
                    test_tube = 0
        hnoise_raw = np.array(_calculate_hnoise(content, self._symbols))
        hnoise_ref = self._hnoise[:len(hnoise_raw)]
        hnoise = hnoise_raw - hnoise_ref
        hnoise[hnoise < 0] = 0
        empty = get_num_tiles(np.array(content), [0]) / (len(content[0]) * len(content))
        fenemies = _caculate_fenemies(content, self._symbols)

        def _run(agent_name, iterations, sticky):
            return runLevel(lvl, agent_name, self._timer, iterations, sticky)

        def _extract(result):
            acts, locs = [], []
            for ae in result.getAgentEvents():
                acts.append(_convert_action(ae.getActions()))
                locs.append([ae.getMarioX(), ae.getMarioY()])
            return acts, locs, result

        # heuristic 시도
        best_result = _run("heuristic", self._solver_iterations, self._sticky_actions)
        if best_result.getCompletionPercentage() < 1.0:
            # astar로 2x iterations, sticky=8 한 번 재시도
            result2 = _run("astar", self._solver_iterations * 2, self._sticky_actions)
            if result2.getCompletionPercentage() >= best_result.getCompletionPercentage():
                best_result = result2

        actions, locations, _ = _extract(best_result)

        return {
            "width": len(content[0]),
            "height": len(content),
            "tube": tube_issue,
            "empty": empty,
            "noise": hnoise,
            "fenemies": fenemies,
            "complete": best_result.getCompletionPercentage(),
            "enemies": max(0, best_result.getKillsTotal() - best_result.getKillsByFall()),
            "coins": best_result.getCurrentCoins(),
            "jumps": best_result.getNumJumps(),
            "actions": actions,
            "locations": locations,
        }

    def quality(self, info):
        tube = get_range_reward(info["tube"], 0, 0, 0, 10)
        empty = get_range_reward(info["empty"], 0, self._empty, 1)
        fenemeis = get_range_reward(info["fenemies"], 0, 0, self._fenemies, 1)
        noise = 0
        if empty >= 1 and tube >= 1:
            for n in info["noise"]:
                noise += get_range_reward(n, 0, 0, 0, 1)
        noise /= 16
        return (tube + empty + noise + fenemeis + 4 * info["complete"]) / 8.0

    def diversity(self, info1, info2):
        total = 0
        visited_1 = np.zeros((info1["height"], info1["width"]))
        for loc in info1["locations"]:
            x, y = max(0, min(15, int(loc[0] / 16))), max(0, min(15, int(loc[1] / 16)))
            visited_1[y][x] += 1
            total += 1
        visited_2 = np.zeros((info2["height"], info2["width"]))
        for loc in info2["locations"]:
            x, y = max(0, min(15, int(loc[0] / 16))), max(0, min(15, int(loc[1] / 16)))
            visited_2[y][x] += 1
            total += 1
        if total == 0:
            return 0.0
        return get_range_reward(abs(visited_1 - visited_2).sum() / total, 0, self._diversity, 1.0)
    
    def controlability(self, info, control):
        enemies = get_range_reward(info["enemies"], 0, max(0, control["enemies"] - self._cerror["enemies"]), control["enemies"] + self._cerror["enemies"], 100)
        jumps = get_range_reward(info["jumps"], 0, max(0, control["jumps"] - self._cerror["jumps"]), control["jumps"] + self._cerror["jumps"], 100)
        coins = get_range_reward(info["coins"], 0, max(0, control["coins"] - self._cerror["coins"]), control["coins"] + self._cerror["coins"], 100)
        return (enemies + jumps + coins) / 3.0
    
    def render(self, content):
        scale = 16
        graphics = {
            # empty locations
            "-": Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),

            # Flag
            "^": Image.open(os.path.dirname(__file__) + "/images/flag_top.png").convert('RGBA'),
            "f": Image.open(os.path.dirname(__file__) + "/images/flag_white.png").convert('RGBA'),
            "I": Image.open(os.path.dirname(__file__) + "/images/flag_middle.png").convert('RGBA'),

            # starting location
            "M": Image.open(os.path.dirname(__file__) + "/images/mario.png").convert('RGBA'),

            # Enemies
            "y": Image.open(os.path.dirname(__file__) + "/images/spiky.png").convert('RGBA'),
            "g": Image.open(os.path.dirname(__file__) + "/images/gomba.png").convert('RGBA'),
            "k": Image.open(os.path.dirname(__file__) + "/images/greenkoopa.png").convert('RGBA'),
            "r": Image.open(os.path.dirname(__file__) + "/images/redkoopa.png").convert('RGBA'),
            
            # solid tiles
            "X": Image.open(os.path.dirname(__file__) + "/images/floor.png").convert('RGBA'),
            "#": Image.open(os.path.dirname(__file__) + "/images/solid.png").convert('RGBA'),

            # Question Mark Blocks
            "Q": Image.open(os.path.dirname(__file__) + "/images/question_coin.png").convert('RGBA'),

            # Brick Blocks
            "S": Image.open(os.path.dirname(__file__) + "/images/brick.png").convert('RGBA'),

            # Coin
            "o": Image.open(os.path.dirname(__file__) + "/images/coin.png").convert('RGBA'),

            # Pipes
            "<": Image.open(os.path.dirname(__file__) + "/images/tubetop_left.png").convert('RGBA'),
            ">": Image.open(os.path.dirname(__file__) + "/images/tubetop_right.png").convert('RGBA'),
            "[": Image.open(os.path.dirname(__file__) + "/images/tube_left.png").convert('RGBA'),
            "]": Image.open(os.path.dirname(__file__) + "/images/tube_right.png").convert('RGBA'),
            "O": Image.open(os.path.dirname(__file__) + "/images/tubetop.png").convert('RGBA'),
            "H": Image.open(os.path.dirname(__file__) + "/images/tube.png").convert('RGBA'),
        }

        levelLines = _convert2str(content, self._symbols).split('\n')
        for i in range(len(levelLines)):
            levelLines[i] = levelLines[i].strip()
            if len(levelLines[i]) == 0:
                levelLines.pop(i)
                i-= 1
        width = len(levelLines[0])
        height = len(levelLines)

        decodedMap = []
        exit_x = -1
        exit_y = -1
        for y in range(height):
            decodedMap.append([])
            for x in range(width):
                char = levelLines[y][x]
                if char == "F":
                    exit_x = x
                    exit_y = y
                    char = "-"
                if char == "t":
                    singlePipe = True
                    topPipe = True
                    if(x < width - 1 and levelLines[y][x+1] == 't') or (x > 0 and levelLines[y][x-1] == 't'):
                        singlePipe = False
                    if y > 0 and levelLines[y-1][x] == 't':
                        topPipe = False
                    if singlePipe:
                        if topPipe:
                            char = "O"
                        else:
                            char = "H"
                    else:
                        if topPipe:
                            char = "<"
                            if x > 0 and levelLines[y][x-1] == 't':
                                char = ">"
                        else:
                            char = "["
                            if x > 0 and levelLines[y][x-1] == 't':
                                char = "]"
                decodedMap[y].append(char)

        if exit_x > 1:
            decodedMap[1][exit_x] = "^"
            decodedMap[2][exit_x - 1] = "f"
        for y in range(2,exit_y+1):
            decodedMap[y][exit_x] = "I"

        lvl_image = Image.new("RGBA", (width*scale, height*scale), (109,143,252,255))
        for y in range(height):
            for tx in range(width):
                x = width - tx - 1
                shift_x = 0
                if decodedMap[y][x] == "f":
                    shift_x = 8
                lvl_image.paste(graphics[decodedMap[y][x]], (x*scale + shift_x, y*scale, (x+1)*scale + shift_x, (y+1)*scale))
        return lvl_image

    def render_solution(self, info, map_name="map"):
        """locations를 따라 매 프레임마다 맵을 동적으로 렌더링하여 저장"""
        locations = info.get("locations", [])
        is_clear = info.get("complete", 0.0) >= 1.0

        if len(locations) == 0:
            return [], is_clear

        scale = 16
        content = info["content"]
        graphics = {
            "-": Image.open(os.path.dirname(__file__) + "/../smb/images/empty.png").convert('RGBA'),
            "^": Image.open(os.path.dirname(__file__) + "/../smb/images/flag_top.png").convert('RGBA'),
            "f": Image.open(os.path.dirname(__file__) + "/../smb/images/flag_white.png").convert('RGBA'),
            "I": Image.open(os.path.dirname(__file__) + "/../smb/images/flag_middle.png").convert('RGBA'),
            "M": Image.open(os.path.dirname(__file__) + "/../smb/images/mario.png").convert('RGBA'),
            "y": Image.open(os.path.dirname(__file__) + "/../smb/images/spiky.png").convert('RGBA'),
            "g": Image.open(os.path.dirname(__file__) + "/../smb/images/gomba.png").convert('RGBA'),
            "k": Image.open(os.path.dirname(__file__) + "/../smb/images/greenkoopa.png").convert('RGBA'),
            "r": Image.open(os.path.dirname(__file__) + "/../smb/images/redkoopa.png").convert('RGBA'),
            "X": Image.open(os.path.dirname(__file__) + "/../smb/images/floor.png").convert('RGBA'),
            "#": Image.open(os.path.dirname(__file__) + "/../smb/images/solid.png").convert('RGBA'),
            "Q": Image.open(os.path.dirname(__file__) + "/../smb/images/question_coin.png").convert('RGBA'),
            "S": Image.open(os.path.dirname(__file__) + "/../smb/images/brick.png").convert('RGBA'),
            "o": Image.open(os.path.dirname(__file__) + "/../smb/images/coin.png").convert('RGBA'),
            "<": Image.open(os.path.dirname(__file__) + "/../smb/images/tubetop_left.png").convert('RGBA'),
            ">": Image.open(os.path.dirname(__file__) + "/../smb/images/tubetop_right.png").convert('RGBA'),
            "[": Image.open(os.path.dirname(__file__) + "/../smb/images/tube_left.png").convert('RGBA'),
            "]": Image.open(os.path.dirname(__file__) + "/../smb/images/tube_right.png").convert('RGBA'),
            "O": Image.open(os.path.dirname(__file__) + "/../smb/images/tubetop.png").convert('RGBA'),
            "H": Image.open(os.path.dirname(__file__) + "/../smb/images/tube.png").convert('RGBA'),
        }

        # 기본 decodedMap 생성 (render 로직과 동일)
        levelLines = _convert2str(content, self._symbols).split('\n')
        levelLines = [l for l in levelLines if len(l) > 0]
        width = len(levelLines[0])
        height = len(levelLines)

        base_map = []
        exit_x, exit_y = -1, -1
        for y in range(height):
            base_map.append([])
            for x in range(width):
                char = levelLines[y][x]
                if char == "F":
                    exit_x, exit_y = x, y
                    char = "-"
                if char == "t":
                    singlePipe = not ((x < width-1 and levelLines[y][x+1] == 't') or (x > 0 and levelLines[y][x-1] == 't'))
                    topPipe = not (y > 0 and levelLines[y-1][x] == 't')
                    if singlePipe:
                        char = "O" if topPipe else "H"
                    else:
                        if topPipe:
                            char = ">" if (x > 0 and levelLines[y][x-1] == 't') else "<"
                        else:
                            char = "]" if (x > 0 and levelLines[y][x-1] == 't') else "["
                base_map[y].append(char)
        if exit_x > 1:
            base_map[1][exit_x] = "^"
            base_map[2][exit_x - 1] = "f"
        for y in range(2, exit_y + 1):
            base_map[y][exit_x] = "I"
        # 원본 마리오 위치(M)를 빈칸으로 제거 — 매 프레임마다 동적으로 표시
        for y in range(height):
            for x in range(width):
                if base_map[y][x] == "M":
                    base_map[y][x] = "-"

        def _render_frame(mario_x, mario_y):
            frame_map = [row[:] for row in base_map]
            mx = max(0, min(width - 1, int(mario_x / scale)))
            my = max(0, min(height - 1, int(mario_y / scale)))
            frame_map[my][mx] = "M"
            img = Image.new("RGBA", (width * scale, height * scale), (109, 143, 252, 255))
            for y in range(height):
                for tx in range(width):
                    x = width - tx - 1
                    ch = frame_map[y][x]
                    shift_x = 8 if ch == "f" else 0
                    img.paste(graphics[ch], (x*scale + shift_x, y*scale, (x+1)*scale + shift_x, (y+1)*scale))
            return img

        frames = [_render_frame(loc[0], loc[1]) for loc in locations]

        import shutil
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "smbtile", map_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(save_dir, f"frame_{i:04d}.png"))

        return frames, is_clear

    def score_frames(self, frames, is_clear):
        """이미지 프레임 리스트를 받아 LLM 점수를 반환 (현재는 -1 고정)"""
        return -1

    def llmscore(self, info, map_name="map"):
        frames, is_clear = self.render_solution(info, map_name=map_name)
        score = self.score_frames(frames, is_clear)
        return score, is_clear

