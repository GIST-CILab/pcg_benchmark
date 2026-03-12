from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from pcg_benchmark.probs.utils import get_number_regions, get_range_reward, get_num_tiles, discretize, get_distance_length, get_normalized_value
from pcg_benchmark.probs.sokoban.engine import State, BFSAgent, AStarAgent
from PIL import Image
import numpy as np
from difflib import SequenceMatcher
import os

def load_level(filepath):
    """
    레벨 파일(engine 포맷: #/@/$/./ )을 읽어 content numpy 배열로 변환.
    # -> 0(solid), ' ' -> 1(empty), @ -> 2(player), $ -> 3(crate), . -> 4(target)
    """
    char_map = {'#': 0, ' ': 1, '@': 2, '$': 3, '.': 4}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if l.strip() != '']
    height = len(lines)
    width = max(len(l) for l in lines)
    arr = np.ones((height, width), dtype=int)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            arr[y][x] = char_map.get(ch, 1)
    return arr

def _sol2str(content, sol):
    if len(sol) == 0:
        return ""
    lvl = np.pad(content, 1)
    gameCharacters="# @$."
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
            if j == lvl.shape[1]-1:
                lvlString += "\n"
    state = State()
    state.stringInitialize(lvlString.split("\n"))
    crateMoves = []
    tempSol = []
    for a in sol:
        crateMoves.append(state.update(a["x"], a["y"]))
        tempSol.append({"x": a["x"], "y": a["y"]})
    sol = tempSol
    if abs(sol[0]["x"]) == 0:
        for a in sol:
            temp = a["x"]
            a["x"] = a["y"]
            a["y"] = temp
    if sol[0]["x"] < 0:
        for a in sol:
            a["x"] *= -1
    flipV = False
    for a in sol:
        if abs(a["y"]) > 0:
            if a["y"] < 0:
                flipV = True
            break
    if flipV:
        for a in sol:
            a["y"] *= -1
    result = ""
    for c,a in zip(crateMoves, sol):
        if not c:
            if a["x"] == 1:
                result += "r"
            if a["x"] == -1:
                result += "l"
            if a["y"] == 1:
                result += "d"
            if a["y"] == -1:
                result += "u"
        else:
            if a["x"] == 1:
                result += "e"
            if a["x"] == -1:
                result += "k"
            if a["y"] == 1:
                result += "s"
            if a["y"] == -1:
                result += "y"
    return result

def _run_game(content, solver_power):
    lvl = np.pad(content, 1)
    gameCharacters="# @$."
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
            if j == lvl.shape[1]-1:
                lvlString += "\n"
    state = State()
    state.stringInitialize(lvlString.split("\n"))
    aStarAgent = AStarAgent()
    bfsAgent = BFSAgent()

    sol,solState,_ = bfsAgent.getSolution(state, solver_power)
    if solState.checkWin():
        return 0, sol, sol
    sol,solState,_ = aStarAgent.getSolution(state, 1, solver_power)
    if solState.checkWin():
        return 0, sol, sol
    sol,solState,_ = aStarAgent.getSolution(state, 0.5, solver_power)
    if solState.checkWin():
        return 0, sol, sol
    sol,solState,_ = aStarAgent.getSolution(state, 0, solver_power)
    if solState.checkWin():
        return 0, sol, sol
    return solState.getHeuristic(), [], sol

class SokobanProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._diff = kwargs.get("difficulty")
        self._power = kwargs.get("solver", 5000)
        self._diversity = kwargs.get("diversity", 0.5)

        self._content_space = ArraySpace((self._width, self._height), IntegerSpace(5))
        self._control_space = DictionarySpace({"crates": IntegerSpace(1, max(self._width+1, self._height+1))})

    def info(self, content):
        content = np.array(content)
        number_player = get_num_tiles(content, [2])
        number_crates = get_num_tiles(content, [3])
        number_targets = get_num_tiles(content, [4])
        heuristic, sol, partial_sol = -1, [], []
        if number_player == 1 and number_crates > 0 and number_crates == number_targets:
            heuristic, sol, partial_sol = _run_game(content, self._power)

        return {
            "players": number_player, "crates": number_crates,
            "targets": number_targets, "content": content,
            "heuristic": heuristic, "solution": sol,
            "partial_solution": partial_sol,
        }

    def quality(self, info):
        player = get_range_reward(info["players"], 0, 1, 1,\
                                    self._width * self._height)
        crates = get_range_reward(info["crates"], 0, 1, self._width * self._height)
        crate_target = get_range_reward(abs(info["crates"] - info["targets"]), 0, 0, 0,\
                                    self._width * self._height)
        stats = (player + crates + crate_target) / 3.0

        added = 0
        if info["heuristic"] >= 0:
            heuristic = get_range_reward(info["heuristic"], 0, 0, 0,\
                                        (self._width + self._height) * info["crates"])
            sol_length = get_range_reward(len(info["solution"]), 0,\
                                        (self._width + self._height) * self._diff,\
                                        (self._width * self._height)**2)
            added = (heuristic + sol_length) / 2.0

        return (stats + added) / 2.0
    
    def diversity(self, info1, info2):
        solution1 = _sol2str(info1["content"], info1["solution"])
        csol1 = ""
        if len(solution1) > 0:
            csol1 = solution1[0]
            for l in solution1[1:]:
                if csol1[-1] == l:
                    continue
                csol1 += l
        solution2 = _sol2str(info2["content"], info2["solution"])
        csol2 = ""
        if len(solution2) > 0:
            csol2 = solution2[0]
            for l in solution2[1:]:
                if csol2[-1] == l:
                    continue
                csol2 += l
        
        ratio = SequenceMatcher(None, csol1, csol2).ratio()
        return get_range_reward(1 - ratio, 0, self._diversity, 1.0)
    
    def controlability(self, info, control):
        crates = get_range_reward(info["crates"], 0, max(control["crates"]-1,1), control["crates"]+1, self._width * self._height)
        return crates

    def render_solution(self, info, map_name="map"):
        """solution(또는 partial_solution)을 따라 시뮬레이션하며 프레임 리스트를 생성하고 저장"""
        sol = info["solution"]
        is_clear = len(sol) > 0
        if not is_clear:
            sol = info.get("partial_solution", [])
        if len(sol) == 0:
            return [], False

        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/solid.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/crate.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/target.png").convert('RGBA'),
        ]

        lvl = np.pad(np.array(info["content"]), 1)
        gameCharacters = "# @$."
        lvlString = ""
        for i in range(lvl.shape[0]):
            for j in range(lvl.shape[1]):
                lvlString += gameCharacters[int(lvl[i][j])]
                if j == lvl.shape[1] - 1:
                    lvlString += "\n"
        state = State()
        state.stringInitialize(lvlString.split("\n"))

        def _render_state(st):
            img = Image.new("RGBA", (st.width * scale, st.height * scale), (0, 0, 0, 255))
            for y in range(st.height):
                for x in range(st.width):
                    if st.solid[y][x]:
                        tile = graphics[0]
                    else:
                        crate = st.checkCrateLocation(x, y) is not None
                        target = st.checkTargetLocation(x, y) is not None
                        player = st.player["x"] == x and st.player["y"] == y
                        if player:
                            tile = graphics[2]
                        elif crate:
                            tile = graphics[3]
                        elif target:
                            tile = graphics[4]
                        else:
                            tile = graphics[1]
                    img.paste(tile, (x * scale, y * scale, (x + 1) * scale, (y + 1) * scale))
            return img

        frames = [_render_state(state)]
        for action in sol:
            state.update(action["x"], action["y"])
            frames.append(_render_state(state))


        import shutil
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "sokoban", map_name)
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

    def render(self, content):
        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/solid.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/crate.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/target.png").convert('RGBA')
        ]
        lvl = np.pad(np.array(content), 1)
        lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
        for y in range(lvl.shape[0]):
            for x in range(lvl.shape[1]):
                lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
        return lvl_image