from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, DictionarySpace, IntegerSpace
from pcg_benchmark.probs.utils import get_number_regions, get_num_tiles, _get_certain_tiles, get_range_reward
from pcg_benchmark.probs.mdungeons.engine import State, BFSAgent, AStarAgent, TreasureCollectorState
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import os
import shutil

def load_level(filepath):
    """mdungeons 레퍼런스 맵 파일(숫자 포맷)을 읽어 content numpy 배열로 반환.
    0=wall, 1=empty, 2=player, 3=exit, 4=potion, 5=treasure, 6=goblin, 7=ogre
    """
    level = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                level.append([int(c) for c in line])
    return np.array(level)

def _get_solution_sequence(content, sol):
    lvl = np.pad(content, 1)
    gameCharacters="# @H*$go"
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
            if j == lvl.shape[1]-1:
                lvlString += "\n"
    state = State()
    state.stringInitialize(lvlString.split("\n"))

    result = ""
    for a in sol:
        result += state.update(a["action"]["x"], a["action"]["y"])
    return result

def _run_game_treasure_collector(content, solver_power):
    """TreasureCollectorState 휴리스틱으로 보물/적 최대 수집 + 최단 경로 탐색"""
    lvl = np.pad(content, 1)
    gameCharacters = "# @H*$go"
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
        lvlString += "\n"

    state = TreasureCollectorState()
    state.stringInitialize(lvlString.split("\n"))

    aStarAgent = AStarAgent()
    bfsAgent = BFSAgent()

    for balance in [1, 0.5, 0]:
        sol, solState, _ = aStarAgent.getSolution(state, balance, solver_power)
        if solState.checkWin():
            return 0, sol, solState.getGameStatus()
    sol, solState, _ = bfsAgent.getSolution(state, solver_power)
    if solState.checkWin():
        return 0, sol, solState.getGameStatus()

    return solState.getHeuristic(), [], solState.getGameStatus()


def _run_game(content, solver_power):
    lvl = np.pad(content, 1)
    gameCharacters="# @H*$go"
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

    sol,solState,_ = aStarAgent.getSolution(state, 1, solver_power)
    if solState.checkWin():
        return 0, sol, solState.getGameStatus()
    sol,solState,_ = aStarAgent.getSolution(state, 0.5, solver_power)
    if solState.checkWin():
        return 0, sol, solState.getGameStatus()
    sol,solState,_ = aStarAgent.getSolution(state, 0, solver_power)
    if solState.checkWin():
        return 0, sol, solState.getGameStatus()
    sol,solState,_ = bfsAgent.getSolution(state, solver_power)
    if solState.checkWin():
        return 0, sol, solState.getGameStatus()

    return solState.getHeuristic(), [], solState.getGameStatus()

class MiniDungeonProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._enemies = kwargs.get("enemies")
        self._solver = kwargs.get("solver", 200000)
        self._tc_solver = kwargs.get("tc_solver", 500000)
        self._diversity = kwargs.get("diversity", 0.4)

        self._content_space = ArraySpace((self._height, self._width), IntegerSpace(8))
        self._control_space = DictionarySpace({"col_treasures": IntegerSpace(self._width + self._height), 
                                               "solution_length": IntegerSpace(2*self._enemies, int(self._width * self._height / 2))})
        self._cerror = {"col_treasures": max(0.2 * (self._width + self._height), 1), 
                        "solution_length": max(0.5 * self._enemies, 1)}

    def info(self, content):
        content = np.array(content)

        regions = get_number_regions(content, [1, 2, 3, 4, 5, 6, 7])
        players = _get_certain_tiles(content, [2])
        layout = get_num_tiles(content, [0,1])
        enemies = _get_certain_tiles(content, [6, 7])
        exits = get_num_tiles(content, [3])
        potions = get_num_tiles(content, [4])
        treasures = get_num_tiles(content, [5])

        heuristic, solution, stats = -1, [], {}
        tc_heuristic, tc_solution, tc_stats = -1, [], {}
        if regions == 1 and len(players) == 1 and exits == 1:
            heuristic, solution, stats = _run_game(content, self._solver)
            tc_heuristic, tc_solution, tc_stats = _run_game_treasure_collector(content, self._tc_solver)
        result = {
            "regions": regions, "players": len(players), "exits": exits, "layout": layout,
            "heuristic": heuristic, "solution": solution, "content": content,
            "potions": potions, "treasures": treasures, "enemies": len(enemies),
            "solution_length": len(solution), "enemies_loc": enemies,
            "tc_heuristic": tc_heuristic, "tc_solution": tc_solution,
            "tc_solution_length": len(tc_solution),
        }
        for name in stats:
            result[name] = stats[name]
        return result
    
    def quality(self, info):
        regions = get_range_reward(info["regions"], 0, 1, 1, self._width * self._height / 10)
        player = get_range_reward(info["players"], 0, 1, 1, self._width * self._height)
        exit = get_range_reward(info["exits"], 0, 1, 1, self._width * self._height)
        enemies = get_range_reward(info["enemies"], 0, self._enemies, self._width * self._height)
        layout = get_range_reward(info["layout"], 0, self._width * self._height / 2, self._width * self._height)
        stats = (player + exit + regions + enemies + layout) / 5.0
        solution, enemies = 0, 0
        if player == 1 and exit == 1 and regions == 1:
            solution += get_range_reward(info["heuristic"],0,0,0,(self._width * self._height)**2)
            solution += get_range_reward(len(info["solution"]), 0, self._enemies + 2, (self._width * self._height)**2)
            solution /= 2.0
            
            dist_enemies = []
            for e in info["enemies_loc"]:
                distances = []
                for l in info["solution"]:
                    distances.append(abs(l["x"]-e[0]-1) + abs(l["y"]-e[1]-1)) 
                if len(distances) > 0:   
                    dist_enemies.append(min(distances))
                else:
                    dist_enemies.append(self._width + self._height)
            dist_enemies.sort()
            dist_enemies = dist_enemies[0:self._enemies]
            enemies += get_range_reward(sum(dist_enemies), 0, 0, 0, self._enemies * (self._width + self._height))
            # added += get_range_reward(info["col_enemies"], 0, self._enemies, self._width * self._height)
            # added /= 2.0
        return (stats + solution + enemies) / 3.0
    
    def diversity(self, info1, info2):
        seq1 = _get_solution_sequence(info1["content"], info1["solution"])
        seq2 = _get_solution_sequence(info2["content"], info2["solution"])
        hamming = (abs(info1["content"] - info2["content"]) > 0).sum() / (self._width * self._height)
        seq_score = 1 - SequenceMatcher(None, seq1, seq2).ratio()
        return get_range_reward(seq_score * 0.8 + hamming * 0.2, 0, self._diversity, 1.0)
                
    
    def controlability(self, info, control):
        if info["heuristic"] == -1:
            return 0.0
        treasures = get_range_reward(info["col_treasures"], 0, control["col_treasures"] - self._cerror["col_treasures"], control["col_treasures"] + self._cerror["col_treasures"])
        sol_length = get_range_reward(info["solution_length"], 0, control["solution_length"] - self._cerror["solution_length"], control["solution_length"] + self._cerror["solution_length"])
        return (treasures + sol_length) / 2.0
    
    def render(self, content):
        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/solid.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/exit.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/potion.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/treasure.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/goblin.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/ogre.png").convert('RGBA'),
        ]
        lvl = np.pad(np.array(content), 1)
        lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
        for y in range(lvl.shape[0]):
            for x in range(lvl.shape[1]):
                lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
        return lvl_image

    def render_solution(self, info, map_name="map",
                        show_ui=True, treasure_score=100, enemy_score=50):
        """_run_game의 solution 액션 리스트를 따라 State를 재현해 동적 렌더링"""
        content = np.array(info["content"])
        # tc_solution(treasure collector) 우선, 없으면 일반 solution
        solution = info.get("tc_solution") or info.get("solution", [])
        tc_solved = info.get("tc_heuristic", -1) == 0
        is_clear = tc_solved or (info.get("heuristic", -1) == 0 and len(info.get("solution", [])) > 0)

        scale = 16
        img_dir = os.path.dirname(__file__) + "/images/"
        graphics = [
            Image.open(img_dir + "solid.png").convert('RGBA'),    # 0: wall
            Image.open(img_dir + "empty.png").convert('RGBA'),    # 1: empty
            Image.open(img_dir + "player.png").convert('RGBA'),   # 2: player
            Image.open(img_dir + "exit.png").convert('RGBA'),     # 3: exit
            Image.open(img_dir + "potion.png").convert('RGBA'),   # 4: potion
            Image.open(img_dir + "treasure.png").convert('RGBA'), # 5: treasure
            Image.open(img_dir + "goblin.png").convert('RGBA'),   # 6: goblin
            Image.open(img_dir + "ogre.png").convert('RGBA'),     # 7: ogre
        ]

        # gameCharacters="# @H*$go" 로 lvlString 생성
        gameCharacters = "# @H*$go"
        lvl = np.pad(content, 1)
        lvlString = ""
        for i in range(lvl.shape[0]):
            for j in range(lvl.shape[1]):
                lvlString += gameCharacters[int(lvl[i][j])]
            lvlString += "\n"

        from PIL import ImageDraw, ImageFont

        use_tc = bool(info.get("tc_solution")) and info.get("tc_heuristic", -1) == 0
        StateClass = TreasureCollectorState if use_tc else State

        # 초기 보물/적 수 기억 → 점수 계산 기준
        _init_state_tmp = StateClass()
        _init_state_tmp.stringInitialize(lvlString.split("\n"))
        init_treasure_count = len(_init_state_tmp.treasures)
        init_enemy_count    = len(_init_state_tmp.enemies)

        def _calc_score(state):
            collected_t = init_treasure_count - len(state.treasures)
            collected_e = init_enemy_count    - len(state.enemies)
            return collected_t * treasure_score + collected_e * enemy_score

        def _state_to_frame(state):
            """현재 State 객체에서 프레임 이미지 생성 (하단 HP/점수 UI 포함)"""
            h, w = content.shape
            arr = np.zeros((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    arr[y][x] = 0 if state.solid[y+1][x+1] else 1
            if state.door:
                dx, dy = state.door["x"]-1, state.door["y"]-1
                if 0 <= dy < h and 0 <= dx < w:
                    arr[dy][dx] = 3
            for p in state.potions:
                px, py = p["x"]-1, p["y"]-1
                if 0 <= py < h and 0 <= px < w:
                    arr[py][px] = 4
            for t in state.treasures:
                tx, ty = t["x"]-1, t["y"]-1
                if 0 <= ty < h and 0 <= tx < w:
                    arr[ty][tx] = 5
            for e in state.enemies:
                ex, ey = e["x"]-1, e["y"]-1
                if 0 <= ey < h and 0 <= ex < w:
                    arr[ey][ex] = 6 if e["damage"] == 1 else 7
            if state.player:
                plx, ply = state.player["x"]-1, state.player["y"]-1
                if 0 <= ply < h and 0 <= plx < w:
                    arr[ply][plx] = 2

            padded = np.pad(arr, 1)
            map_w = padded.shape[1] * scale
            map_h = padded.shape[0] * scale

            ui_h = 20 if show_ui else 0
            img = Image.new("RGBA", (map_w, map_h + ui_h), (0, 0, 0, 255))

            for y in range(padded.shape[0]):
                for x in range(padded.shape[1]):
                    img.paste(graphics[padded[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))

            if not show_ui:
                return img

            draw = ImageDraw.Draw(img)
            ui_y  = map_h
            max_hp = 5
            hp    = state.player["health"] if state.player else 0
            score = _calc_score(state)

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            except Exception:
                font = ImageFont.load_default()

            # HP 바 (왼쪽 절반)
            bar_x, bar_w, bar_h = 4, map_w // 2 - 8, 10
            bar_y = ui_y + (ui_h - bar_h) // 2
            draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(80, 0, 0, 255))
            filled_w = int(bar_w * hp / max_hp)
            if filled_w > 0:
                draw.rectangle([bar_x, bar_y, bar_x + filled_w, bar_y + bar_h], fill=(220, 40, 40, 255))
            draw.text((bar_x + 2, bar_y), f"HP {hp}/{max_hp}", fill=(255, 255, 255, 255), font=font)

            # 점수 (오른쪽 절반)
            score_x = map_w // 2 + 4
            draw.text((score_x, bar_y), f"SCORE: {score}", fill=(255, 215, 0, 255), font=font)

            return img

        state = StateClass()
        state.stringInitialize(lvlString.split("\n"))

        frames = [_state_to_frame(state)]
        for a in solution:
            state.update(a["action"]["x"], a["action"]["y"])
            frames.append(_state_to_frame(state))
            if state.checkOver():
                break

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "mdungeons", map_name)
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

