from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from pcg_benchmark.probs.utils import get_number_regions, get_range_reward, get_num_tiles, get_distance_length, get_path
from difflib import SequenceMatcher
import numpy as np
from PIL import Image
import os
import shutil

def load_level(filepath):
    """
    zelda 레퍼런스 맵 파일을 읽어 content numpy 배열로 변환.
    w->0(wall), .->1(empty), @->2(player), +->3(key), g->4(door), e->5(enemy)
    """
    char_map = {'w': 0, '.': 1, '@': 2, '+': 3, 'g': 4, 'e': 5}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if l.strip() != '']
    height = len(lines)
    width = max(len(l) for l in lines)
    arr = np.zeros((height, width), dtype=int)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            arr[y][x] = char_map.get(ch, 1)
    return arr

class ZeldaProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._enemies = kwargs.get("enemies")
        self._diversity = kwargs.get("diversity", 0.3)
        self._erange = max(int(self._enemies * 0.25), 1)

        self._target = kwargs.get("sol_legnth", self._width + self._height)
        self._cerror = max(int(self._target / 2 * 0.25), 1)

        self._content_space = ArraySpace((self._height, self._width), IntegerSpace(6))
        self._control_space = DictionarySpace({
            "player_key": IntegerSpace(int(self._target / 2 + self._cerror), int(self._width * self._height / 4)),
            "key_door": IntegerSpace(int(self._target / 2 + self._cerror), int(self._width * self._height / 4))
        })

    def info(self, content):
        content = np.array(content)
        number_regions = get_number_regions(content, [1, 2, 3, 4, 5])
        number_player = get_num_tiles(content, [2])
        number_key = get_num_tiles(content, [3])
        number_door = get_num_tiles(content, [4])
        number_enemies = get_num_tiles(content, [5])
        player_key = get_distance_length(content, [2], [3], [1, 2, 3, 5])
        pk_path = get_path(content, [2], [3], [1, 2, 3, 5])
        key_door = get_distance_length(content, [3], [4], [1, 2, 3, 4, 5])
        kd_path = get_path(content, [3], [4], [1, 2, 3, 4, 5])

        return {
            "regions": number_regions, "players": number_player,
            "keys": number_key, "doors": number_door, "enemies": number_enemies,
            "player_key": player_key, "key_door": key_door,
            "pk_path": pk_path, "kd_path": kd_path,
            "content": content,
        }

    def quality(self, info):
        regions = get_range_reward(info["regions"], 0, 1, 1, self._width * self._height / 10)

        player = get_range_reward(info["players"], 0, 1, 1, self._width * self._height)
        key = get_range_reward(info["keys"], 0, 1, 1, self._width * self._height)
        door = get_range_reward(info["doors"], 0, 1, 1, self._width * self._height)
        enemies = get_range_reward(info["enemies"], 0, self._enemies - self._erange, \
                                   self._enemies + self._erange, self._width * self._height)
        stats = (player + key + door + enemies) / 4.0

        added = 0
        if player >= 1 and key >= 1 and door >= 1:
            playable = 0
            if info["player_key"] > 0:
                playable += 1.0
            if info["key_door"] > 0:
                playable += 1.0
            playable /= 2.0
            added += playable
            if playable == 1:
                sol_length = get_range_reward(info["player_key"] + info["key_door"], 0, self._target,\
                                       self._width * self._height)
                added += sol_length
        return (regions + stats + added) / 4.0

    def diversity(self, info1, info2):
        path1 = info1["pk_path"] + info1["kd_path"]
        new_path1 = ""
        for x,y in path1:
            if path1[0][0] > self._width / 2:
                x = self._width - x - 1
            if path1[0][1] > self._height / 2:
                y = self._height - y - 1
            new_path1 += f"{x},{y}|"

        path2 = info2["pk_path"] + info2["kd_path"]
        new_path2 = ""
        for x,y in path2:
            if path2[0][0] > self._width / 2:
                x = self._width - x - 1
            if path2[0][1] > self._height / 2:
                y = self._height - y - 1
            new_path2 += f"{x},{y}|"
        ratio = SequenceMatcher(None, new_path1, new_path2).ratio()
        return get_range_reward(1 - ratio, 0, self._diversity, 1.0)
    
    def controlability(self, info, control):
        player_key = get_range_reward(info["player_key"], 0, control["player_key"]-self._cerror, control["player_key"]+self._cerror, int(self._width * self._height / 4))
        key_door = get_range_reward(info["key_door"], 0, control["key_door"]-self._cerror, control["key_door"]+self._cerror, int(self._width * self._height / 4))
        return (player_key + key_door) / 2
    
    def render(self, content):
        scale = 16
        graphics = [
            Image.open(os.path.dirname(__file__) + "/images/solid.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/empty.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/player.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/key.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/door.png").convert('RGBA'),
            Image.open(os.path.dirname(__file__) + "/images/bat.png").convert('RGBA'),
        ]
        lvl = np.pad(np.array(content), 1)
        lvl_image = Image.new("RGBA", (lvl.shape[1]*scale, lvl.shape[0]*scale), (0,0,0,255))
        for y in range(lvl.shape[0]):
            for x in range(lvl.shape[1]):
                lvl_image.paste(graphics[lvl[y][x]], (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
        return lvl_image

    def render_solution(self, info, map_name="map"):
        """pk_path + kd_path를 따라 맵 상태를 동적으로 렌더링한 프레임 리스트를 생성하고 저장"""
        pk_path = info.get("pk_path", [])
        kd_path = info.get("kd_path", [])
        is_clear = info.get("player_key", 0) > 0 and info.get("key_door", 0) > 0

        if len(pk_path) == 0 and len(kd_path) == 0:
            return [], is_clear

        scale = 16
        content = np.array(info["content"], dtype=int)
        img_dir = os.path.dirname(__file__) + "/images/"
        graphics = [
            Image.open(img_dir + "solid.png").convert('RGBA'),
            Image.open(img_dir + "empty.png").convert('RGBA'),
            Image.open(img_dir + "player.png").convert('RGBA'),
            Image.open(img_dir + "key.png").convert('RGBA'),
            Image.open(img_dir + "door.png").convert('RGBA'),
            Image.open(img_dir + "bat.png").convert('RGBA'),
        ]

        def _render_map(lvl):
            padded = np.pad(lvl, 1)
            img = Image.new("RGBA", (padded.shape[1] * scale, padded.shape[0] * scale), (0, 0, 0, 255))
            for y in range(padded.shape[0]):
                for x in range(padded.shape[1]):
                    img.paste(graphics[padded[y][x]], (x * scale, y * scale, (x + 1) * scale, (y + 1) * scale))
            return img

        # 원본에서 플레이어(2) 제거한 베이스 맵 (플레이어 위치는 매 프레임 동적으로 추가)
        base = content.copy()
        base[base == 2] = 1  # 원본 플레이어 위치를 빈칸으로

        frames = []

        # --- pk_path: 키가 있는 상태에서 플레이어 이동 ---
        for (x, y) in pk_path:
            state = base.copy()
            state[y][x] = 2
            frames.append(_render_map(state))

        # 키 획득: 키(3) → 빈칸(1)
        base[base == 3] = 1

        # --- kd_path: 키 사라진 상태에서 문까지 이동 ---
        for (x, y) in kd_path:
            state = base.copy()
            state[y][x] = 2
            frames.append(_render_map(state))

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "zelda", map_name)
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

