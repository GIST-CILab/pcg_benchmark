from pcg_benchmark.probs import Problem
from pcg_benchmark.probs.utils import get_range_reward
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, FloatSpace, DictionarySpace
from pcg_benchmark.probs.talakat.engine import parameters, generateTalakatScript, runPattern
from pcg_benchmark.probs.talakat.engine.helper import calculateBuckets, calculateEntropy
import numpy as np
import json
import math
import shutil
from PIL import Image, ImageDraw
import os

def load_level(filepath):
    """talakat 레퍼런스 JSON 파일을 직접 로드해서 script dict 반환"""
    with open(filepath, 'r') as f:
        return json.load(f)

class TalakatProblem(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._spawnerComplexity = kwargs.get("spawnerComplexity")
        self._maxHealth = kwargs.get("maxHealth")
        self._pattern_sections = max(1, int(self._maxHealth / 30))
        
        self._diversity = kwargs.get("diversity", 0.5)
        self._renderSampling = kwargs.get("renderSampling", 5)
        self._empty_area = kwargs.get("empty", 0.4)
        self._min_bullets = kwargs.get("min_bullets", 50)
        self._target = kwargs.get("coverage", 0.95)

        parameters["maxHealth"] = self._maxHealth
        parameters["width"] = self._width
        parameters["height"] = self._height
        parameters["bucketsX"] = max(1, int(self._width / 20))
        parameters["bucketsY"] = max(1, int(self._height / 20))

        self._cerror = int(0.5 * self._min_bullets)
        self._render_type = "image"

        self._content_space = ArraySpace((self._spawnerComplexity, 100), IntegerSpace(100))
        self._control_space = DictionarySpace({
            "bullets": ArraySpace((self._pattern_sections), IntegerSpace(self._min_bullets+self._cerror, 
                                                                         int(parameters["maxNumBullets"]/2)-self._cerror))
        })
    
    def info(self, content):
        script = generateTalakatScript(content)
        return self._compute_info(script)

    def info_from_script(self, script):
        """레퍼런스 JSON script를 직접 평가"""
        return self._compute_info(script)

    def _compute_info(self, script):
        connections = set()
        nextSpawners = ["spawner_0"]
        while len(nextSpawners) > 0:
            currentSpawner = nextSpawners.pop(0)
            for spawned in script["spawners"][currentSpawner]["pattern"]:
                if "spawner_" in spawned:
                    if not spawned in connections:
                        connections.update([spawned])
                        nextSpawners.append(spawned)
        result = runPattern(script)

        # 실제 결과 길이 기반으로 sections 계산 (레퍼런스 맵이 더 길 수 있음)
        actual_sections = max(1, math.ceil(len(result) / 30))
        bullets = np.zeros((actual_sections, parameters["bucketsX"] * parameters["bucketsY"]))
        num_bullets = [0.0] * actual_sections
        coverage = np.zeros(parameters["bucketsX"] * parameters["bucketsY"])
        for i, (world, _) in enumerate(result):
            temp = np.array(calculateBuckets(self._width, self._height, parameters["bucketsX"], parameters["bucketsY"], world.bullets))
            section = min(int(i / 30), actual_sections - 1)
            bullets[section] += temp / max(1, temp.sum())
            coverage += temp / max(1, temp.sum())
            num_bullets[section] += len(world.bullets)
        return {
            "script": script,
            "script_connectivity": (len(connections) + 1) / self._spawnerComplexity,
            "percentage": len(result) / self._maxHealth,
            "bullets": np.array(num_bullets) / 30,
            "bullet_coverage": calculateEntropy(coverage / self._maxHealth),
            "bullet_locations": bullets / 30,
        }
    
    def quality(self, info):
        playable = info["percentage"]
        coverage = 0.0
        min_bullets = 0.0
        empty = 0.0
        if playable >= 1.0:
            coverage = get_range_reward(info["bullet_coverage"], 0, self._target, 1)
            min_bullets = 0
            for b in info["bullets"]:
                min_bullets += get_range_reward(b, 0, self._min_bullets, parameters["maxNumBullets"], 100 * parameters["maxNumBullets"])
            min_bullets /= len(info["bullets"])
            for locs in info["bullet_locations"]:
                empty += get_range_reward((np.array(locs) == 0).sum() / (parameters["bucketsX"] * parameters["bucketsY"]), 0, self._empty_area, 1)
            empty /= max(1, len(info["bullet_locations"]))
        return (playable + coverage + min_bullets + empty) / 4.0
    
    def diversity(self, info1, info2):
        diversity = []
        length = max(len(info1["bullet_locations"]), len(info2["bullet_locations"]))
        for i in range(length):
            index1 = min(i, len(info1["bullet_locations"]) - 1)
            index2 = min(i, len(info2["bullet_locations"]) - 1)
            diversity.append(abs(info1["bullet_locations"][index1] - info2["bullet_locations"][index2]).sum() / 2.0)
        return get_range_reward(np.array(diversity).min(), 0, self._diversity, 1)
    
    def controlability(self, info, control):
        bulletCoverage = 0
        for v,c in zip(info["bullets"], control["bullets"]):
            bulletCoverage += get_range_reward(v, 0, c - self._cerror, c + self._cerror, parameters["maxNumBullets"])
        return bulletCoverage / len(control["bullets"])
    
    def render(self, content):
        script = generateTalakatScript(content)

        if self._render_type == "string":
            pretty_json = json.dumps(script, indent=2)
            return pretty_json
        
        if self._render_type == "script":
            pretty_json = json.dumps(script, indent=2).split("\n")
            img = Image.new("RGBA", (400, len(pretty_json) * 12 + 16), (71,45,60,255))
            draw = ImageDraw.Draw(img)
            y = 8
            for l in pretty_json:
                x = 8
                for c in l:
                    if c == " ":
                        x += 8
                    else:
                        break
                draw.text((x,y), l.strip(), fill=(207,198,184,255))
                y+=12
            return img

        bossGfx = Image.open(os.path.dirname(__file__) + "/images/boss.png").convert('RGBA')
        result = runPattern(script)
        images = []
        for i in range(0, len(result), self._renderSampling):
            img = Image.new("RGBA", (parameters["width"], parameters["height"]), (71,45,60,255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0,0,img.width,img.height], fill=(71,45,60,255))
            for b in result[i][0].bullets:
                draw.ellipse([b.x - b.radius, b.y - b.radius, b.x + b.radius, b.y + b.radius], fill=(207,198,184,255), outline=(230,72,46,255), width=2)
            img.paste(bossGfx, (int(result[i][0].boss.x - bossGfx.width/2), int(result[i][0].boss.y-bossGfx.height/2),\
                                int(result[i][0].boss.x+bossGfx.width/2), int(result[i][0].boss.y+bossGfx.height/2)), bossGfx)
            images.append(img)
        return images

    def render_solution(self, info, map_name="map"):
        """JSON script를 받아 render와 동일한 방식으로 프레임을 생성하고 videos/talakat/{map_name}/에 저장"""
        script = info["script"]
        is_clear = info.get("percentage", 0.0) >= 1.0

        bossGfx = Image.open(os.path.dirname(__file__) + "/images/boss.png").convert('RGBA')
        result = runPattern(script)

        frames = []
        for i in range(0, len(result), self._renderSampling):
            img = Image.new("RGBA", (parameters["width"], parameters["height"]), (71, 45, 60, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, img.height], fill=(71, 45, 60, 255))
            for b in result[i][0].bullets:
                draw.ellipse(
                    [b.x - b.radius, b.y - b.radius, b.x + b.radius, b.y + b.radius],
                    fill=(207, 198, 184, 255), outline=(230, 72, 46, 255), width=2
                )
            img.paste(bossGfx,
                      (int(result[i][0].boss.x - bossGfx.width / 2),
                       int(result[i][0].boss.y - bossGfx.height / 2),
                       int(result[i][0].boss.x + bossGfx.width / 2),
                       int(result[i][0].boss.y + bossGfx.height / 2)), bossGfx)
            # 플레이어 표시 (흰색 원)
            p = result[i][0].player
            draw.ellipse(
                [p.x - p.radius, p.y - p.radius, p.x + p.radius, p.y + p.radius],
                fill=(255, 255, 255, 255), outline=(100, 200, 255, 255), width=2
            )
            frames.append(img)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        save_dir = os.path.join(project_root, "videos", "talakat", map_name)
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

