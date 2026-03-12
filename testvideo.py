# Testing file for all the problems in the pcg benchmark
import pcg_benchmark
from pcg_benchmark.probs.sokoban.problem import load_level as sokoban_load_level
from pcg_benchmark.probs.smbtile.problem import load_level as smbtile_load_level
from pcg_benchmark.probs.zelda.problem import load_level as zelda_load_level
from pcg_benchmark.probs.talakat.problem import load_level as talakat_load_level
from pcg_benchmark.probs.loderunnertile.problem import load_level as loderunner_load_level
from pcg_benchmark.probs.mdungeons.problem import load_level as mdungeons_load_level
import os

# ── 게임 스위치: 테스트할 게임을 True로 설정 ──────────────────────────────
RUN_SOKOBAN    = False
RUN_SMBTILE    = False
RUN_ZELDA      = False
RUN_TALAKAT    = False
RUN_LODERUNNER = False
RUN_MDUNGEONS  = True
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # --- sokoban-v0 (레퍼런스 맵은 solver_power를 늘려서 실행)
    if RUN_SOKOBAN:
        name = "sokoban-v0"
        env = pcg_benchmark.make(name)
        env._problem._power = 50000  # 레퍼런스 맵 전용으로 depth 확장
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "sokoban")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".txt"))

        infos = []
        for lf in level_files:
            content = sokoban_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info(content)
            infos.append((lf, lvl_info))

        playable = [(lf, i) for lf, i in infos if len(i["solution"]) > 0 or len(i.get("partial_solution", [])) > 0]
        print(f"[sokoban] Loaded {len(infos)} levels, {sum(1 for _, i in infos if len(i['solution']) > 0)} cleared")

        if playable:
            names = [os.path.splitext(lf)[0] for lf, _ in playable]
            l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in playable], map_names=names)
            for idx, (lf, _) in enumerate(playable):
                print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
            print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")
        else:
            print(f"\tLLM Score ({name}): no playable level found")

    # --- smbtile-v0 (레퍼런스 맵은 solver를 늘려서 실행)
    if RUN_SMBTILE:
        name = "smbtile-v0"
        env = pcg_benchmark.make(name)
        env._problem._solver_iterations = 500
        env._problem._sticky_actions = 4
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "smb")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".txt"))

        infos = []
        for lf in level_files:
            content = smbtile_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info(content)
            lvl_info["content"] = content
            infos.append((lf, lvl_info))

        playable = infos
        print(f"\n[smbtile] Loaded {len(infos)} levels, {sum(1 for _, i in infos if i['complete'] >= 1.0)} completed")

        if playable:
            names = [os.path.splitext(lf)[0] for lf, _ in playable]
            l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in playable], map_names=names)
            for idx, (lf, _) in enumerate(playable):
                print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
            print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")
        else:
            print(f"\tLLM Score ({name}): no playable level found")

    # --- zelda-v0 ---
    if RUN_ZELDA:
        name = "zelda-v0"
        env = pcg_benchmark.make(name)
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "zelda")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".txt"))

        infos = []
        for lf in level_files:
            content = zelda_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info(content)
            infos.append((lf, lvl_info))

        playable = [(lf, i) for lf, i in infos if i["player_key"] > 0 and i["key_door"] > 0]
        print(f"\n[zelda] Loaded {len(infos)} levels, {len(playable)} playable")

        if playable:
            names = [os.path.splitext(lf)[0] for lf, _ in playable]
            l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in playable], map_names=names)
            for idx, (lf, _) in enumerate(playable):
                print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
            print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")
        else:
            print(f"\tLLM Score ({name}): no playable level found")

    # --- talakat-v0 ---
    if RUN_TALAKAT:
        name = "talakat-v0"
        env = pcg_benchmark.make(name)
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "talakat")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".json"))

        infos = []
        for lf in level_files:
            script = talakat_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info_from_script(script)
            infos.append((lf, lvl_info))

        print(f"\n[talakat] Loaded {len(infos)} levels, {sum(1 for _, i in infos if i['percentage'] >= 1.0)} completed")

        names = [os.path.splitext(lf)[0] for lf, _ in infos]
        l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in infos], map_names=names)
        for idx, (lf, _) in enumerate(infos):
            print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
        print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")

    # --- loderunnertile-v0 ---
    if RUN_LODERUNNER:
        name = "loderunnertile-v0"
        env = pcg_benchmark.make(name)
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "loderunnertile")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".txt"))

        infos = []
        for lf in level_files:
            content = loderunner_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info(content)
            infos.append((lf, lvl_info))

        playable = [(lf, i) for lf, i in infos if i["player"] == 1 and (i["exploration"] > 0).sum() > 0]
        print(f"\n[loderunnertile] Loaded {len(infos)} levels, {len(playable)} playable")

        if playable:
            names = [os.path.splitext(lf)[0] for lf, _ in playable]
            l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in playable], map_names=names)
            for idx, (lf, _) in enumerate(playable):
                print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
            print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")
        else:
            print(f"\tLLM Score ({name}): no playable level found")

    # --- mdungeons-v0 ---
    if RUN_MDUNGEONS:
        name = "mdungeons-v0"
        env = pcg_benchmark.make(name)
        levels_dir = os.path.join(os.path.dirname(__file__), "reference", "mdungeons")
        level_files = sorted(f for f in os.listdir(levels_dir) if f.endswith(".txt"))

        infos = []
        for lf in level_files:
            content = mdungeons_load_level(os.path.join(levels_dir, lf))
            lvl_info = env._problem.info(content)
            infos.append((lf, lvl_info))
            solved = lvl_info.get("heuristic", -1) == 0
            tc_solved = lvl_info.get("tc_heuristic", -1) == 0
            print(f"  {lf}: solved={solved}, tc_solved={tc_solved}, sol_len={lvl_info.get('solution_length',0)}, tc_sol_len={lvl_info.get('tc_solution_length',0)}")

        playable = [(lf, i) for lf, i in infos if i.get("heuristic", -1) == 0 or i.get("tc_heuristic", -1) == 0]
        print(f"\n[mdungeons] Loaded {len(infos)} levels, {len(playable)} solved")

        if playable:
            names = [os.path.splitext(lf)[0] for lf, _ in playable]
            l_pass, l_scores, l_clears, _ = env.llmscore([i for _, i in playable], map_names=names)
            for idx, (lf, _) in enumerate(playable):
                print(f"\tLLM Score ({name}, {lf}): clear={l_clears[idx]}, score={l_scores[idx]}")
            print(f"\tLLM Score ({name}) overall clear rate: {l_pass}")
        else:
            print(f"\tLLM Score ({name}): no solvable level found")

