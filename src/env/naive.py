import random
from MAMEToolkit.sf_environment import Environment

roms_path = "/home/sujinhua/app/roms"  # Replace this with the path to your ROMs
env = Environment("env1", roms_path)
env.start()
while True:
    move_action = random.randint(0, 8)
    attack_action = random.randint(0, 9)
    frames, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
    if game_done:
        env.new_game()
    elif stage_done:
        env.next_stage()
    elif round_done:
        env.next_round()
