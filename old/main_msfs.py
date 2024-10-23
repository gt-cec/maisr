import pygame
from env import MAISREnv
from msfs_integration import MSFSConnector
import time
""" README: This has been incorporated into main.py using the use_msfs boolean config parameter."""

def run_game_with_msfs(env_config):
    # Initialize both environments (unchanged from main)
    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((env_config["window size"][0], env_config["window size"][1]))
    env = MAISREnv(env_config, window, clock=clock, render=True)

    # Initialize MSFS connection # TODO this is new
    msfs = MSFSConnector()
    if not msfs.spawn_ai_aircraft():
        print("Failed to initialize MSFS integration")
        return

    game_count = 0
    agent0_id = env.num_ships
    agent1_id = env.num_ships + 1

    try:
        while True:
            game_count += 1
            state = env.reset()
            done = False

            while not done:
                actions = []

                # Get AI aircraft (agent0) action
                agent0_action, _ = agent0_policy(env, env.aircraft_ids[0], **kwargs)
                actions.append((env.aircraft_ids[0], agent0_action))

                # Update AI aircraft in MSFS
                if agent0_action is not None:
                    msfs.update_ai_aircraft( # TODO this is new
                        agent0_action[0],  # x coordinate
                        agent0_action[1],  # y coordinate
                        env.agents[agent0_id].direction * (180 / math.pi),  # convert radians to degrees
                        env_config["gameboard size"]
                    )

                # Get player position from MSFS # TODO this is new
                player_x, player_y, player_heading = msfs.get_player_position(env_config["gameboard size"])
                if player_x is not None and player_y is not None:
                    agent1_action = (player_x, player_y)
                    actions.append((env.aircraft_ids[1], agent1_action))

                # Handle other events # TODO this is new
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        msfs.cleanup()
                        pygame.quit()
                        return

                # Update game state
                state, reward, done, _ = env.step(actions)
                env.render()

                # Maintain frame rate
                clock.tick(60) # TODO this is new

            print("Game complete:", game_count)

    finally: # TODO this is new
        msfs.cleanup()
        pygame.quit()


if __name__ == "__main__":
    # Your existing env_config here
    env_config = {
        "gameboard size": 700,
        "num aircraft": 2,
        "gameplay color": "white",
        "gameboard border margin": 35,
        "targets iteration": "C",
        "motion iteration": "F",
        "search pattern": "ladder",
        "verbose": False,
        "window size": (1500, 850)
    }

    run_game_with_msfs(env_config)