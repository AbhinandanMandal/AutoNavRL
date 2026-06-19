import argparse
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train import make_env

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained model for robot navigation')
    parser.add_argument('--model-path', type=str, default="models/td3.zip",
                       help='Path to the trained model')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization during evaluation')
    args = parser.parse_args()

    # Load the trained TD3 model
    model = TD3.load(args.model_path)

    # Create the environment with visualization enabled unless disabled
    eval_env = DummyVecEnv([make_env(render=not args.no_render)])

    # Initialize metrics
    success_count = 0
    total_timesteps = 0
    total_distance = 0
    total_velocity = 0
    collision_count = 0
    time_limit_count = 0

    # A* specific metrics
    total_waypoints_planned = 0
    total_waypoints_reached = 0

    # Run evaluation episodes
    for episode in range(args.num_episodes):
        # Get initial state
        obs, _, _, _ = eval_env.step([[0.0, 0.0]])
        done = False
        total_reward = 0

        # Record how many waypoints were planned at the start of this episode
        inner_env = eval_env.envs[0]          # unwrap DummyVecEnv
        n_planned = len(inner_env.sim.waypoints)
        total_waypoints_planned += n_planned

        # Run episode
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, done, info = eval_env.step(action)

            # Accumulate reward
            total_reward += reward[0]  # Get reward from first (and only) environment

        # Update metrics from episode info
        episode_info = info[0]  # Get info from first (and only) environment

        # Count how many waypoints the robot actually advanced through
        n_reached = inner_env.sim.current_waypoint_index
        total_waypoints_reached += n_reached

        if episode_info['success']:
            success_count += 1
        if episode_info['collision']:
            collision_count += 1
        if episode_info['time_limit_reached']:
            time_limit_count += 1
            
        total_timesteps += episode_info['steps']
        total_distance += episode_info['total_distance']
        total_velocity += episode_info['average_velocity']

        print(
            f"Episode {episode + 1:3d} | "
            f"reward={total_reward:7.1f} | "
            f"waypoints {n_reached}/{n_planned} | "
            f"{'SUCCESS' if episode_info['success'] else 'FAIL   '}"
        )

    # Calculate final metrics
    success_rate = (success_count / args.num_episodes) * 100
    collision_rate = (collision_count / args.num_episodes) * 100
    time_limit_rate = (time_limit_count / args.num_episodes) * 100
    avg_timesteps = total_timesteps / args.num_episodes
    avg_distance = total_distance / args.num_episodes
    avg_velocity = total_velocity / args.num_episodes
    avg_waypoints_planned = total_waypoints_planned/args.num_episodes
    avg_waypoints_reached = total_waypoints_reached / args.num_episodes

    # Print evaluation results
    print("\nFinal Evaluation Metrics:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Time Limit Rate: {time_limit_rate:.1f}%")
    print(f"Average Timesteps per Episode: {avg_timesteps:.1f}")
    print(f"Average Distance Traveled: {avg_distance:.2f}")
    print(f"Average Velocity: {avg_velocity:.2f}")
    print(f"Average Waypoints Planned: {avg_waypoints_planned:.2f}")
    print(f"Average Waypoints Reached: {avg_waypoints_reached:.2f}")
    if total_waypoints_planned>0:
        wp_follow_rate = total_waypoints_reached / total_waypoints_planned * 100
        print(f"Waypoint Follow Rate: {wp_follow_rate:.1f}%")
    
