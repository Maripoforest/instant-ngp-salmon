import sys
import time

def generate_reward(action):
    # Replace this with your reward generation logic based on the action
    return action * 0.1

if __name__ == "__main__":
    while True:
        # Read action from a.py
        try:
            action = sys.stdin.readline().strip()

            if not action:
                break  # Exit the loop if no more actions are received

            # Generate reward based on the received action
            reward = generate_reward(float(action))
            time.sleep(3)
            # Send the reward back to a.py
            print(reward)
            sys.stdout.flush()  # Ensure the output is flushed to avoid buffering issues

        except Exception as e:
            print(f"Error: {e}")
            break
