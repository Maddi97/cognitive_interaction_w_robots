from environment import Environment

class Agent():
    def __init__(self):
        self.env = Environment(scan_interval=10)

    def main_loop(self):
        for i in range(10):
            print(self.env.get_current_state())


if __name__ == '__main__':
    Agent().main_loop()
