import signal


class SignalHandler:
    def __init__(self):
        self.SIGINT = False
        signal.signal(signal.SIGINT, self.__call__)

    def __call__(self, sig, frame):
        print("Stop requested… will finish current iteration.")
        self.SIGINT = True