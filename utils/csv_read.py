
def load_csv(filename):
    timesteps = []
    rewards = []
    with open(filename, "r") as f:
        # remove header
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            s = line.split(",")
            timesteps.append(int(s[1]))
            rewards.append(float(s[0]))
            line = f.readline()
    return timesteps, rewards