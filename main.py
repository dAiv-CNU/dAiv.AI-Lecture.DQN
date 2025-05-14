import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [normal|genetic|dqn]")
        return

    mode = sys.argv[1].lower()

    if mode == "normal":
        from snake import main as normal_main
        normal_main()
    elif mode == "genetic":
        from genetic.main_genetic import main_genetic
        main_genetic()
    elif mode == "dqn":
        from dqn.main_dqn import main_dqn
        main_dqn()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: normal, genetic, dqn")

if __name__ == "__main__":
    main()