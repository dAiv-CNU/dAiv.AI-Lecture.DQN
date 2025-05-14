import sys


if len(sys.argv) < 2:
    print("Usage: python main.py [normal|genetic|dqn]")
    sys.exit(1)

mode = sys.argv[1].lower()

if mode == "normal":
    from .normal import normal_main
    normal_main()
elif mode == "genetic":
    from .genetic import genetic_main
    genetic_main()
elif mode == "dqn":
    from .dqn import dqn_main
    dqn_main()
else:
    print(f"Unknown mode: {mode}")
    print("Available modes: normal, genetic, dqn")
