from .fastpath import optimize as fastpath_opt

SOLVERS = {
    "fastpath": fastpath_opt
    # add more here
}

def get_solver(name: str):
    try: return SOLVERS[name]
    except KeyError: raise ValueError(f"Unknown solver '{name}'. Options: {list(SOLVERS)}")