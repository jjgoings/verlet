# verlet 

`verlet` provides code to perform constant-temperature dynamics for a single particle in one or two dimensions. It aims to provide easy access to performing dynamics with analytic potentials as a way of understanding simple molecular dynamics simulations. 

### Usage

You can try out the sample simulation in the repo

```
python example.py
```

The most important thing to remember is that you must provide a user-defined potential that governs the evolution of the particle dynamics. It is entered as a string, which is then parsed by `sympy` and converted to an analytic expression for the forces.


### Dependencies
You'll need `numpy` and `sympy` (for analytic potential and forces). You can install them all at once if you have `pip`:

```
pip install numpy sympy
```

