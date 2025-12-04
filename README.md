# PLsim

Warning: PLsim is currently undergoing structural changes!

[Documentation webpage](https://YooJung-Kim.github.io/PLsim/)

PLsim is a python package for simulating photonic lantern (+ photonic integrated circuit) observables for astronomical scenes.

* For numerical simulation studies on using photonic lanterns for spectroastrometric measurements, please refer to [Kim et al. JATIS 10, 045001](https://ui.adsabs.harvard.edu/abs/2024JATIS..10d5001K/abstract).

* For laboratory spectral characterizations of standard 3-port photonic lanterns, please refer to [Kim et al. JATIS 10, 045004](https://ui.adsabs.harvard.edu/abs/2024JATIS..10d5004K/abstract). PLsim relies on the mathematical description of PL models defined in this paper; please cite this reference if you use PLsim for your research.

The codes for photonic integrated circuit are currently under development for open-use!

## Installation

`PLsim` uses [`lightbeam`](https://github.com/jw-lin/lightbeam) for calculation of LP modes.
```bash
pip install git+https://github.com/jw-lin/lightbeam.git
```

Then install `PLsim`.

```bash
pip install git+https://github.com/YooJung-Kim/PLsim.git
```

## Tutorials

Please see [tutorials.ipynb](./tutorials/tutorials.ipynb) for detailed examples and usage instructions.

* For actual photonic lantern data reduction, please check out the package [PLred](https://github.com/YooJung-Kim/PLred).

## Contact

**Yoo Jung Kim**  
Email: yjkim@astro.ucla.edu  


