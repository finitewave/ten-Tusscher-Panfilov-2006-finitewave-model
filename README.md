## Finitewave model template (replace with the model name)

Implements the ten Tusscher–Panfilov 2006 (TP06) human ventricular ionic model in 2D.

The TP06 model is a detailed biophysical model of the human ventricular 
action potential, designed to simulate realistic electrical behavior in 
tissue including alternans, reentrant waves, and spiral wave breakup.

This model implementation can be used separately from the Finitewave, allowing for standalone simulations and testing of the model dynamics without the need for the entire framework.

### Developer TODO Checklist (template repository only)
- [ ] **Change model entry point in `pyproject.toml`**  
  Update the entry point in the `pyproject.toml` file to reflect the new model's identifier. This ensures that the model can be correctly referenced and utilized within the project.
  
  In pyproject.toml, replace "model_template" with the actual model id. It must match the name of the directory where ops.py is located.
  ```toml
  [project.entry-points."finitewave.models"] 
  model_template = "finitewave_models.model_template"
  ```

- [ ] **Implement `ops.py` model equations**  
  Implement the model equations in the `ops.py` module. This module is the single source of truth for the model equations. Provide pure Python functions with scalar inputs/outputs (no NumPy arrays, no classes, no globals). Do NOT add numba/jax/torch here — the Finitewave runtime will wrap these functions for you. Stimulus and time integration are handled outside of the model. Here you only return time derivatives.

- [ ] **Create 0D implementation for tests and examples**  
  Design a zero-dimensional (0D) implementation of the model that can be used for testing and demonstration purposes. This should simplify the model to its core functionality, making it easier to validate and showcase.

- [ ] **Provide at least one workable example for the model**  
  Create a practical example that demonstrates how to use the model: model initialization, model stimulation and visualization of the model's AP.

- [ ] **Implement model tests**  
  Write unit tests to verify the correctness of the model's implementation: check model attributes and estimate AP parameters (min/max amplitude, duration) - this ensures that the model is excitable and generates the expected AP. 

### Reference
ten Tusscher KH, Panfilov AV. 
Alternans and spiral breakup in a human ventricular tissue model.
Am J Physiol Heart Circ Physiol. 2006 Sep;291(3):H1088–H1100.

DOI: 10.1152/ajpheart.00109.2006

### How to use (quickstart)
```bash
python -m examples.ten_tusscher_panfilov_2006_example
```

### How to test
```bash
python -m pytest -q
```

### Repository structure
```text
.
├── ten_tusscher_panfilov_2006/                # equations package (ops.py)
│   ├── __init__.py
│   └── ops.py                                 # model equations (pure functions)
├── implementation/                            # 0D model implementation
│   ├── __init__.py
│   └── ten_tusscher_panfilov_2006_0d.py
├── example/
│   └── ten_tusscher_panfilov_2006_example.py  # minimal script to run a short trace
├── tests/
│   └── test.py                                # smoke test; reproducibility checks
├── .gitignore
├── LICENSE                                    # MIT
├── pyproject.toml                             # configuration file
└── README.md                                  # this file
```

### Variables
- `u = -84.5` - Membrane potential (mV)
- `cai = 0.00007` - Intracellular calcium concentration (mM)
- `casr = 1.3` - SR calcium concentration (mM)
- `cass = 0.00007` - Subspace calcium concentration (mM)
- `nai = 7.67` - Intracellular sodium concentration (mM)  
- `Ki = 138.3` - Intracellular potassium concentration (mM)  
- `m = 0.0` - Sodium activation gate  
- `h = 0.75` - Sodium inactivation gate  
- `j = 0.75` - Sodium inactivation gate  
- `xr1 = 0.0` - Rapid delayed rectifier potassium activation gate  
- `xr2 = 1.0` - Rapid delayed rectifier potassium activation gate  
- `xs = 0.0` - Slow delayed rectifier potassium activation gate  
- `r = 0.0` - Transient outward potassium activation gate  
- `s = 1.0` - Transient outward potassium inactivation gate  
- `d = 0.0` - L-type calcium channel activation gate  
- `f = 1.0` - L-type calcium channel inactivation gate  
- `f2 = 1.0` - L-type calcium channel inactivation gate  
- `fcass = 1.0` - Calcium release inactivation gate  
- `rr = 1.0` - Ryanodine receptor activation gate  
- `oo = 0.0` - Ryanodine receptor open probability  

### Parameters

self.ko  = 5.4     # Potassium extracellular concentration
self.cao = 2.0     # Calcium extracellular concentration
self.nao = 140.0   # Sodium extracellular concentration

# Cell Volume (in uL)
self.Vc  = 0.016404   # Cytoplasmic volume
self.Vsr = 0.001094   # Sarcoplasmic reticulum volume
self.Vss = 0.00005468 # Subsarcolemmal space volume

# Buffering Parameters
self.Bufc   = 0.2     # Cytoplasmic buffer concentration
self.Kbufc  = 0.001   # Cytoplasmic buffer affinity
self.Bufsr  = 10.0    # SR buffer concentration
self.Kbufsr = 0.3     # SR buffer affinity
self.Bufss  = 0.4     # Subsarcolemmal buffer concentration
self.Kbufss = 0.00025 # Subsarcolemmal buffer affinity

# Calcium Handling Parameters
self.Vmaxup = 0.006375  # Maximal calcium uptake rate
self.Kup    = 0.00025   # Calcium uptake affinity
self.Vrel   = 0.102     # Calcium release rate from SR
self.k1_    = 0.15      # Transition rate for SR calcium release
self.k2_    = 0.045
self.k3     = 0.060
self.k4     = 0.005      # Alternative transition rate
self.EC     = 1.5        # Calcium-induced calcium release sensitivity
self.maxsr  = 2.5        # Maximum SR calcium release permeability
self.minsr  = 1.0        # Minimum SR calcium release permeability
self.Vleak  = 0.00036    # SR calcium leak rate
self.Vxfer  = 0.0038     # Calcium transfer rate from subspace to cytosol

# Physical Constants
self.R     = 8314.472   # Universal gas constant (J/(kmol·K))
self.F     = 96485.3415 # Faraday constant (C/mol)
self.T     = 310.0      # Temperature (Kelvin, 37°C)
self.RTONF = 26.71376   # RT/F constant for Nernst equation

# Membrane Capacitance
self.CAPACITANCE = 0.185 # Membrane capacitance (μF/cm²)

# Ion Channel Conductances
self.gkr  = 0.153       # Rapid delayed rectifier K+ conductance
self.gks  = 0.392       # Slow delayed rectifier K+ conductance
self.gk1  = 5.405       # Inward rectifier K+ conductance
self.gto  = 0.294       # Transient outward K+ conductance
self.gna  = 14.838      # Fast Na+ conductance
self.gbna = 0.00029     # Background Na+ conductance
self.gcal = 0.00003980  # L-type Ca2+ channel conductance
self.gbca = 0.000592    # Background Ca2+ conductance
self.gpca = 0.1238      # Sarcolemmal Ca2+ pump current conductance
self.KpCa = 0.0005      # Sarcolemmal Ca2+ pump affinity
self.gpk  = 0.0146      # Na+/K+ pump current conductance

# Na+/K+ Pump Parameters
self.pKNa = 0.03        # Na+/K+ permeability ratio
self.KmK  = 1.0         # Half-saturation for K+ activation
self.KmNa = 40.0        # Half-saturation for Na+ activation
self.knak = 2.724       # Maximal Na+/K+ pump rate

# Na+/Ca2+ Exchanger Parameters
self.knaca = 1000       # Maximal Na+/Ca2+ exchanger current
self.KmNai = 87.5       # Half-saturation for Na+ binding
self.KmCa  = 1.38       # Half-saturation for Ca2+ binding
self.ksat  = 0.1        # Saturation factor
self.n_   = 0.35        # Exponent for Na+ dependence

- `ko = 5.4` - Potassium extracellular concentration
- `cao = 2.0` - Calcium extracellular concentration
- `nao = 140.0` - Sodium extracellular concentration

- `Vc = 0.016404` - Cytoplasmic volume (in uL)  
- `Vsr = 0.001094` - Sarcoplasmic reticulum volume  
- `Vss = 0.00005468` - Subsarcolemmal space volume  
- `Bufc = 0.2` - Cytoplasmic buffer concentration  
- `Kbufc = 0.001` - Cytoplasmic buffer affinity  
- `Bufsr = 10.0` - SR buffer concentration  
- `Kbufsr = 0.3` - SR buffer affinity  
- `Bufss = 0.4` - Subsarcolemmal buffer concentration  
- `Kbufss = 0.00025` - Subsarcolemmal buffer affinity  
- `Vmaxup = 0.006375` - Maximal calcium uptake rate  
- `Kup = 0.00025` - Calcium uptake affinity  
- `Vrel = 0.102` - Calcium release rate from SR  
- `k1_ = 0.15` - Transition rate for SR calcium release  
- `k2_ = 0.045` - Transition rate for SR calcium release  
- `k3 = 0.060` - Transition rate for SR calcium release  
- `k4_ = 0.005` - Alternative transition rate  
- `EC = 1.5` - Calcium-induced calcium release sensitivity  
- `maxsr = 2.5` - Maximum SR calcium release permeability  
- `minsr = 1.0` - Minimum SR calcium release permeability  
- `Vleak = 0.00036` - SR calcium leak rate  
- `Vxfer = 0.0038` - Calcium transfer rate from subspace to cytosol  
- `R = 8314.472` - Universal gas constant (J/(kmol·K))  
- `F = 96485.3415` - Faraday constant (C/mol)  
- `T = 310.0` - Temperature (Kelvin, 37°C)  
- `RTONF = 26.71376` - RT/F constant for Nernst equation  
- `CAPACITANCE = 0.185` - Membrane capacitance (μF/cm²)
- `gkr  = 0.153` - Rapid delayed rectifier K+ conductance
- `gks  = 0.392` - Slow delayed rectifier K+ conductance
- `gk1  = 5.405` - Inward rectifier K+ conductance
- `gto  = 0.294` - Transient outward K+ conductance
- `gna  = 14.838` - Fast Na+ conductance
- `gbna = 0.00029` - Background Na+ conductance
- `gcal = 0.00003980` - L-type Ca2+ channel conductance
- `gbca = 0.000592` - Background Ca2+ conductance
- `gpca = 0.1238` - Sarcolemmal Ca2+ pump current conductance
- `KpCa = 0.0005` - Sarcolemmal Ca2+ pump affinity
- `gpk  = 0.0146` - Na+/K+ pump current conductance
- `pKNa = 0.03` - Na+/K+ permeability ratio
- `KmK  = 1.0` - Half-saturation for K+ activation
- `KmNa = 40.0` - Half-saturation for Na+ activation
- `knak = 2.724` - Maximal Na+/K+ pump rate
- `knaca = 1000` - Maximal Na+/Ca2+ exchanger current
- `KmNai = 87.5` - Half-saturation for Na+ binding
- `KmCa  = 1.38` - Half-saturation for Ca2+ binding
- `ksat  = 0.1` - Saturation factor
- `n_   = 0.35` - Exponent for Na+ dependence

